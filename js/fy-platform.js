/* =====================================================================
   fy-platform.js  —  Four You Consultants platform layer  (v1)
   ---------------------------------------------------------------------
   FY.auth      Username/password sign-in screen, server-side lockout
                (10 tries -> 30 min) recorded in the security log.
   FY.sync      Cross-device sync (Supabase Realtime) + multi-tab sync
                (BroadcastChannel). The database is the source of truth.
   FY.state     UI state (active client, view, range, drafts, scroll)
                saved per user, restored on any device.
   FY.features  Deterministic, frozen feature transforms. Identical
                formulas and rounding to the database (fy_features_v1).
                Nothing is fitted, nothing is ever refit.
   FY.importer  Reads FourYou_Client_Template_v1.xlsx and uploads one
                batch per social platform (fy_ingest_batch).
   FY.stream    Per-client live stream with event-time sliding windows.

   Needs: supabase-js v2 (window.supabase) and, for the importer,
   SheetJS (window.XLSX). Emits DOM events the app can listen to:
     fy:signed-in   {session}
     fy:signed-out  {}
     fy:remote-change {table, ids, source:'realtime'|'tab'}
     fy:ui-state    {state, source}
     fy:stream      {clientId, rows}
   ===================================================================== */
(function (global) {
  'use strict';

  const CFG = Object.assign({
    url: 'https://guyqmvkipcvabkvfmreq.supabase.co',
    publishableKey: 'sb_publishable_C8Q9w-MnX9uYvPPvjDld3w_IQpvdK1P',
    siteUrl: (global.location && global.location.origin && global.location.origin.indexOf('http') === 0)
               ? global.location.origin : 'https://foryouconsultants.netlify.app',
    brand: 'Four You Consultants'
  }, global.FY_SUPABASE || {}, global.FY_CONFIG || {});

  const FY = { version: '1.0.0', config: CFG, client: null, session: null };
  const emit = (name, detail) => { try { global.dispatchEvent(new CustomEvent(name, { detail })); } catch (e) {} };
  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  FY.init = function (opts) {
    opts = opts || {};
    if (opts.client) FY.client = opts.client;
    if (!FY.client) {
      if (!global.supabase || !global.supabase.createClient) throw new Error('supabase-js did not load');
      FY.client = global.supabase.createClient(CFG.url, opts.publishableKey || CFG.publishableKey,
        { auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true } });
    }
    FY.client.auth.onAuthStateChange((event, session) => {
      FY.session = session || null;
      if (event === 'SIGNED_OUT') { FY.sync.stop(); emit('fy:signed-out', {}); }
    });
    return FY;
  };

  /* =================================================================
     FEATURES — must match public.fy_scale() in the database exactly
     ================================================================= */
  const LN = Math.log;
  // Postgres round(numeric, 6): half away from zero
  const round6 = v => { const s = v < 0 ? -1 : 1; return s * Math.round(Math.abs(v) * 1e6 + 1e-9) / 1e6; };
  const slog = x => Math.sign(x) * LN(1 + Math.abs(x));
  const pos = x => Math.max(x, 0);
  const SCALERS = {
    followers:         x => LN(1 + pos(x)) / LN(1 + 1e9),
    views:             x => LN(1 + pos(x)) / LN(1 + 1e10),
    engagement_rate:   x => LN(1 + pos(x)) / LN(101),
    follower_delta:    x => slog(x) / slog(1e7),
    pct_change:        x => slog(x) / slog(1000),
    velocity_per_day:  x => LN(1 + pos(x)) / LN(1 + 1e9),
    count_small:       x => LN(1 + pos(x)) / LN(101),
    rate:              x => LN(1 + 100 * pos(x)) / LN(101),
    pace_vs_benchmark: x => slog(x) / slog(100)
  };
  // Inverses (for turning a model output back into real units)
  const INVERSE = {
    engagement_rate: y => Math.exp(y * LN(101)) - 1,
    followers:       y => Math.exp(y * LN(1 + 1e9)) - 1,
    views:           y => Math.exp(y * LN(1 + 1e10)) - 1
  };
  FY.features = {
    spec: 'fy_features_v1',
    slog,
    scale(kind, x) {
      if (x === null || x === undefined || x === '' || !isFinite(+x)) return null;
      const f = SCALERS[kind]; if (!f) throw new Error('unknown feature kind ' + kind);
      return round6(f(+x));
    },
    inverse(kind, y) { const f = INVERSE[kind]; if (!f) throw new Error('no inverse for ' + kind); return f(+y); },
    timeIndex(isoDate) {
      const [y, m, d] = String(isoDate).slice(0, 10).split('-').map(Number);
      const days = (Date.UTC(y, m - 1, d) - Date.UTC(2020, 0, 1)) / 86400000;
      return round6(days / 3650);
    },
    /** Season for any date, any year (north: Dec-Feb winter ...). */
    calendar(isoDate, hemisphere) {
      const [y, m, d] = String(isoDate).slice(0, 10).split('-').map(Number);
      let north = (m === 12 || m <= 2) ? 'winter' : m <= 5 ? 'spring' : m <= 8 ? 'summer' : 'fall';
      const flip = { winter: 'summer', summer: 'winter', spring: 'fall', fall: 'spring' };
      const season = (String(hemisphere || 'north').toLowerCase() === 'south') ? flip[north] : north;
      const sy = m === 12 ? y : (m <= 2 ? y - 1 : y);
      const sm = m === 12 || m <= 2 ? 12 : m <= 5 ? 3 : m <= 8 ? 6 : 9;
      const start = Date.UTC(sy, sm - 1, 1), now = Date.UTC(y, m - 1, d);
      return { season, season_year: sy, month: m, year: y, days_into_season: Math.round((now - start) / 86400000) };
    },
    /** The exact 7-input vector for a week of a series. Fixed transforms only. */
    weekVector(series, i) {
      const S = FY.features.scale, er = series.er, f = series.followers;
      return [
        FY.features.timeIndex(series.dates[i]),
        S('engagement_rate', er[Math.max(0, i - 1)]),
        S('engagement_rate', er[Math.max(0, i - 2)]),
        S('followers', f[i]),
        S('count_small', (series.posts || [])[i] || 0),
        (series.paid || [])[i] ? 1 : 0,
        (series.reach[i] > 0 && (series.saves || [])[i] != null) ? S('rate', series.saves[i] / series.reach[i]) : 0
      ];
    },
    target(er) { return FY.features.scale('engagement_rate', er); },
    untarget(y) { return INVERSE.engagement_rate(y); }
  };

  /* =================================================================
     AUTH — sign-in screen, reset email, server lockout
     ================================================================= */
  const CSS = `
  #fy-auth{position:fixed;inset:0;z-index:99999;display:flex;align-items:center;justify-content:center;
    background:radial-gradient(1200px 600px at 50% 40%,#16203a 0%,#0b1220 60%,#070b14 100%);font-family:'DM Sans',system-ui,-apple-system,Segoe UI,sans-serif;padding:16px}
  #fy-auth .box{width:100%;max-width:420px;background:#0f1830;border:1px solid #23304d;border-radius:18px;padding:34px 30px;box-shadow:0 30px 80px rgba(0,0,0,.45);color:#e6e9f2}
  #fy-auth .brand{display:flex;gap:12px;align-items:center;margin-bottom:26px}
  #fy-auth .logo{width:46px;height:46px;border-radius:12px;background:#C9A84C;color:#0b1220;font-weight:800;display:flex;align-items:center;justify-content:center;font-size:17px}
  #fy-auth .bn{font-weight:700;font-size:16px}#fy-auth .bn span{color:#C9A84C}#fy-auth .bs{font-size:12px;color:#8b95ab}
  #fy-auth h1{font-size:24px;margin:0 0 6px}#fy-auth p.sub{font-size:13px;color:#8b95ab;margin:0 0 22px}
  #fy-auth label{display:block;font-size:12.5px;color:#aab3c7;margin:14px 0 6px}
  #fy-auth input{width:100%;box-sizing:border-box;padding:12px 14px;border-radius:10px;border:1px solid #2a3756;background:#16203a;color:#fff;font-size:14px;outline:none}
  #fy-auth input:focus{border-color:#C9A84C}
  #fy-auth .pw{position:relative}#fy-auth .pw button{position:absolute;right:8px;top:7px;background:none;border:0;color:#8b95ab;cursor:pointer;font-size:12px;padding:6px}
  #fy-auth .go{width:100%;margin-top:22px;padding:13px;border:0;border-radius:10px;background:#C9A84C;color:#0b1220;font-weight:700;font-size:14.5px;cursor:pointer}
  #fy-auth .go:disabled{opacity:.55;cursor:not-allowed}
  #fy-auth .links{margin-top:16px;text-align:center;font-size:12.5px}#fy-auth .links a{color:#C9A84C;cursor:pointer;text-decoration:none}
  #fy-auth .msg{margin-top:14px;font-size:12.5px;min-height:18px;line-height:1.5}
  #fy-auth .err{color:#ff6b6b}#fy-auth .ok{color:#4ade80}#fy-auth .warn{color:#fbbf24}
  #fy-auth .meter{display:flex;gap:3px;margin-top:8px}#fy-auth .meter i{flex:1;height:4px;border-radius:2px;background:#23304d}#fy-auth .meter i.on{background:#fbbf24}#fy-auth .meter i.used{background:#ff6b6b}`;

  function ensureCss() {
    if (document.getElementById('fy-auth-css')) return;
    const s = document.createElement('style'); s.id = 'fy-auth-css'; s.textContent = CSS; document.head.appendChild(s);
  }
  function shell(inner) {
    ensureCss();
    let el = document.getElementById('fy-auth');
    if (!el) { el = document.createElement('div'); el.id = 'fy-auth'; document.body.appendChild(el); }
    el.innerHTML = '<div class="box" role="dialog" aria-label="Sign in"><div class="brand"><div class="logo">4U</div><div>' +
      '<div class="bn">Four You <span>Consultants</span></div><div class="bs">Analytics Platform</div></div></div>' + inner + '</div>';
    el.style.display = 'flex';
    return el;
  }
  const $ = id => document.getElementById(id);
  const say = (cls, html) => { const m = $('fy-msg'); if (m) { m.className = 'msg ' + cls; m.innerHTML = html; } };
  const fmtTime = iso => new Date(iso).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });

  function meter(left, max) {
    let h = '<div class="meter" title="' + left + ' of ' + max + ' attempts left">';
    for (let i = 0; i < max; i++) h += '<i class="' + (i < max - left ? 'used' : 'on') + '"></i>';
    return h + '</div>';
  }

  FY.auth = {
    /** Resolves with a session. Shows the sign-in screen only when there is no valid session. */
    async require() {
      if (!FY.client) FY.init();
      const { data } = await FY.client.auth.getSession();
      if (data && data.session && !data.session.user.is_anonymous) {
        FY.session = data.session; FY.auth.hide(); await FY.auth._afterSignIn(); return data.session;
      }
      if (data && data.session && data.session.user.is_anonymous) {
        try { await FY.client.auth.signOut({ scope: 'local' }); } catch (e) {}
      }
      return new Promise(resolve => { FY.auth._resolve = resolve; FY.auth.showSignIn(); });
    },

    showSignIn() {
      shell(
        '<h1>Welcome back</h1><p class="sub">Sign in to your analytics dashboard</p>' +
        '<form id="fy-form" autocomplete="on">' +
        '<label for="fy-user">Username</label>' +
        '<input id="fy-user" type="text" autocomplete="username" autocapitalize="none" spellcheck="false" required placeholder="J.washington@fouryouconsultants.com"/>' +
        '<label for="fy-pass">Password</label><div class="pw">' +
        '<input id="fy-pass" type="password" autocomplete="current-password" required placeholder="••••••••"/>' +
        '<button type="button" id="fy-eye">Show</button></div>' +
        '<button class="go" id="fy-go" type="submit">Sign in securely</button></form>' +
        '<div class="msg" id="fy-msg"></div>');
      $('fy-eye').onclick = () => { const p = $('fy-pass'); const show = p.type === 'password'; p.type = show ? 'text' : 'password'; $('fy-eye').textContent = show ? 'Hide' : 'Show'; };
      $('fy-form').onsubmit = e => { e.preventDefault(); FY.auth.signIn($('fy-user').value, $('fy-pass').value); };
      setTimeout(() => { const u = $('fy-user'); if (u) u.focus(); }, 50);
    },

    async signIn(email, password) {
      email = String(email || '').trim().toLowerCase();
      if (email && email.indexOf('@') < 0) email = email + '@fouryouconsultants.com';   /* "J.washington" works as the username */
      if (!email || !password) { say('err', 'Enter your username and password.'); return; }
      const go = $('fy-go'); go.disabled = true; say('', 'Checking…');
      try {
        const gate = await FY.client.rpc('login_gate', { p_email: email });
        if (gate.data && gate.data.locked) {
          go.disabled = false;
          say('err', '🔒 This account is locked after 10 unsuccessful attempts. Try again after <strong>' + fmtTime(gate.data.unlock_at) + '</strong>.');
          return;
        }
        const r = await FY.client.auth.signInWithPassword({ email, password });
        if (r.error) {
          const st = await FY.client.rpc('login_failed', { p_email: email });
          const s = st.data || {};
          go.disabled = false;
          if (s.locked) {
            say('err', '🔒 Too many unsuccessful attempts. The account is locked for 30 minutes (until <strong>' + fmtTime(s.unlock_at) +
              '</strong>). The lock is recorded in the security log.');
          } else {
            say('err', 'Username or password is incorrect. <strong>' + s.attempts_left + '</strong> of 10 attempts left before a 30-minute lock.' + meter(s.attempts_left, 10));
          }
          return;
        }
        FY.session = r.data.session;
        try { await FY.client.rpc('login_succeeded'); } catch (e) {}
        say('ok', 'Signed in.');
        FY.auth.hide();
        await FY.auth._afterSignIn();
        if (FY.auth._resolve) { FY.auth._resolve(FY.session); FY.auth._resolve = null; }
      } catch (e) {
        go.disabled = false; say('err', 'Could not reach the database: ' + esc(e.message || e));
      }
    },

    hide() { const el = $('fy-auth'); if (el) el.style.display = 'none'; },

    async signOut() {
      FY.sync.stop();
      try { await FY.client.auth.signOut(); } catch (e) {}
      FY.session = null;
      global.location.reload();
    },

    async _afterSignIn() {
      const { data } = await FY.client.auth.getSession();
      FY.session = data.session;
      await FY.sync.start();
      emit('fy:signed-in', { session: FY.session });
    }
  };

  /* =================================================================
     SYNC — realtime across devices + BroadcastChannel across tabs
     ================================================================= */
  const TABS = ('BroadcastChannel' in global) ? new BroadcastChannel('fy-sync-v1') : null;
  const TAB_ID = Math.random().toString(36).slice(2);
  let channel = null, pending = {}, flushTimer = null;
  const recentLocal = new Map();   // id -> time, to ignore our own echoes

  function queue(table, id, source) {
    if (id && recentLocal.has(id) && Date.now() - recentLocal.get(id) < 4000) return;
    const k = table + '|' + source;
    (pending[k] = pending[k] || { table, source, ids: new Set() }).ids.add(id);
    clearTimeout(flushTimer);
    flushTimer = setTimeout(() => {
      const batch = pending; pending = {};
      Object.values(batch).forEach(b => emit('fy:remote-change', { table: b.table, ids: Array.from(b.ids), source: b.source }));
    }, 250);
  }

  FY.sync = {
    status: 'off',
    async start() {
      if (!FY.session) return;
      FY.sync.stop();
      const uid = FY.session.user.id;
      channel = FY.client.channel('fy-sync-' + uid);
      ['clients', 'metrics_weekly', 'posts', 'feature_values'].forEach(t => {
        channel.on('postgres_changes', { event: '*', schema: 'public', table: t, filter: 'owner_id=eq.' + uid },
          p => queue(t, (p.new && (p.new.id || p.new.client_id)) || (p.old && p.old.id) || null, 'realtime'));
      });
      channel.on('postgres_changes', { event: '*', schema: 'public', table: 'user_ui_state', filter: 'user_id=eq.' + uid }, p => {
        if (p.new && p.new.device !== FY.state.deviceId) { FY.state.current = p.new.state || {}; emit('fy:ui-state', { state: FY.state.current, source: 'realtime' }); }
      });
      channel.subscribe(s => { FY.sync.status = s === 'SUBSCRIBED' ? 'live' : String(s).toLowerCase(); emit('fy:sync-status', { status: FY.sync.status }); });
      if (TABS) TABS.onmessage = ev => {
        const m = ev.data || {};
        if (m.tab === TAB_ID || m.uid !== uid) return;
        if (m.type === 'changed') queue(m.table, m.id, 'tab');
        if (m.type === 'ui') { FY.state.current = m.state; emit('fy:ui-state', { state: m.state, source: 'tab' }); }
        if (m.type === 'signout') global.location.reload();
      };
      await FY.state.load();
    },
    stop() { if (channel) { try { FY.client.removeChannel(channel); } catch (e) {} channel = null; } FY.sync.status = 'off'; },
    /** Call after the app writes a row, so other tabs refresh at once and our own echo is ignored. */
    notifyLocalWrite(table, id) {
      if (id) recentLocal.set(id, Date.now());
      if (TABS && FY.session) TABS.postMessage({ type: 'changed', table, id, tab: TAB_ID, uid: FY.session.user.id });
    },
    /** Pull every client (with payload) for the signed-in user — the database is the truth. */
    async pullClients() {
      const r = await FY.client.from('clients').select('*').eq('is_active', true).order('created_at');
      if (r.error) throw r.error;
      return (r.data || []).map(row => Object.assign({}, row.payload || {}, { id: row.id, name: row.name, updated_at: row.updated_at }));
    }
  };

  /* =================================================================
     STATE — UI state synced per user
     ================================================================= */
  let saveTimer = null;
  FY.state = {
    deviceId: (function () { try { let d = localStorage.getItem('fy_device_id'); if (!d) { d = 'dev-' + Math.random().toString(36).slice(2, 10); localStorage.setItem('fy_device_id', d); } return d; } catch (e) { return 'dev-' + TAB_ID; } })(),
    current: {},
    async load() {
      if (!FY.session) return {};
      const r = await FY.client.from('user_ui_state').select('state').eq('user_id', FY.session.user.id).maybeSingle();
      FY.state.current = (r.data && r.data.state) || {};
      emit('fy:ui-state', { state: FY.state.current, source: 'load' });
      return FY.state.current;
    },
    /** Merge a patch, e.g. {activeClient:'x', view:'analytics', range:'month', scroll:{analytics:420}, drafts:{week:{...}}} */
    set(patch) {
      FY.state.current = Object.assign({}, FY.state.current, patch);
      if (TABS && FY.session) TABS.postMessage({ type: 'ui', state: FY.state.current, tab: TAB_ID, uid: FY.session.user.id });
      clearTimeout(saveTimer);
      saveTimer = setTimeout(FY.state.flush, 800);
    },
    async flush() {
      if (!FY.session) return;
      await FY.client.from('user_ui_state').upsert({ user_id: FY.session.user.id, state: FY.state.current, device: FY.state.deviceId, updated_at: new Date().toISOString() });
    }
  };

  /* =================================================================
     IMPORTER — FourYou_Client_Template_v1.xlsx -> one batch per platform
     ================================================================= */
  const PLATFORM_BY_LABEL = { tiktok: 'tiktok', instagram: 'instagram', facebook: 'facebook', x: 'x', 'x (twitter)': 'x', twitter: 'x', linkedin: 'linkedin' };
  const CONTENT_TYPES = ['Event setup', 'Product / recipe', 'POV hook', 'Behind the scenes', 'Announcement', 'Static graphic'];
  const pad = n => String(n).padStart(2, '0');

  function toDate(v) {
    if (v === null || v === undefined || v === '') return null;
    if (typeof v === 'number' && global.XLSX) { const p = global.XLSX.SSF.parse_date_code(v); return p ? p.y + '-' + pad(p.m) + '-' + pad(p.d) : null; }
    const m = String(v).trim().match(/^(\d{4})-(\d{1,2})-(\d{1,2})/);
    if (m) return m[1] + '-' + pad(m[2]) + '-' + pad(m[3]);
    const us = String(v).trim().match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})/);
    return us ? us[3] + '-' + pad(us[1]) + '-' + pad(us[2]) : null;
  }
  function toDateTime(v) {           // local wall-clock time -> ISO with offset
    if (v === null || v === undefined || v === '') return null;
    let y, mo, d, h = 0, mi = 0;
    if (typeof v === 'number' && global.XLSX) {
      const p = global.XLSX.SSF.parse_date_code(v); if (!p) return null; y = p.y; mo = p.m; d = p.d; h = p.H; mi = p.M;
    } else {
      const m = String(v).trim().match(/^(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T](\d{1,2}):(\d{2}))?/);
      if (!m) return null; y = +m[1]; mo = +m[2]; d = +m[3]; h = +(m[4] || 0); mi = +(m[5] || 0);
    }
    return new Date(y, mo - 1, d, h, mi).toISOString();
  }
  function toInt(v) {
    if (v === null || v === undefined || v === '') return null;
    const n = Number(String(v).replace(/[,\s]/g, ''));
    return isFinite(n) ? Math.round(n) : null;
  }
  async function sha256(buf) {
    const h = await crypto.subtle.digest('SHA-256', buf);
    return Array.from(new Uint8Array(h)).map(b => b.toString(16).padStart(2, '0')).join('');
  }
  function readSentenceSheet(ws) {
    const rows = global.XLSX.utils.sheet_to_json(ws, { header: 1, raw: true, defval: null, blankrows: false });
    if (!rows.length) return [];
    const keys = rows[0] || [];
    const out = [];
    for (let r = 2; r < rows.length; r++) {
      const row = rows[r] || [];
      if (String(row[0] || '').indexOf('Example') === 0) continue;
      const o = {}; let any = false;
      keys.forEach((k, c) => { if (k && k !== '_') { const v = row[c]; o[k] = v; if (v !== null && v !== '') any = true; } });
      if (any) out.push(o);
    }
    return out;
  }

  FY.importer = {
    templateVersion: 'FY_TEMPLATE_v1',
    /** Parse only (no upload) — used for the preview step. */
    async parse(file) {
      if (!global.XLSX) throw new Error('Spreadsheet reader (SheetJS) did not load');
      const buf = await file.arrayBuffer();
      const wb = global.XLSX.read(buf, { type: 'array', cellDates: false });
      const start = wb.Sheets['Start Here'];
      if (!start || !start.A1 || start.A1.v !== FY.importer.templateVersion)
        throw new Error('This is not the Four You template (FY_TEMPLATE_v1). Download the template and fill it in.');
      const client = {};
      const cs = wb.Sheets['Client'];
      if (cs) global.XLSX.utils.sheet_to_json(cs, { header: 1, raw: true, defval: null }).forEach((row, i) => {
        if (i >= 3 && row && row[3]) client[row[3]] = row[1] == null ? '' : String(row[1]).trim();
      });
      const batches = {}, problems = [];
      const B = pk => (batches[pk] = batches[pk] || { platform: pk, weeks: [], posts: [] });
      [['tiktok', 'TikTok'], ['instagram', 'Instagram'], ['facebook', 'Facebook'], ['x', 'X'], ['linkedin', 'LinkedIn']].forEach(([pk, name]) => {
        const ws = wb.Sheets[name + ' Weekly']; if (!ws) return;
        readSentenceSheet(ws).forEach((o, i) => {
          const w = { week_start: toDate(o.week_start), followers: toInt(o.followers), reach: toInt(o.reach), likes: toInt(o.likes),
            comments: toInt(o.comments), shares: toInt(o.shares), saves: toInt(o.saves), posts: toInt(o.posts),
            profile_visits: toInt(o.profile_visits), clicks: toInt(o.clicks) };
          if (!w.week_start || w.followers === null || w.reach === null) { problems.push(name + ' Weekly row ' + (i + 4) + ': needs a date, total followers and views'); return; }
          B(pk).weeks.push(w);
        });
      });
      const ps = wb.Sheets['Posts'];
      if (ps) readSentenceSheet(ps).forEach((o, i) => {
        const pk = PLATFORM_BY_LABEL[String(o.platform || '').trim().toLowerCase()];
        if (!pk) { problems.push('Posts row ' + (i + 4) + ': choose a platform'); return; }
        const p = { title: o.title ? String(o.title).trim() : '', posted_at: toDateTime(o.posted_at), measured_at: toDateTime(o.measured_at),
          content_class: CONTENT_TYPES.indexOf(String(o.content_type || '').trim()), reach: toInt(o.reach), likes: toInt(o.likes),
          comments: toInt(o.comments), shares: toInt(o.shares), saves: toInt(o.saves), permalink: o.permalink || null };
        if (p.content_class < 0) p.content_class = null;
        if (!p.title || p.reach === null) { problems.push('Posts row ' + (i + 4) + ': needs a title and views'); return; }
        B(pk).posts.push(p);
      });
      Object.values(batches).forEach(b => b.weeks.sort((a, c) => a.week_start < c.week_start ? -1 : 1));
      return { client, batches: Object.values(batches), problems, fileName: file.name, sha: await sha256(buf) };
    },

    /** Upload: creates the client if needed, then one fy_ingest_batch call per platform. */
    async upload(parsed, clientId) {
      if (!FY.session) throw new Error('Sign in first');
      let id = clientId;
      if (!id) {
        const name = parsed.client.name || 'New client';
        id = name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') + '-' + FY.session.user.id.replace(/-/g, '').slice(0, 8);
        const handles = {};
        ['tiktok', 'instagram', 'facebook', 'x', 'linkedin'].forEach(k => { if (parsed.client['handle_' + k]) handles[k] = parsed.client['handle_' + k].replace(/^@/, ''); });
        const primary = PLATFORM_BY_LABEL[String(parsed.client.primary_platform || '').toLowerCase()] || (parsed.batches[0] && parsed.batches[0].platform) || 'tiktok';
        const ex = await FY.client.from('clients').select('id,payload').eq('id', id).maybeSingle();
        const payload = Object.assign({}, (ex.data && ex.data.payload) || {}, {
          id, name, industry: parsed.client.industry || '', handles, primary, platforms: [primary],
          avatar: (name.replace(/[^A-Za-z0-9]/g, '').slice(0, 2) || 'CL').toUpperCase(), color: '#9B7FE8',
          hemisphere: (parsed.client.hemisphere || 'North').toLowerCase(), contact_email: parsed.client.contact_email || ''
        });
        const up = await FY.client.from('clients').upsert({ id, owner_id: FY.session.user.id, name, industry: parsed.client.industry || null,
          primary_platform: primary, platforms: [primary], payload, is_active: true, updated_at: new Date().toISOString() });
        if (up.error) throw up.error;
      }
      const results = [];
      for (const b of parsed.batches) {
        const r = await FY.client.rpc('fy_ingest_batch', { p_client: id, p_platform: b.platform, p_weeks: b.weeks, p_posts: b.posts,
          p_file_name: parsed.fileName, p_sha256: parsed.sha + ':' + b.platform });
        results.push(r.error ? { platform: b.platform, error: r.error.message } : r.data);
      }
      FY.sync.notifyLocalWrite('clients', id);
      return { clientId: id, results };
    },

    /** Small preview modal: shows what will be uploaded, then uploads on confirm. */
    async openPicker(clientId, onDone) {
      const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.xlsx';
      inp.onchange = async () => {
        const f = inp.files[0]; if (!f) return;
        let parsed;
        try { parsed = await FY.importer.parse(f); } catch (e) { alert(e.message); return; }
        const lines = parsed.batches.map(b => '• ' + b.platform + ': ' + b.weeks.length + ' weeks, ' + b.posts.length + ' posts').join('\n');
        const warn = parsed.problems.length ? '\n\nSkipped:\n' + parsed.problems.slice(0, 8).join('\n') : '';
        if (!parsed.batches.length) { alert('No filled-in rows found.' + warn); return; }
        if (!confirm('Upload this to the database?\n\nClient: ' + (parsed.client.name || '(current client)') + '\n' + lines + warn)) return;
        const res = await FY.importer.upload(parsed, clientId);
        if (onDone) onDone(res);
      };
      inp.click();
    }
  };

  /* =================================================================
     STREAM — per client, event-time sliding windows from the database
     ================================================================= */
  const streams = {};
  FY.stream = {
    /** Subscribe to one client. Every change re-reads v_rolling_metrics (28/84-day RANGE windows by event date). */
    async open(clientId) {
      if (streams[clientId]) return streams[clientId];
      const read = async () => {
        const r = await FY.client.from('v_rolling_metrics').select('*').eq('client_id', clientId).order('week_start');
        if (!r.error) emit('fy:stream', { clientId, rows: r.data || [] });
        return r.data || [];
      };
      const onChange = ev => { if (ev.detail && ['metrics_weekly', 'posts', 'clients'].indexOf(ev.detail.table) > -1) read(); };
      global.addEventListener('fy:remote-change', onChange);
      streams[clientId] = { read, close() { global.removeEventListener('fy:remote-change', onChange); delete streams[clientId]; } };
      await read();
      return streams[clientId];
    },
    close(clientId) { if (streams[clientId]) streams[clientId].close(); }
  };

  global.FY = FY;
})(typeof window !== 'undefined' ? window : globalThis);
