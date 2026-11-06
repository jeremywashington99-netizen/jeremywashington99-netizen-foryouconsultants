/* ==========================================================================
   SEGMENT 12 - Site wiring: every training run lives in Supabase

   NET (the Train Model network):
     FYSync.trainRemote(client) asks the Supabase function train-net to train and save a run
     FYSync.restoreNet(client)  loads the last saved run so the page can predict and draw
     FYSync.fillForecasts(c)    caches 6-week forecasts for the scorecard
     FYSync.ensureNet(client)   restore if saved, otherwise train (and save)

   Training happens in Supabase, never in this browser. Nothing here touches
   localStorage. If the database is unreachable the screen says so; it never
   trains locally or keeps a quiet local copy instead.
   ========================================================================== */
(function () {
  'use strict';
  const S = window.FYStore, X = window.FYExplain;
  const Sync = { lastSave: null, lastError: null, restoring: false, tried: {} };
  window.FYSync = Sync;

  function ownerId() { return (typeof SESSION !== 'undefined' && SESSION && SESSION.user) ? SESSION.user.id : null; }
  function weeksOf(c) {
    return Math.max.apply(null, [0].concat((c.platforms || []).map(function (pk) { const s = c.series && c.series[pk]; return s && s.er ? s.er.length : 0; })));
  }
  function postsOf(c) { return (c.posts_log || []).length; }
  function mm(o) { return { lo: o.lo, hi: o.hi, norm: function (v) { return o.hi === o.lo ? 0.5 : (v - o.lo) / (o.hi - o.lo); }, denorm: function (v) { return v * (o.hi - o.lo) + o.lo; } }; }
  function plainNorms(n) {
    const out = {};
    Object.keys(n || {}).forEach(function (pk) { out[pk] = {}; Object.keys(n[pk]).forEach(function (k) { out[pk][k] = { lo: n[pk][k].lo, hi: n[pk][k].hi }; }); });
    return out;
  }
  function liveNorms(n) {
    const out = {};
    Object.keys(n || {}).forEach(function (pk) { out[pk] = {}; Object.keys(n[pk]).forEach(function (k) { out[pk][k] = (n[pk][k] && n[pk][k].frozen && typeof FYNetCore !== 'undefined' && FYNetCore.FROZEN[k]) ? FYNetCore.FROZEN[k] : mm(n[pk][k]); }); });
    return out;
  }

  /* ---------- train (in Supabase) ---------- */

  /* Only the numbers the network learns from; the account is proved by the signed-in session, not by anything in here. */
  function payload(c) {
    const series = {}, platforms = [];
    (c.platforms || []).forEach(function (pk) {
      const s = c.series && c.series[pk]; if (!s || !s.er || !s.er.length) return;
      const n = s.er.length;
      const col = function (a) { const out = []; for (let i = 0; i < n; i++) { const v = a ? Number(a[i]) : 0; out.push(Number.isFinite(v) ? v : 0); } return out; };
      series[pk] = { er: col(s.er), followers: col(s.followers), reach: col(s.reach), posts: col(s.posts), paid: col(s.paid), saves: col(s.saves) };
      if (s.dates) series[pk].dates = s.dates.slice(0, n).map(String);
      platforms.push(pk);
    });
    return {
      platforms: platforms, primary: c.primary && series[c.primary] ? c.primary : platforms[0], series: series,
      posts_log: (c.posts_log || []).map(function (p) { return { platform: p.platform, cls: p.cls, posted_at: p.posted_at ? String(p.posted_at).slice(0, 10) : null }; })
    };
  }
  Sync.payload = payload;

  /* Ask the Supabase function to train and save, then load what it saved. Nothing is trained or kept in this browser. */
  Sync.trainRemote = async function (c) {
    if (NET.training) return;
    NET.training = true; NET.lastError = null;
    try {
      if (!sb) throw new Error('not connected to the database');
      const ses = await sb.auth.getSession();
      const tok = ses && ses.data && ses.data.session && ses.data.session.access_token;
      if (!tok) throw new Error('not signed in');
      const ctl = new AbortController(), timer = setTimeout(function () { ctl.abort(); }, 90000);
      let res, out;
      try {
        res = await fetch(SB_URL + '/functions/v1/train-net', {
          method: 'POST', signal: ctl.signal,
          headers: { authorization: 'Bearer ' + tok, apikey: SB_PUBLISHABLE_KEY, 'content-type': 'application/json' },
          body: JSON.stringify({ client_id: c.id, client: payload(c), feature_version: NET.FEATURE_VERSION })
        });
        out = await res.json().catch(function () { return null; });
      } catch (e) {
        throw new Error(e && e.name === 'AbortError' ? 'the database took too long to answer' : 'could not reach the database to train (' + (e.message || e) + ')');
      } finally { clearTimeout(timer); }
      if (!res.ok || !out || !out.ok) throw new Error((out && out.error) || ('the database answered ' + res.status));
      const r = await Sync.restoreNet(c);
      if (!r.ok || !r.restored) throw new Error(r.error || 'trained, but the saved result could not be loaded');
      Sync.lastSave = new Date(out.trained_at || Date.now()); Sync.lastError = null;
      return out;
    } finally { NET.training = false; }
  };

  /* ---------- restore ---------- */
  Sync.restoreNet = async function (c) {
    const r = await S.latest(sb, c.id, 'net');
    if (!r.ok) { Sync.lastError = r.error; return { ok: false, error: r.error }; }
    if (!r.run || !r.run.weights || !r.run.weights.reg) return { ok: true, restored: false };
    if (r.run.weights.fv !== NET.FEATURE_VERSION) return { ok: true, restored: false };   /* saved with older inputs: retrain instead of misreading */
    try {
      const w = r.run.weights;
      NET.reset();
      NET.reg = await S.restoreModel(tf, w.reg);
      NET.clf = w.clf ? await S.restoreModel(tf, w.clf) : null;
      NET.norms = liveNorms(w.norms);
      NET.classScores = w.classScores || {};
      NET.guard = w.guard || (r.run.summary && r.run.summary.guardrails) || null; NET.classCounts = w.classCounts || null;
      NET.lossHistory = w.lossHistory || [];
      NET.lastFit = w.lastFit || { actual: [], predicted: [], labels: [] };
      NET.metrics = Object.assign({ mae: 0, rmse: 0, r2: 0, epochs: 0, params: 0, backend: '', rows: 0 }, r.run.metrics || {});
      NET.trained = true; NET.clientId = c.id;
      NET.savedAt = r.run.trained_at; NET.savedRun = r.run;
      await Sync.fillForecasts(c);
      return { ok: true, restored: true, run: r.run, stale: S.staleness(r.run, weeksOf(c), postsOf(c)) };
    } catch (e) { Sync.lastError = e.message; NET.reset(); return { ok: false, error: 'could not rebuild the saved model: ' + e.message }; }
  };

  Sync.fillForecasts = async function (c) {
    NET.forecasts = {};
    for (const pk of (c.platforms || [])) {
      try { const f = await NET.forecast(c, pk, 6); if (f && f.length) NET.forecasts[pk] = f; } catch (e) {}
    }
    return NET.forecasts;
  };

  /* ---------- the one entry the AI screen calls ---------- */
  Sync.ensureNet = async function (c) {
    if (!c) return { ok: false };
    if (NET.clientId !== c.id) { NET.reset(); NET.clientId = c.id; NET.savedRun = null; NET.savedAt = null; }
    if (NET.trained) return { ok: true, restored: false };
    if (Sync.tried[c.id]) return { ok: true, restored: false };
    Sync.tried[c.id] = true;
    const r = await Sync.restoreNet(c);
    if (window.FYLab && FYLab.autoForecast) { try { await FYLab.autoForecast(c); } catch (e) { console.warn('forecast', e); } }
    if (r.ok && r.restored) return r;
    if (typeof hasEnoughToTrain === 'function' && hasEnoughToTrain(c)) { await retrainModels(); }
    return { ok: true, restored: false };
  };

  /* Item 9 uses the forecast's own walk-forward test through the change, when it is available. */
  Sync.guardWithEvidence = function (c) {
    try {
      const pk = c.primary && c.series[c.primary] ? c.primary : (c.platforms || [])[0];
      const bt = window.FYLab && FYLab.testedForecast ? FYLab.testedForecast(c.id, pk) : null;
      if (!bt || !bt.shift) return NET.guard;
      return FYGuard.applyShiftEvidence(NET.guard, FYGuard.drift(c.series[pk]), bt.shift, bt.best);
    } catch (e) { return NET.guard; }
  };

  /* ---------- plain-English panel ---------- */
  Sync.netPanelHTML = function (c) {
    if (!NET.trained) return '';
    const sm = X.netSummary({ r2: NET.metrics.r2, rows: NET.metrics.rows });
    const stale = S.staleness(NET.savedRun, weeksOf(c), postsOf(c));
    const col = sm.fitPct >= 55 ? 'var(--green)' : sm.fitPct >= 30 ? 'var(--amber)' : 'var(--red)';
    const where = NET.savedRun && NET.savedRun.hyperparams && NET.savedRun.hyperparams.trained_on === 'supabase'
      ? '<span style="color:var(--green)">&#10003; Trained inside your Supabase database</span> &middot; '
      : '<span style="color:var(--amber)">This saved version was trained earlier in the browser. Press Train Model to retrain it in your database.</span> &middot; ';
    const saved = Sync.lastError
      ? '<span style="color:var(--red)">Not saved to the database: ' + esc(Sync.lastError) + '</span>'
      : (NET.savedAt || Sync.lastSave)
        ? '<span style="color:var(--green)">&#10003; Saved to your database ' + esc(new Date(NET.savedAt || Sync.lastSave).toLocaleString()) + '</span>' : '';
    const tech = NET.describe().filter(function (l) { return !/dropout/.test(l.name); }).map(function (l) {
      const p = X.layerPlain({ type: /dropout/.test(l.name) ? 'Dropout' : 'Dense', role: /out/.test(l.name) ? 'output' : 'hidden', activation: l.activation });
      const a = X.activationPlain(l.activation);
      return '<tr><td>' + esc(p.stage) + '</td><td>' + esc(p.does) + '</td><td>' + esc(a.label) + ' &mdash; ' + esc(a.plain) + '</td><td style="font-family:monospace;font-size:11px;color:var(--text2)">' + esc(l.name) + ' &middot; ' + l.params + '</td></tr>';
    }).join('');
    return '<div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap"><div style="font-size:26px;font-weight:700;color:' + col + '">' + (sm.fitPct == null ? '' : sm.fitPct + '%') + '</div>' +
      '<div><div style="font-weight:600">' + esc(sm.verdict) + '</div><div class="card-sub">' + esc(sm.line) + '</div></div></div>' +
      '<p class="card-sub" style="margin:10px 0 4px">' + esc(sm.learnedFrom) + '</p>' +
      '<p class="card-sub" style="margin:0 0 6px">' + esc(sm.caveat) + '</p>' +
      (sm.weak ? '<div class="callout callout-warn" style="margin:8px 0">' + esc(sm.weak) + '</div>' : '') +
      (stale.stale ? '<div class="callout callout-info" style="margin:8px 0">' + esc(stale.reason) + '. Press <strong>Train Model</strong> to refresh it.</div>' : '') +
      '<div class="card-sub" style="margin:6px 0">' + where + saved + '</div>' +
      (NET.guard && window.FYGuard ? FYGuard.html(Sync.guardWithEvidence(c), esc) : '') +
      '<details style="margin-top:8px"><summary class="card-sub" style="cursor:pointer">Technical detail (for analysts)</summary>' +
      '<div style="overflow-x:auto;margin-top:8px"><table class="data-table"><thead><tr><th>Stage</th><th>What it does</th><th>Rule it uses</th><th>Layer</th></tr></thead><tbody>' + tech + '</tbody></table></div></details>';
  };
})();
