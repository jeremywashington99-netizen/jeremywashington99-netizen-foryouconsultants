/* ==========================================================================
   SEGMENT 9 - Screens for the instant metrics (Segments 6-8)

   Adds, without touching the original layout:
     - Dashboard: an instant scorecard that appears with the FIRST entry,
       graded against published benchmarks, plus a "what is missing" line.
     - Posts table: an Analyze button per post opening its past / present /
       future report, and a way to re-read its figures so the decay is
       measured rather than modelled.
     - AI screen: a forecast available from entry one, naming its method, and
       the full marketing framework with the field that unlocks each metric.
     - Settings: a demo client, so all of the above can be tried at once.

   Everything degrades: no client, no data, one row, or a missing field each
   produce a plain statement rather than an empty panel or a fake number.
   ========================================================================== */
(function () {
  'use strict';

  const B = (window.FYMkt || {}).bench, M = (window.FYMkt || {}).metrics, L = (window.FYMkt || {}).life;
  if (!B || !M || !L) { console.warn('FYMkt segments not loaded'); return; }

  const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const $ = id => document.getElementById(id);

  /* The app declares DB, NET, PLATFORMS and CONTENT_CLASSES with const, which
     puts them in the global lexical scope and NOT on window. A classic script
     can still read them by name, so they are reached by bare identifier
     behind a typeof guard rather than through window. */
  function activeClient() { return (typeof DB !== 'undefined' && DB && DB.active) ? DB.active() : null; }
  function allClients() { return (typeof DB !== 'undefined' && DB && DB.clients) ? DB.clients : []; }
  function plabel(pk) {
    return (typeof PLATFORMS !== 'undefined' && PLATFORMS && PLATFORMS[pk] && PLATFORMS[pk].label) ? PLATFORMS[pk].label : pk;
  }
  function clsName(i) {
    return (typeof CONTENT_CLASSES !== 'undefined' && CONTENT_CLASSES && CONTENT_CLASSES[i] != null) ? CONTENT_CLASSES[i] : null;
  }
  function netForecast(pk) {
    try {
      if (typeof NET !== 'undefined' && NET && NET.trained && NET.forecasts && NET.forecasts[pk] && NET.forecasts[pk].length) return NET.forecasts[pk][0];
    } catch (e) { /* model not built */ }
    return null;
  }
  const fmtN = v => v == null ? '—' : (typeof fmt === 'function' ? fmt(v) : String(v));
  const TONE = { bad: 'var(--red)', warn: 'var(--amber)', ok: 'var(--text)', good: 'var(--green)', great: 'var(--gold)' };

  /* Basis -> a short badge the reader can trust at a glance. */
  const BASIS_LABEL = {
    measured: 'measured', derived: 'from your figures', benchmark: 'industry benchmark',
    'benchmark+measured': 'benchmark + your figure', modelled: 'modelled estimate',
    unavailable: 'not available', missing: 'not tracked'
  };
  function basisBadge(basis) {
    const cls = basis === 'measured' || basis === 'derived' ? 'badge-green'
      : basis === 'benchmark' || basis === 'benchmark+measured' ? 'badge-blue'
      : basis === 'modelled' ? 'badge-amber' : 'badge-purple';
    return '<span class="badge ' + cls + '">' + esc(BASIS_LABEL[basis] || basis) + '</span>';
  }

  /* ------------------------------------------------------------------
     Shared helpers over the active client
     ------------------------------------------------------------------ */

  /** Latest follower count for a platform, so per-follower rates work. */
  function followersAt(c, pk) {
    const s = c && c.series && c.series[pk];
    if (s && s.followers && s.followers.length) {
      for (let i = s.followers.length - 1; i >= 0; i--) if (s.followers[i] > 0) return s.followers[i];
    }
    const snap = (c && c.snapshots || []).filter(x => x.platform === pk || !x.platform).pop();
    return snap && snap.followers ? snap.followers : null;
  }

  /** The latest week of a platform as a flat row Segment 7 can read. */
  function latestWeek(c, pk) {
    const s = c && c.series && c.series[pk];
    if (!s || !s.dates || !s.dates.length) return null;
    const i = s.dates.length - 1;
    const row = { platform: pk, week_start: s.dates[i] };
    ['er', 'followers', 'reach', 'clicks', 'posts', 'likes', 'saves', 'shares', 'comments']
      .forEach(k => { if (Array.isArray(s[k]) && s[k][i] != null) row[k] = s[k][i]; });
    return row;
  }

  function platformsWithData(c) {
    if (!c) return [];
    return (c.platforms || []).filter(pk => {
      const s = c.series && c.series[pk];
      return (s && s.dates && s.dates.length) || (c.posts_log || []).some(p => p.platform === pk);
    });
  }

  function postsOf(c, pk) { return (c && c.posts_log || []).filter(p => !pk || p.platform === pk); }

  function baselineFor(c, pk) {
    return M.baseline(postsOf(c, pk), { followersAt: followersAt(c, pk) });
  }

  /* ------------------------------------------------------------------
     Dashboard: the instant scorecard
     ------------------------------------------------------------------ */

  function scoreRow(r) {
    const g = r.grade || {};
    const colour = g.ok ? (TONE[g.tone] || 'var(--text)') : 'var(--text2)';
    const right = g.ok
      ? '<strong style="color:' + colour + '">' + esc(g.band) + '</strong>' +
        '<div style="font-size:11px;color:var(--text3)">' + esc(g.how || '') +
        (g.ref != null ? ' · ' + g.ref + '%' : '') + '</div>'
      : '<span style="font-size:11px;color:var(--text3)">no yardstick yet</span>';
    return '<tr><td>' + esc(r.label) + '</td>' +
      '<td style="text-align:right"><strong>' + r.value + '%</strong></td>' +
      '<td>' + right + '</td>' +
      '<td>' + basisBadge(g.basis || 'derived') + '</td></tr>';
  }

  function instantCard(c) {
    const plats = platformsWithData(c);
    if (!c) return '';
    if (!plats.length) {
      return card('Instant analytics', 'Ready as soon as one figure is in',
        '<div class="callout callout-info">Nothing is logged for ' + esc(c.name) + ' yet. ' +
        'Add <strong>one</strong> week or <strong>one</strong> post and this fills in immediately — ' +
        'rates, a grade against the published platform average, and a forecast. No four-week wait.</div>' +
        '<div style="margin-top:12px"><button class="btn btn-primary" onclick="openAddWeek()">+ Add a week</button> ' +
        '<button class="btn btn-ghost" onclick="openAddPost()">+ Log a post</button> ' +
        '<button class="btn btn-ghost" onclick="fyLoadDemo()">Load demo client instead</button></div>');
    }
    let body = '';
    plats.forEach(pk => {
      const label = plabel(pk);
      const week = latestWeek(c, pk);
      const posts = postsOf(c, pk);
      const base = baselineFor(c, pk);
      const entry = week || posts[posts.length - 1];
      if (!entry) return;
      const src = week ? ('latest week, ' + esc(week.week_start)) : (posts.length + ' logged post(s)');
      const cardData = M.scorecard(entry, { platform: pk, followersAt: followersAt(c, pk), baseline: base });
      const rows = cardData.rows.filter(r => r.value != null);
      body += '<div style="margin-bottom:18px">' +
        '<div style="font-size:13px;font-weight:600;margin-bottom:6px">' + esc(label) +
        ' <span style="font-weight:400;color:var(--text3);font-size:11px">· ' + src + '</span></div>';
      body += rows.length
        ? '<div class="table-wrap"><table><thead><tr><th>Metric</th><th style="text-align:right">Value</th>' +
          '<th>Against the yardstick</th><th>Where it comes from</th></tr></thead><tbody>' +
          rows.map(scoreRow).join('') + '</tbody></table></div>'
        : '<div class="callout callout-info">What is entered for ' + esc(label) +
          ' does not yet allow a rate. Reach plus any interaction count is enough.</div>';
      const miss = (cardData.missing || []).filter(m => ['follow_rate', 'ctr', 'profile_visit_rate', 'roas', 'hook_rate'].indexOf(m.key) > -1);
      if (miss.length) {
        body += '<div style="font-size:11px;color:var(--text3);margin-top:6px">Also available once entered: ' +
          miss.map(m => esc(m.needs)).filter((v, i, a) => a.indexOf(v) === i).slice(0, 4).join('; ') + '.</div>';
      }
      body += '</div>';
    });
    return card('Instant analytics', 'Computed from the first entry · graded against published platform averages',
      body + sourceFootnote());
  }

  function sourceFootnote() {
    const seen = {};
    Object.keys(B.SOURCES).forEach(k => { seen[k] = B.SOURCES[k]; });
    return '<div style="font-size:11px;color:var(--text3);margin-top:10px;line-height:1.7">' +
      'Benchmarks are published industry averages, not this client’s data, and are labelled as such wherever they appear. ' +
      'Read ' + esc(B.FETCHED) + ' from ' +
      ['SI-B', 'SI-R', 'SI-L', 'SG-H'].map(k => '<a href="' + seen[k].url + '" target="_blank" rel="noopener" style="color:var(--gold)">' +
        esc(seen[k].name.replace(/^Socialinsider - /, 'Socialinsider: ').replace(/^Scott Graffius - /, 'Graffius: ')) + '</a>').join(', ') +
      '.</div>';
  }

  function card(title, sub, body) {
    return '<div class="card" style="margin-bottom:16px"><div class="card-header"><div>' +
      '<div class="card-title">' + esc(title) + '</div>' +
      '<div class="card-sub">' + sub + '</div></div></div>' + body + '</div>';
  }

  function renderInstant() {
    const host = $('mkt-instant');
    if (!host) return;
    host.innerHTML = instantCard(activeClient());
  }

  /* ------------------------------------------------------------------
     AI screen: forecast from entry one, and the framework
     ------------------------------------------------------------------ */

  function forecastCard(c) {
    const plats = platformsWithData(c);
    if (!c) return '';
    let body = '';
    const list = plats.length ? plats : [(c.primary || 'instagram')];
    list.forEach(pk => {
      const label = plabel(pk);
      const s = c.series && c.series[pk];
      const ers = (s && s.er ? s.er : []).filter(v => Number(v) > 0);
      const fol = followersAt(c, pk);
      const metric = M.ER_KEY_FOR[pk] || 'er_follower';
      /* The series holds the rate as the owner entered it (by reach for most
         platforms). Forecast that same series; the benchmark anchor is only
         used while there is nothing to trend. */
      const f = L.forecast(ers, { platform: pk, metric: metric, followers: fol, modelValue: netForecast(pk) });
      if (f.value == null) {
        body += '<div style="margin-bottom:14px"><strong>' + esc(label) + '</strong><div class="callout callout-info" style="margin-top:6px">' +
          esc(f.why) + '</div></div>';
        return;
      }
      body += '<div style="margin-bottom:16px">' +
        '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap">' +
        '<div><strong>' + esc(label) + '</strong> next period engagement rate</div>' +
        '<div style="font-size:20px;font-weight:700;color:var(--gold)">' + f.value + '%</div></div>' +
        '<div style="font-size:12px;color:var(--text2);margin-top:4px">Indicative range ' + f.low + '% to ' + f.high + '% ' +
        '· ' + basisBadge(f.basis) + ' <span class="badge badge-purple">' + esc(f.method) + '</span></div>' +
        '<div style="font-size:12px;color:var(--text2);margin-top:6px">' + esc(f.why) + '</div>' +
        '<div style="font-size:11px;color:var(--text3);margin-top:4px">Confidence: ' + esc(f.confidence) +
        '. The range is an indicative spread, not a statistical confidence interval.</div>' +
        '</div>';
    });
    return card('Forecast — available from the first entry',
      'Method changes as history grows: benchmark → anchored → trend → trained model',
      body +
      '<div style="font-size:11px;color:var(--text3);line-height:1.7">' +
      '<strong>benchmark</strong> nothing logged yet, so the published platform average is shown. ' +
      '<strong>anchored</strong> one entry, averaged with that benchmark because one reading is weak evidence. ' +
      '<strong>trend</strong> two or three entries, damped so a short run is not read as a trajectory. ' +
      '<strong>model</strong> four or more, handed to the trained network on this screen.</div>');
  }

  function frameworkCard(c) {
    const entries = postsOf(c).concat(platformsWithData(c).map(pk => latestWeek(c, pk)).filter(Boolean));
    const cov = M.coverage(entries, { followersAt: c ? followersAt(c, c.primary) : null });
    const groups = { past: 'Past — what already happened', present: 'Present — what is happening now', future: 'Future — what to expect' };
    let body = '';
    const onCount = cov.filter(r => r.available).length;
    Object.keys(groups).forEach(when => {
      const rows = cov.filter(r => r.when === when);
      body += '<div style="font-size:13px;font-weight:600;margin:14px 0 6px">' + groups[when] +
        ' <span style="font-weight:400;color:var(--text3);font-size:11px">· ' +
        rows.filter(r => r.available).length + ' of ' + rows.length + ' available</span></div>' +
        '<div class="table-wrap"><table><thead><tr><th>Metric</th><th>How it is calculated</th>' +
        '<th>Why it matters</th><th>Status</th></tr></thead><tbody>' +
        rows.map(r => '<tr><td><strong>' + esc(r.label) + '</strong><div style="font-size:11px;color:var(--text3)">' + esc(r.group) + '</div></td>' +
          '<td style="font-size:11px;font-family:monospace;color:var(--text2)">' + esc(r.formula) + '</td>' +
          '<td style="font-size:11px;color:var(--text2)">' + esc(r.why) + '</td>' +
          '<td>' + (r.available
            ? '<span class="badge badge-green">live</span>'
            : '<span class="badge badge-amber">needs</span><div style="font-size:11px;color:var(--text3)">' +
              esc((r.needs || []).join(', ')) + '</div>') + '</td></tr>').join('') +
        '</tbody></table></div>';
    });
    return card('The marketing framework — ' + onCount + ' of ' + cov.length + ' metrics live',
      'Everything a social marketer should watch, across past, present and future. ' +
      'Where a metric is not live, the field that unlocks it is named.', body);
  }

  function renderFuture() {
    const host = $('mkt-future');
    if (!host) return;
    const c = activeClient();
    host.innerHTML = c ? forecastCard(c) + frameworkCard(c) : '';
  }

  /* ------------------------------------------------------------------
     Per-post report: past / present / future
     ------------------------------------------------------------------ */

  function findPost(id) {
    const c = activeClient();
    return c ? (c.posts_log || []).find(p => p.id === id) || null : null;
  }

  window.fyPostReport = function (id, ev) {
    if (ev) ev.stopPropagation();
    const c = DB.active(), p = findPost(id);
    if (!p) { if (typeof toast === 'function') toast('That post is no longer logged', 'error'); return; }
    const cls = clsName(p.cls);
    const r = L.postReport(p, {
      posts: c.posts_log || [], followersAt: followersAt(c, p.platform),
      baseline: baselineFor(c, p.platform), readings: p.readings || [], className: cls
    });

    /* ---- past ---- */
    const rows = r.past.scorecard.rows.filter(x => x.value != null);
    let past = '<div style="font-size:13px;font-weight:600;margin-bottom:6px">How it did</div>';
    past += rows.length
      ? '<div class="table-wrap"><table><thead><tr><th>Metric</th><th style="text-align:right">Value</th>' +
        '<th>Against the yardstick</th><th>Where it comes from</th></tr></thead><tbody>' +
        rows.map(scoreRow).join('') + '</tbody></table></div>'
      : '<div class="callout callout-info">No rate could be worked out from what was entered for this post.</div>';
    if (r.past.rank) {
      past += '<div style="font-size:12px;color:var(--text2);margin-top:8px">Ranked <strong>' + r.past.rank.position +
        ' of ' + r.past.rank.of + '</strong> among posts logged on this platform, beating ' + r.past.rank.percentile +
        '% of them. Median for the others: ' + r.past.rank.medianPeer + '%.</div>';
    } else {
      past += '<div style="font-size:12px;color:var(--text3);margin-top:8px">No other posts on this platform yet, so there is nothing to rank it against.</div>';
    }

    /* ---- present ---- */
    const li = r.present;
    let present = '<div style="font-size:13px;font-weight:600;margin:16px 0 6px">Where it is now ' + basisBadge(li.basis) + '</div>';
    if (li.ageHours == null) {
      present += '<div class="callout callout-warn">' + esc(li.notes.join(' ')) + '</div>';
    } else {
      present += '<div class="table-wrap"><table><tbody>' +
        tr('Age', li.ageHoursNow >= 48 ? (li.ageHoursNow / 24).toFixed(1) + ' days' : li.ageHoursNow + ' hours') +
        tr('Stage', li.stage ? '<strong>' + esc(li.stage) + '</strong>' : '—') +
        tr('Engagement so far', fmtN(li.engagements)) +
        tr('Engagement per hour', li.velocity == null ? '—' : li.velocity) +
        (li.collected != null ? tr('Share of its life collected', Math.round(li.collected * 100) + '%') : '') +
        (li.halfLifeUsedMin ? tr('Half-life used', (li.halfLifeUsedMin / 60).toFixed(1) + ' hours') : '') +
        '</tbody></table></div>';
      if (li.collected != null) {
        present += '<div style="height:8px;background:var(--surface2);border-radius:5px;overflow:hidden;margin-top:10px">' +
          '<div style="height:100%;width:' + Math.min(100, Math.round(li.collected * 100)) + '%;background:var(--gold)"></div></div>';
      }
      if (li.notes.length) present += '<div style="font-size:11px;color:var(--text3);margin-top:8px;line-height:1.7">' + esc(li.notes.join(' ')) + '</div>';
    }

    /* ---- future ---- */
    let future = '<div style="font-size:13px;font-weight:600;margin:16px 0 6px">What is still to come</div>';
    if (r.future.lifetime != null) {
      future += '<div class="table-wrap"><table><tbody>' +
        tr('Expected lifetime engagement', '<strong>' + fmtN(r.future.lifetime) + '</strong> ' + basisBadge(r.future.basis)) +
        tr('Still to arrive', fmtN(r.future.remaining)) +
        '</tbody></table></div>';
    } else if (li.lifetimeFloor != null && li.stage === 'likely finished') {
      future += '<div class="table-wrap"><table><tbody>' +
        tr('Lifetime engagement', '<strong>' + fmtN(li.lifetimeFloor) + '</strong> <span class="badge badge-green">measured, not extrapolated</span>') +
        tr('Still to come', 'very little \u2014 ' + (li.ageDays || 0) + ' days old') +
        '</tbody></table></div>' +
        '<div style="font-size:11px;color:var(--text3);margin-top:6px">There is no published half-life for this platform, so no curve was assumed. ' +
        'Re-read the figures to confirm it has stopped.</div>';
    } else {
      future += '<div class="callout callout-info">A lifetime total cannot be estimated for this post yet. ' +
        'Use <strong>Re-read figures</strong> below: with two readings the app measures its actual decay instead of assuming one.</div>';
    }
    const sim = r.future.similar;
    future += '<div style="font-size:12px;color:var(--text2);margin-top:10px">' +
      (sim.ok
        ? 'A future post like this one: around <strong>' + sim.er + '%</strong> engagement (seen between ' +
          sim.erLow + '% and ' + sim.erHigh + '%) and about <strong>' + fmtN(sim.reach) + '</strong> reach. ' +
          esc(sim.why) + (sim.note ? ' ' + esc(sim.note) : '')
        : esc(sim.why)) + '</div>';

    const head = '<div style="font-size:12px;color:var(--text2);margin-bottom:14px">' +
      esc(plabel(p.platform)) +
      (cls ? ' · ' + esc(cls) : '') +
      (p.posted_at ? ' · posted ' + esc(String(p.posted_at).replace('T', ' ')) : ' · no posting time recorded') + '</div>';

    showModal('Post analytics — ' + p.title, head + past + present + future,
      '<button class="btn btn-ghost" onclick="closeModal()">Close</button>' +
      '<button class="btn btn-ghost" onclick="fyReread(\'' + p.id + '\')">Re-read figures</button>' +
      '<button class="btn btn-primary" onclick="openAddPost(\'' + p.id + '\')">Edit post</button>');
  };

  function tr(k, v) {
    return '<tr><td style="color:var(--text2)">' + esc(k) + '</td><td style="text-align:right">' + v + '</td></tr>';
  }

  /* ---- re-read: a second reading turns a modelled curve into a measured one ---- */

  window.fyReread = function (id) {
    const p = findPost(id);
    if (!p) return;
    const now = new Date(); now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    const current = M.engagements(p);
    showModal('Re-read figures — ' + p.title,
      '<div class="callout callout-info" style="margin-bottom:14px">Open the post in your analytics and enter what it shows <em>now</em>. ' +
      'With two readings taken at different times, the app measures how fast this post actually settles instead of ' +
      'assuming the published average — and the lifetime estimate becomes a measured figure.</div>' +
      '<div class="mrow"><label>Read at</label><input type="datetime-local" id="rr-at" value="' + now.toISOString().slice(0, 16) + '"/></div>' +
      '<div class="mrow"><label>Total interactions now (likes + comments + shares + saves)</label>' +
      '<input type="number" id="rr-value" placeholder="' + (current == null ? '' : current) + '"/>' +
      '<div style="font-size:11px;color:var(--text3);margin-top:4px">At the last reading this post had ' +
      (current == null ? 'no interaction counts entered' : fmtN(current)) + '.</div></div>' +
      '<div id="rr-status" style="font-size:12px;min-height:18px;color:var(--text2)"></div>',
      '<button class="btn btn-ghost" onclick="fyPostReport(\'' + p.id + '\')">Back</button>' +
      '<button class="btn btn-primary" onclick="fySaveReading(\'' + p.id + '\')">Save reading</button>');
  };

  window.fySaveReading = async function (id) {
    const c = DB.active(), p = findPost(id);
    const st = $('rr-status');
    if (!p) return;
    const at = ($('rr-at') || {}).value;
    const value = Number(($('rr-value') || {}).value);
    const fail = msg => { if (st) { st.textContent = msg; st.style.color = 'var(--red)'; } };
    if (!at) return fail('Enter when you read the figures.');
    if (!isFinite(value) || value <= 0) return fail('Enter the total interactions the post shows now.');
    if (!p.posted_at) return fail('This post has no posting time, so a reading cannot be placed on its timeline. Add one with Edit post.');
    if (Date.parse(at) <= Date.parse(p.posted_at)) return fail('The reading must be later than when the post went out.');

    p.readings = (p.readings || []).slice();
    /* The figures already on the post are the first reading. */
    if (!p.readings.length) {
      const first = M.engagements(p);
      if (first != null) p.readings.push({ at: p.measured_at || p.posted_at, value: first, note: 'figures as first entered' });
    }
    p.readings.push({ at: at, value: value });
    p.readings.sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
    await DB.save(c);
    if (typeof logEvent === 'function') logEvent('post_reread', { client: c.id, post: p.id });
    if (typeof toast === 'function') toast('Reading saved · decay now measured from ' + p.readings.length + ' readings', 'success');
    window.fyPostReport(p.id);
  };

  /* ------------------------------------------------------------------
     Analyze button in the posts table
     ------------------------------------------------------------------ */

  function addAnalyzeColumn() {
    const body = $('posts-body');
    if (!body) return;
    const head = body.parentElement && body.parentElement.querySelector('thead tr');
    if (head && !head.querySelector('.mkt-th')) {
      const th = document.createElement('th');
      th.className = 'mkt-th'; th.style.width = '1%';
      head.appendChild(th);
    }
    const c = activeClient();
    const logs = c ? (c.posts_log || []).slice().sort((a, b) => b.reach - a.reach) : [];
    Array.prototype.forEach.call(body.querySelectorAll('tr'), (tr, i) => {
      if (tr.querySelector('.mkt-td')) return;
      if (tr.cells.length === 1) return;                   /* the empty-state row */
      const p = logs[i];
      const td = document.createElement('td');
      td.className = 'mkt-td';
      td.innerHTML = p
        ? '<button class="btn btn-ghost" style="font-size:11px;padding:4px 9px;white-space:nowrap" onclick="fyPostReport(\'' +
          p.id + '\',event)">Analyze</button>'
        : '';
      tr.appendChild(td);
    });
  }

  /* ------------------------------------------------------------------
     Demo client
     ------------------------------------------------------------------ */

  const DEMO_ID_BASE = 'demo-sample-client';
  const DAY = 86400000;

  function iso(ms) { return new Date(ms).toISOString(); }
  function localISO(ms) { const d = new Date(ms); d.setMinutes(d.getMinutes() - d.getTimezoneOffset()); return d.toISOString().slice(0, 16); }
  function weekStart(ms) { return iso(ms).slice(0, 10); }

  /** Six weeks of plausible figures, two platforms, and eight posts. */
  function buildDemo(now) {
    const t = now || Date.now();
    const series = {};
    /* 26 weeks: 20 earlier weeks with a gentle climb, a repeating rhythm and ordinary noise,
       then the original 6 recent weeks. Enough history for the accuracy test to run. */
    const EARLY = 20;
    const mk = (base, grow, erSeq, reachBase, early) => {
      const s = emptySeries();
      for (let i = 0; i < EARLY + 6; i++) {
        const j = i - EARLY;                         /* j >= 0: the recent six weeks */
        const when = t - (EARLY + 6 - i) * 7 * DAY;
        const fol = j >= 0 ? Math.round(base + grow * j) : Math.round(base * (0.7 + 0.3 * i / EARLY));
        const reach = j >= 0 ? Math.round(reachBase * (1 + 0.08 * j) * (j === 3 ? 1.6 : 1))
                             : Math.round(reachBase * (0.62 + 0.019 * i) * (1 + 0.05 * Math.sin(i)));
        const posts = j >= 0 ? [3, 4, 4, 5, 4, 5][j] : 3 + ((i * 5) % 3);
        const er = j >= 0 ? erSeq[j] : +early(i).toFixed(1);
        const engTotal = Math.round(reach * er / 100);
        const likes = Math.round(engTotal * 0.78), comments = Math.round(engTotal * 0.06);
        const shares = Math.round(engTotal * 0.06), saves = engTotal - likes - comments - shares;
        s.dates.push(weekStart(when)); s.weeks.push(weekStart(when).slice(5));
        s.er.push(er); s.followers.push(fol); s.reach.push(reach);
        s.clicks.push(Math.round(reach * 0.012)); s.posts.push(posts);
        s.likes.push(likes); s.saves.push(Math.max(0, saves)); s.shares.push(shares); s.comments.push(comments);
        s.paid.push(j === 4 ? 60 : (j < 0 && i % 9 === 4 ? 40 : 0));
      }
      return s;
    };
    const wob = i => ((i * 37) % 11 - 5) * 0.04;
    series.tiktok = mk(3200, 180, [4.1, 4.6, 4.4, 6.2, 5.1, 5.4], 9000, i => 4.0 + 0.02 * i + 0.55 * Math.sin(i * 0.9) + wob(i));
    series.instagram = mk(1850, 70, [3.2, 3.0, 3.6, 4.4, 3.9, 4.1], 2600, i => 3.0 + 0.02 * i + 0.4 * Math.sin(i * 0.9 + 1) + wob(i + 3));

    const P = (hoursAgo, platform, title, cls, reach, er, extra) => {
      const when = t - hoursAgo * 3600000;
      const eng = Math.round(reach * er / 100);
      const likes = Math.round(eng * 0.78), comments = Math.round(eng * 0.06);
      const shares = Math.round(eng * 0.06), saves = Math.max(0, eng - likes - comments - shares);
      const d = new Date(when);
      return Object.assign({
        id: 'demo-p' + hoursAgo + '-' + platform, platform: platform, title: title, cls: cls,
        reach: reach, likes: likes, comments: comments, shares: shares, saves: saves, er: er,
        posted_at: localISO(when), dow: (d.getDay() + 6) % 7, hour: d.getHours(),
        hourBucket: Math.min(5, Math.max(0, Math.round((d.getHours() - 6) / 3)))
      }, extra || {});
    };

    const posts = [
      /* Fresh, still surging - shows the "present" stage and a lifetime estimate. */
      P(3, 'instagram', 'DEMO · Reel: venue setup in 15 seconds', 0, 1400, 5.8,
        { profile_visits: 61, new_follows: 18, link_clicks: 22, views_3s: 1050, completion_rate: 38, measured_at: localISO(t) }),
      /* Yesterday, mid-life. */
      P(26, 'tiktok', 'DEMO · POV: the bar nobody expected', 2, 24000, 7.4,
        { profile_visits: 940, new_follows: 310, link_clicks: 180, views_3s: 17800, completion_rate: 44, measured_at: localISO(t) }),
      /* A week old, with two readings so the decay is MEASURED, not modelled. */
      P(7 * 24, 'tiktok', 'DEMO · Signature cocktail, three ways', 1, 41000, 8.1,
        { profile_visits: 1620, new_follows: 540, link_clicks: 300, views_3s: 31000, completion_rate: 47,
          measured_at: localISO(t),
          readings: [
            { at: localISO(t - 7 * 24 * 3600000 + 6 * 3600000), value: 1900, note: 'read 6 hours after posting' },
            { at: localISO(t), value: 3321 }
          ] }),
      P(10 * 24, 'instagram', 'DEMO · Carousel: wedding package pricing', 4, 3100, 2.9,
        { profile_visits: 120, new_follows: 22, link_clicks: 74, spend: 60, revenue: 480, conversions: 4 }),
      P(14 * 24, 'tiktok', 'DEMO · Behind the scenes: 400-guest build', 3, 12500, 5.2,
        { profile_visits: 380, new_follows: 95, views_3s: 8900, completion_rate: 36 }),
      P(21 * 24, 'instagram', 'DEMO · Static: corporate event menu', 5, 1900, 1.8, { profile_visits: 44, new_follows: 6 }),
      P(28 * 24, 'tiktok', 'DEMO · Announcement: autumn dates open', 4, 7600, 3.4,
        { profile_visits: 210, new_follows: 41, link_clicks: 96 }),
      P(35 * 24, 'tiktok', 'DEMO · Product: espresso martini tower', 1, 18800, 6.6,
        { profile_visits: 700, new_follows: 190, views_3s: 13900, completion_rate: 41 })
    ];

    const id = (typeof scopeId === 'function' ? scopeId(DEMO_ID_BASE) : DEMO_ID_BASE);
    return {
      id: id, name: 'DEMO — Sample Client', industry: 'Demo data · safe to delete',
      location: 'Sample', bio: 'Generated sample client for testing. Every figure is invented.',
      avatar: 'DM', color: '#9B7FE8', primary: 'tiktok', platforms: ['tiktok', 'instagram'],
      handles: { tiktok: 'demo_sample', instagram: 'demo_sample' }, profile_urls: {},
      demographics: null, series: series, posts_log: posts, is_demo: true,
      created_at: iso(t)
    };
  }

  window.fyLoadDemo = async function () {
    const demo = buildDemo(Date.now());
    const existing = allClients().find(c => c.id === demo.id);
    if (existing) {
      showModal('Demo client already loaded',
        '<div class="callout callout-info">A demo client is already in this workspace. Reloading it replaces its figures with a fresh set dated from today.</div>',
        '<button class="btn btn-ghost" onclick="closeModal()">Cancel</button>' +
        '<button class="btn btn-primary" onclick="fyLoadDemoConfirm()">Reload it</button>');
      return;
    }
    await fyLoadDemoConfirm();
  };

  window.fyLoadDemoConfirm = async function () {
    const demo = buildDemo(Date.now());
    await DB.save(demo);
    DB.setActive(demo.id);
    if (typeof logEvent === 'function') logEvent('demo_client_loaded', { client: demo.id });
    if (typeof closeModal === 'function') closeModal();
    if (typeof renderClients === 'function') renderClients();
    if (typeof goView === 'function') goView('dashboard');
    if (typeof renderDashboard === 'function') await renderDashboard();
    const el = $('tb-client'); if (el) el.textContent = demo.name;
    const av = $('tb-avatar'); if (av) av.textContent = demo.avatar;
    if (typeof toast === 'function') toast('Demo client loaded · 6 weeks and 8 posts · delete it like any other client', 'success', 5000);
  };

  function addDemoRow() {
    /* Settings uses .settings-row blocks, so the demo action matches them and
       sits with the other data actions rather than in a table of its own. */
    const anchor = Array.prototype.find.call(
      document.querySelectorAll('#view-settings .settings-row'),
      row => /Restore from backup/.test(row.textContent));
    if (!anchor || document.querySelector('.mkt-demo-row')) return;
    const row = document.createElement('div');
    row.className = 'settings-row mkt-demo-row';
    row.innerHTML = '<div><div class="settings-label">Load demo client</div>' +
      '<div class="settings-desc">6 weeks and 8 posts of invented sample data, including one post read twice ' +
      'so measured decay can be seen. Delete it like any other client.</div></div>' +
      '<button class="btn btn-ghost" style="font-size:11px;padding:5px 10px" onclick="fyLoadDemo()">\u2726 Demo</button>';
    anchor.parentNode.insertBefore(row, anchor.nextSibling);
  }

  /* ------------------------------------------------------------------
     Hook into the existing renderers
     ------------------------------------------------------------------ */

  function ensureHosts() {
    const snap = $('snapshot-card');
    if (snap && !$('mkt-instant')) {
      const d = document.createElement('div'); d.id = 'mkt-instant';
      snap.parentNode.insertBefore(d, snap);
    }
    const ins = $('insights-container');
    if (ins && !$('mkt-future')) {
      const d = document.createElement('div'); d.id = 'mkt-future';
      ins.parentNode.insertBefore(d, ins);
    }
  }

  function wrap(name, after) {
    const orig = window[name];
    if (typeof orig !== 'function' || orig.__mkt) return;
    const fn = function () {
      const r = orig.apply(this, arguments);
      if (r && typeof r.then === 'function') return r.then(v => { try { after(); } catch (e) { console.warn(name, e); } return v; });
      try { after(); } catch (e) { console.warn(name, e); }
      return r;
    };
    fn.__mkt = true;
    window[name] = fn;
  }

  function boot() {
    ensureHosts();
    wrap('renderDashboard', () => { renderInstant(); addAnalyzeColumn(); });
    wrap('renderAI', renderFuture);
    wrap('goView', () => { ensureHosts(); addDemoRow(); });
    addDemoRow();
    try { renderInstant(); addAnalyzeColumn(); } catch (e) { /* nothing rendered yet */ }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => setTimeout(boot, 0));
  else setTimeout(boot, 0);

  window.FYMktUI = { renderInstant: renderInstant, renderFuture: renderFuture, buildDemo: buildDemo,
                     addAnalyzeColumn: addAnalyzeColumn, instantCard: instantCard };
})();
