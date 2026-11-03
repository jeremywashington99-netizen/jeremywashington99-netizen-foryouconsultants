/* ==========================================================================
   SEGMENT 6 - Site wiring (browser only)

   Connects segments 1-5 to the Four You Consultants page:
     AI Insights  -> Deep Learning Lab card (RNN forecast, CNN images, GAN data)
     Clients      -> a Backup button on every client card
     Settings     -> Back up a client / all clients, Restore from backup

   Uses the page's existing globals: tf, sb, SESSION, DB, PLATFORMS,
   CONTENT_CLASSES, toast, esc, setHTML, download, showModal, closeModal,
   todayISO, logEvent, mkChart, baseOpts, renderClients, renderDashboard.
   ========================================================================== */
(function () {
  'use strict';
  const A = FYDL.activations, R = FYDL.rnn, C = FYDL.cnn, G = FYDL.gan;
  const LAB = { tab: 'rnn', rnn: null, rnnClient: null, cnn: null, cnnImages: [], cnnTask: 'multiclass', gan: null, ganRows: [], ganReal: [] };
  window.FYLab = LAB;

  function el(id) { return document.getElementById(id); }
  function platName(pk) { return (typeof PLATFORMS !== 'undefined' && PLATFORMS[pk] && (PLATFORMS[pk].label || PLATFORMS[pk].name)) || pk; }
  function client() { return DB.active(); }
  function platformsWithWeeks(c, min) {
    return (c && c.platforms || []).filter(function (pk) { const s = c.series[pk]; return s && s.er && s.er.length >= min; });
  }

  /* ---------- audit table (shows the activation policy is met) ---------- */
  function auditHTML(model, task, title) {
    const a = A.auditModel(model, task);
    const rows = a.hidden.map(function (h) {
      return '<tr><td>' + esc(h.name) + '</td><td>' + esc(h.type) + '</td><td>hidden</td><td>' + esc(h.activation === 'leaky' ? 'Leaky ReLU' : h.activation === 'relu' ? 'ReLU' : h.activation) + '</td></tr>';
    }).join('') + '<tr><td>' + esc(a.output.name) + '</td><td>Dense</td><td>output</td><td>' + esc(a.output.activation) + '</td></tr>';
    return '<div class="card-sub" style="margin:12px 0 6px">' + esc(title) + ' &middot; ' +
      (a.ok ? '<span style="color:var(--green)">activation policy met</span>' : '<span style="color:var(--red)">' + esc(a.problems.join('; ')) + '</span>') +
      ' &middot; ' + model.countParams().toLocaleString() + ' parameters</div>' +
      '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Layer</th><th>Type</th><th>Role</th><th>Activation</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  /* ---------- lab shell ---------- */
  LAB.render = function () {
    const box = el('dl-lab'); if (!box) return;
    const tabs = [['rnn', 'RNN · weekly forecast'], ['cnn', 'CNN · post images'], ['gan', 'GAN · synthetic weeks']];
    box.innerHTML =
      '<div class="card" style="margin-bottom:14px">' +
      '<div class="card-header"><div><div class="card-title">Deep Learning Lab</div>' +
      '<div class="card-sub">Hidden layers ReLU / Leaky ReLU &middot; output activation matched to each goal &middot; every run is saved to your database, trains on this client only</div></div></div>' +
      '<div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px">' +
      tabs.map(function (t) {
        return '<button class="btn ' + (LAB.tab === t[0] ? 'btn-primary' : 'btn-ghost') + '" style="font-size:12px;padding:6px 12px" onclick="FYLab.show(\'' + t[0] + '\')">' + t[1] + '</button>';
      }).join('') + '</div><div id="dl-pane"></div></div>';
    ({ rnn: renderRNN, cnn: renderCNN, gan: renderGAN })[LAB.tab]();
  };
  LAB.show = function (t) { LAB.tab = t; LAB.render(); };

  /* ---------- RNN: the weekly forecast, in marketing terms ---------- */
  const Store = window.FYStore, Ex = window.FYExplain;
  function ownerId() { return SESSION && SESSION.user ? SESSION.user.id : null; }
  function erIndex() { return R.FEATURES.indexOf(R.TARGET); }

  function platformReport(c, pk) {
    const s = c.series[pk], t = LAB.rnn;
    const bt = LAB.bt && LAB.bt[pk];
    const f = (bt ? bt.forecast.slice(0, 4) : R.forecastRNN(tf, t, s, pk, 4)).map(function (v) { return Math.max(0, v); });
    const posts = (c.posts_log || []).filter(function (p) { return p.platform === pk; });
    const r = Ex.explainForecast({ platform: pk, platformLabel: platName(pk), series: s, forecast: f, posts: posts,
      classNames: CONTENT_CLASSES, missPoints: Ex.missInPoints(t.maeScaled, t.scalers[pk], erIndex()) });
    if (bt && r.ok) {
      /* measured, not assumed: the range is the real size of past misses at that distance */
      r.rows = r.rows.map(function (x, i) { return Object.assign({}, x, { low: Math.max(0, +(x.expected - bt.bands[i]).toFixed(2)), high: +(x.expected + bt.bands[i]).toFixed(2) }); });
      r.reliability = { level: bt.verdict.status, label: bt.verdict.label, why: [bt.verdict.headline].concat(bt.verdict.why) };
      r.method = bt.best; r.candidates = bt.candidates; r.metrics = bt.metrics; r.assessment = bt.assessment; r.folds = bt.folds; r.verified = bt.verdict.verified;
      r.learnedFrom = 'Method used: ' + bt.best + (bt.folds ? ' \u2014 it had the smallest miss of ' + Object.keys(bt.candidates).length + ' methods tested on ' + bt.folds + ' real past stretches of 4 weeks.' : '.');
    }
    return r;
  }
  function reportHTML(r) {
    if (!r.ok) return '<div class="callout callout-info" style="margin-top:12px">' + esc(r.headline) + '</div>';
    const col = r.direction === 'up' ? 'var(--green)' : r.direction === 'down' ? 'var(--amber)' : 'var(--text)';
    const rel = r.reliability;
    const relCol = (rel.level === 'strong' || rel.level === 'useful' || rel.level === 'verified') ? 'var(--green)' : rel.level === 'unreliable' ? 'var(--red)' : 'var(--amber)';
    const rows = r.rows.map(function (x) {
      return '<tr><td>In ' + x.week + ' week' + (x.week === 1 ? '' : 's') + '</td><td><strong>' + x.expected + '%</strong></td><td>' +
        (x.low == null ? '&mdash;' : x.low + '% to ' + x.high + '%') + '</td></tr>';
    }).join('');
    return '<div style="margin-top:14px;padding:14px;border:1px solid var(--border);border-radius:10px">' +
      '<div style="font-size:15px;font-weight:600;color:' + col + '">' + esc(r.headline) + '</div>' +
      '<div style="overflow-x:auto;margin:10px 0"><table class="data-table"><thead><tr><th>When</th><th>Expected engagement</th><th>Likely range</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<div style="margin-bottom:10px"><span class="badge" style="color:' + relCol + '">' + esc(rel.label) + '</span> ' +
      '<span class="card-sub">' + esc(rel.why.join(' ')) + '</span></div>' +
      '<div class="card-sub" style="margin-bottom:6px">' + esc(r.learnedFrom) + '</div>' +
      metricsHTML(r) +
      '<div style="font-weight:600;margin-bottom:4px">What to do next</div><ul class="plain-list" style="margin:0 0 0 18px">' +
      r.doNext.map(function (d) { return '<li>' + esc(d) + '</li>'; }).join('') + '</ul></div>';
  }

  /* The accuracy evidence: standard error measures per method, and the plain assessment table. */
  function metricsHTML(r) {
    if (!r.metrics || !Object.keys(r.metrics).length) return '';
    const names = { 'last week repeated': 'Last week repeated (baseline)', 'recent average': 'Recent average', 'damped trend': 'Damped trend', 'fast-adapting level': 'Fast-adapting level', 'since last shift': 'Level since last shift', 'blend of level and trend': 'Blend of level and trend', 'neural network': 'Neural network' };
    const rows = Object.keys(r.metrics).map(function (k) {
      const m = r.metrics[k], win = k === r.method;
      return '<tr' + (win ? ' style="font-weight:600"' : '') + '><td>' + esc(names[k] || k) + (win ? ' &#10003; used' : '') + '</td><td>' + m.mae + '</td><td>' + m.rmse + '</td><td>' + m.mape + '%</td><td>' +
        (m.r2 == null ? '&mdash;' : m.r2) + '</td><td>' + (m.gain == null ? '&mdash;' : (m.gain > 0 ? '+' : '') + m.gain + '%') + '</td></tr>';
    }).join('');
    const ass = (r.assessment || []).map(function (a) { return '<tr><td>' + esc(a.category) + '</td><td><strong>' + esc(a.rating) + '</strong></td><td class="card-sub">' + esc(a.note) + '</td></tr>'; }).join('');
    return '<details style="margin:6px 0 10px"><summary class="card-sub" style="cursor:pointer">Accuracy evidence: how each method did on weeks it had not seen</summary>' +
      '<div style="overflow-x:auto;margin-top:8px"><table class="data-table"><thead><tr><th>Method</th><th>Avg miss (MAE)</th><th>Big-miss penalty (RMSE)</th><th>Avg % miss (MAPE)</th><th>R&sup2; on new weeks</th><th>Better than baseline</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<div class="card-sub" style="margin:6px 0">A method has to beat "next week = last week" to be worth using; the one with the smallest miss is chosen. Misses are in engagement points.</div>' +
      (ass ? '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Check</th><th>Rating</th><th>Why</th></tr></thead><tbody>' + ass + '</tbody></table></div>' : '') + '</details>';
  }

  function renderRNN() {
    const c = client();
    const ready = platformsWithWeeks(c, 4);
    let h = '<div class="callout callout-info" style="margin-bottom:12px"><strong>Weekly engagement forecast.</strong> This reads the last few weeks of each platform &mdash; engagement, followers, reach, posting frequency, saves and paid boosts &mdash; and projects where engagement is heading over the next 4 weeks, with a realistic range. Every time it learns, the result is saved to your database.</div>';
    if (!ready.length) { setHTML('dl-pane', h + '<div class="callout callout-warn">Needs at least 4 weeks of results on one platform.</div>'); return; }
    h += '<button class="btn btn-primary" id="rnn-btn" onclick="FYLab.trainRNN()">Update forecast</button> <span class="card-sub" id="rnn-status">' +
      (LAB.rnnLoading ? 'Loading your saved forecast&hellip;' : '') + '</span>';
    const live = LAB.rnn && LAB.rnnClient === c.id;
    if (LAB.rnnError) h += '<div class="callout callout-warn" style="margin-top:12px"><strong>The forecast did not finish.</strong> ' + esc(LAB.rnnError) + '</div>';
    if (live) {
      h += LAB.rnn.platforms.map(function (pk) { try { return reportHTML(platformReport(c, pk)); } catch (e) { return ''; } }).join('');
      const st = Store.staleness(LAB.rnnRun, Math.max.apply(null, ready.map(function (pk) { return c.series[pk].er.length; })), (c.posts_log || []).length);
      h += '<div class="card-sub" style="margin-top:10px">' + (LAB.rnnSaveError
        ? '<span style="color:var(--red)">Not saved to the database: ' + esc(LAB.rnnSaveError) + '</span>'
        : LAB.rnnSavedAt ? '<span style="color:var(--green)">&#10003; Saved to your database ' + esc(new Date(LAB.rnnSavedAt).toLocaleString()) + '</span>' : '') +
        (st.stale ? ' &middot; ' + esc(st.reason) + ' &mdash; press Update forecast.' : '') + '</div>';
      if (LAB.rnnGuard && window.FYGuard) h += FYGuard.html(LAB.rnnGuard, esc);
      h += '<details style="margin-top:10px"><summary class="card-sub" style="cursor:pointer">Technical detail (for analysts)</summary>' +
        '<div class="card-sub" style="margin:8px 0">Learned from ' + LAB.rnn.windows + ' examples of ' + LAB.rnn.lookback + ' weeks in a row.</div>' +
        '<div class="chart-wrap" style="height:150px"><canvas id="c-rnn-loss"></canvas></div>' +
        techTable(LAB.rnn.model) + '</details>';
    }
    setHTML('dl-pane', h);
    if (live) {
      mkChart('c-rnn-loss', 'line', { labels: LAB.rnn.history.map(function (_, i) { return i; }),
        datasets: [{ data: LAB.rnn.history, borderColor: '#4FC3F7', pointRadius: 0, borderWidth: 2, tension: .3 }] }, baseOpts(function (v) { return v.toFixed(3); }));
    } else if (!LAB.rnnLoading && !LAB.rnnTried[c.id]) { LAB.autoForecast(c); }
  }

  /* Plain-English layer table; the exact layer names sit in the last column for analysts. */
  function techTable(model) {
    const a = A.auditModel(model, 'regression');
    const row = function (name, type, role, act) {
      const p = Ex.layerPlain({ type: type, role: role, activation: act }), ap = Ex.activationPlain(act);
      return '<tr><td>' + esc(p.stage) + '</td><td>' + esc(p.does) + '</td><td>' + esc(ap.label) + (ap.plain ? ' &mdash; ' + esc(ap.plain) : '') + '</td><td style="font-family:monospace;font-size:11px;color:var(--text2)">' + esc(name) + '</td></tr>';
    };
    const rows = a.hidden.map(function (h) { return row(h.name, h.type, 'hidden', h.activation); }).join('') + row(a.output.name, 'Dense', 'output', a.output.activation);
    return '<div class="card-sub" style="margin:8px 0">' + (a.ok ? '<span style="color:var(--green)">Uses only the approved activation rules</span>' : '<span style="color:var(--red)">' + esc(a.problems.join('; ')) + '</span>') +
      ' &middot; ' + model.countParams().toLocaleString() + ' adjustable values</div>' +
      '<div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Stage</th><th>What it does</th><th>Rule it uses</th><th>Layer</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  /* The same ten safeguards, for the weekly forecast. */
  function rnnGuard(c, t) {
    const G = window.FYGuard; if (!G) return null;
    const issues = { issues: [], bad: {} };
    t.platforms.forEach(function (pk) { G.validateSeries(c.series[pk]).issues.forEach(function (x) { issues.issues.push(Object.assign({ platform: pk }, x)); }); });
    const pk0 = t.platforms[0];
    let rows = [];
    t.platforms.forEach(function (pk) { rows = rows.concat(R.scaleMatrix(R.seriesToMatrix(c.series[pk]), t.scalers[pk])); });
    return G.audit({ history: t.history, trainErr: t.maeScaled, holdErr: null, holdN: 0, r2: null, rows: t.windows, classCounts: null,
      dataIssues: issues, leak: null, health: G.weightsHealth(t.model.getWeights().map(function (w) { return w.dataSync(); })),
      drift: G.drift(c.series[pk0]), scale: G.scaling(rows), weeks: c.series[pk0].er.length });
  }

  const BT = window.FYBacktest;
  LAB.bt = {};
  const METHOD_ORDER = ['last week repeated', 'recent average', 'damped trend', 'fast-adapting level', 'since last shift', 'blend of level and trend', 'neural network'];

  /* Walk-forward test of every method on each platform; picks the one with the smallest real miss. */
  async function runBacktests(c, t) {
    const out = {};
    for (const pk of t.platforms) {
      const s = c.series[pk], vals = s.er.map(Number);
      const fns = Object.assign({}, BT.SIMPLE);
      fns['neural network'] = async function (hist, h, k) {
        const sub = {}; Object.keys(s).forEach(function (key) { sub[key] = Array.isArray(s[key]) ? s[key].slice(0, k) : s[key]; });
        const one = {}; one[pk] = sub;
        const m = await R.trainRNN(tf, one, { epochs: 60 });
        try { return R.forecastRNN(tf, m, sub, pk, h).map(function (v) { return Math.max(0, v); }); } finally { m.model.dispose(); }
      };
      setText('rnn-status', 'Testing the forecast against your past weeks (' + platName(pk) + ')\u2026');
      const bt = await BT.run(vals, fns);
      let best = 'neural network', bands = null, candidates = {}, metricsByMethod = {}, changing = false;
      if (bt.ok) {
        const dr = window.FYGuard ? FYGuard.drift(s) : { state: 'unknown' };
        changing = dr.state === 'shifted' || dr.state === 'moving';
        best = BT.choose(bt, METHOD_ORDER, changing) || 'neural network';
        METHOD_ORDER.forEach(function (k) { const x = bt.results[k]; if (x && !x.failed) { candidates[k] = x.mae; metricsByMethod[k] = { mae: x.mae, rmse: x.rmse, mape: x.mape, r2: x.r2, gain: BT.skill(bt, k) }; } });
        bands = BT.bands(bt.results[best], 6, changing);
      }
      const forecast = (best === 'neural network' ? R.forecastRNN(tf, t, s, pk, 6) : BT.SIMPLE[best](vals, 6)).map(function (v) { return +Math.max(0, v).toFixed(2); });
      const v = BT.verdict(bt, best, vals);
      out[pk] = { best: best, verdict: v, forecast: forecast, bands: bands, candidates: candidates, metrics: metricsByMethod, shift: bt.ok ? BT.shiftTest(bt, best) : null, changing: changing, assessment: BT.assess(bt, best, vals, forecast, v), folds: bt.ok ? bt.folds : 0,
                  perHorizon: bt.ok ? bt.results[best].perHorizon : null, weeks: vals.length };
    }
    return out;
  }

  /* Used by the dashboard: the tested forecast for a platform, or null. */
  LAB.testedForecast = function (clientId, pk) { return LAB.rnnClient === clientId && LAB.bt && LAB.bt[pk] ? LAB.bt[pk] : null; };

  /* Open a client: restore the saved forecast, or build and test one. */
  LAB.autoForecast = async function (c) {
    if (!c || !Store || platformsWithWeeks(c, 4).length === 0) return;
    if (!LAB.rnnTried[c.id]) await LAB.loadSavedRNN();
    if (!(LAB.rnn && LAB.rnnClient === c.id) && !LAB.rnnBusy) await LAB.trainRNN();
  };

  LAB.rnnTried = {};
  LAB.loadSavedRNN = async function () {
    const c = client(); if (!c || !Store) return;
    LAB.rnnTried[c.id] = true; LAB.rnnLoading = true; renderRNN();
    try {
      const r = await Store.latest(sb, c.id, 'rnn');
      if (r.ok && r.run && r.run.weights && r.run.weights.model) {
        const w = r.run.weights;
        if (LAB.rnn) LAB.rnn.model.dispose();
        LAB.rnn = { model: await Store.restoreModel(tf, w.model), scalers: w.scalers, lookback: w.lookback, platforms: w.platforms,
                    history: w.history || [], windows: w.windows, task: w.task || 'regression', maeScaled: w.maeScaled };
        LAB.rnnGuard = w.guard || null; LAB.bt = w.backtest || {};
        LAB.rnnClient = c.id; LAB.rnnRun = r.run; LAB.rnnSavedAt = r.run.trained_at; LAB.rnnSaveError = null;
      } else if (!r.ok) LAB.rnnError = 'Could not load your saved forecast: ' + r.error;
    } catch (e) { LAB.rnnError = 'Could not rebuild the saved forecast: ' + e.message; }
    LAB.rnnLoading = false; renderRNN();
  };

  LAB.trainRNN = async function () {
    const c = client(); const btn = el('rnn-btn');
    const series = {}; platformsWithWeeks(c, 4).forEach(function (pk) { series[pk] = c.series[pk]; });
    if (btn) { btn.disabled = true; btn.textContent = 'Learning...'; }
    if (LAB.rnnBusy) return;
    LAB.rnnBusy = true;
    LAB.rnnError = null; LAB.rnnSaveError = null;
    try {
      if (LAB.rnn) LAB.rnn.model.dispose();
      LAB.rnn = null;
      const t = await R.trainRNN(tf, series, { epochs: 150, onEpoch: function (e, n) {
        if (e % 10 === 0) setText('rnn-status', 'Learning from your results · ' + Math.round(e / n * 100) + '% done'); } });
      LAB.rnn = t; LAB.rnnClient = c.id; LAB.rnnGuard = rnnGuard(c, t);
      LAB.bt = await runBacktests(c, t);
      const reports = {}, fc = {};
      t.platforms.forEach(function (pk) { const r = platformReport(c, pk); reports[pk] = r; fc[pk] = r.rows ? r.rows.map(function (x) { return x.expected; }) : []; });
      const weeks = Math.max.apply(null, t.platforms.map(function (pk) { return c.series[pk].er.length; }));
      setText('rnn-status', 'Saving to your database…');
      const wBlob = { model: await Store.serializeModel(tf, t.model), scalers: t.scalers, lookback: t.lookback, platforms: t.platforms,
                      history: t.history, windows: t.windows, task: t.task, maeScaled: t.maeScaled, guard: LAB.rnnGuard, backtest: LAB.bt };
      const saved = await Store.saveRun(sb, ownerId(), {
        client_id: c.id, kind: 'rnn', status: 'ok', model_name: 'weekly_forecast', backend: (tf.getBackend && tf.getBackend()) || null,
        epochs: t.history.length, loss_final: t.history.length ? t.history[t.history.length - 1] : null, mae: t.maeScaled,
        train_rows: t.windows, data_weeks: weeks, data_posts: (c.posts_log || []).length,
        architecture: { type: 'LSTM', lookback: t.lookback, platforms: t.platforms, params: t.model.countParams() },
        hyperparams: { epochs: 150, hidden: 'relu', output: 'linear' }, metrics: { maeScaled: t.maeScaled, windows: t.windows },
        forecast: fc, summary: { guardrails: LAB.rnnGuard, backtest: LAB.bt, headlines: Object.keys(reports).reduce(function (o, k) { o[k] = reports[k].headline; return o; }, {}), reports: reports },
        activation_audit: A.auditModel(t.model, 'regression'), weights: wBlob });
      if (saved.ok) { LAB.rnnSavedAt = new Date().toISOString(); LAB.rnnRun = { data_weeks: weeks, data_posts: (c.posts_log || []).length }; toast('Forecast updated and saved to your database', 'success'); }
      else { LAB.rnnSaveError = saved.error; toast('Forecast updated, but NOT saved to the database: ' + saved.error, 'error', 8000); }
      logEvent('rnn_trained', { client: c.id, windows: t.windows, lookback: t.lookback });
    } catch (e) { LAB.rnn = null; LAB.rnnError = e.message; toast('Forecast failed: ' + e.message, 'error', 6000); }
    LAB.rnnBusy = false;
    renderRNN();
    try { if (typeof renderDashboard === 'function') await renderDashboard(); } catch (e) {}
    try { if (typeof renderAnalytics === 'function') await renderAnalytics(); } catch (e) {}
    try { if (typeof renderAI === 'function') await renderAI(); } catch (e) {}   /* safeguard 9 uses this test */
  };

  async function saveOther(kind, fields, model) {
    const c = client(); if (!c) return;
    const row = Object.assign({ client_id: c.id, kind: kind, status: 'ok', backend: (tf.getBackend && tf.getBackend()) || null,
      data_weeks: Math.max.apply(null, [0].concat((c.platforms || []).map(function (pk) { return c.series[pk].er.length; }))), data_posts: (c.posts_log || []).length }, fields);
    let r;
    if (model) {
      try { row.weights = await Store.serializeModel(tf, model); } catch (e) { row.weights = null; row.summary = Object.assign({}, row.summary, { weights_note: 'Settings saved; the trained weights were too large to store (' + e.message + ').' }); }
    }
    r = await Store.saveRun(sb, ownerId(), row);
    if (!r.ok) toast('Trained, but NOT saved to the database: ' + r.error, 'error', 8000);
    else toast('Result saved to your database', 'success');
  }

  /* ---------- CNN ---------- */
  function renderCNN() {
    const classes = LAB.cnnTask === 'binary' ? ['Low engagement', 'High engagement'] : CONTENT_CLASSES;
    let h = '<div class="callout callout-info" style="margin-bottom:12px">A convolutional network learns from post images. Pick a goal, add images, label each one, then train. ' +
      'Content class uses a <strong>softmax</strong> output (one class out of ' + CONTENT_CLASSES.length + '); high vs low engagement uses a <strong>sigmoid</strong> output (yes / no). Images stay in your browser.</div>' +
      '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px">' +
      '<select id="cnn-task" class="form-input" style="max-width:260px" onchange="FYLab.setCnnTask(this.value)">' +
      '<option value="multiclass"' + (LAB.cnnTask === 'multiclass' ? ' selected' : '') + '>Goal: content class (softmax)</option>' +
      '<option value="binary"' + (LAB.cnnTask === 'binary' ? ' selected' : '') + '>Goal: high vs low engagement (sigmoid)</option></select>' +
      '<label class="btn btn-ghost" style="font-size:12px">+ Add images<input type="file" accept="image/*" multiple style="display:none" onchange="FYLab.addImages(this)"/></label>' +
      '<button class="btn btn-primary" id="cnn-btn" onclick="FYLab.trainCNN()"' + (LAB.cnnImages.length < 4 ? ' disabled' : '') + '>Train CNN</button>' +
      '<span class="card-sub" id="cnn-status">' + LAB.cnnImages.length + ' image(s) &middot; need 4+ across 2+ labels</span></div>';
    h += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px">' + LAB.cnnImages.map(function (im, i) {
      return '<div style="background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:6px">' +
        '<img src="' + im.url + '" alt="" style="width:100%;height:90px;object-fit:cover;border-radius:6px"/>' +
        '<select class="form-input" style="margin-top:6px;font-size:12px;padding:4px" onchange="FYLab.label(' + i + ',this.value)">' +
        classes.map(function (cl, j) { return '<option value="' + j + '"' + (im.label === j ? ' selected' : '') + '>' + esc(cl) + '</option>'; }).join('') +
        '</select><button class="btn btn-ghost" style="font-size:11px;padding:2px 8px;margin-top:4px;width:100%" onclick="FYLab.removeImage(' + i + ')">Remove</button></div>';
    }).join('') + '</div>';
    if (LAB.cnn) {
      h += '<div style="margin-top:14px;display:flex;gap:10px;align-items:center;flex-wrap:wrap"><label class="btn btn-primary" style="font-size:12px">Predict an image<input type="file" accept="image/*" style="display:none" onchange="FYLab.predictCNN(this)"/></label><span id="cnn-pred" class="card-sub"></span></div>';
      h += auditHTML(LAB.cnn.model, LAB.cnn.task, 'CNN layers');
    }
    setHTML('dl-pane', h);
  }
  LAB.setCnnTask = function (t) {
    LAB.cnnTask = t; LAB.cnnImages.forEach(function (im) { im.label = 0; });
    if (LAB.cnn) { LAB.cnn.model.dispose(); LAB.cnn = null; }
    renderCNN();
  };
  LAB.label = function (i, v) { LAB.cnnImages[i].label = Number(v); };
  LAB.removeImage = function (i) { const im = LAB.cnnImages.splice(i, 1)[0]; if (im) { URL.revokeObjectURL(im.url); im.tensor.dispose(); } renderCNN(); };
  function loadImage(file) {
    return new Promise(function (resolve, reject) {
      const url = URL.createObjectURL(file), img = new Image();
      img.onload = function () {
        const px = tf.browser.fromPixels(img);
        const t = C.preprocessPixels(tf, px, C.DEFAULTS.inputShape[0]); px.dispose();
        resolve({ url: url, tensor: t, name: file.name });
      };
      img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('could not read ' + file.name)); };
      img.src = url;
    });
  }
  LAB.addImages = async function (input) {
    const files = Array.from(input.files || []);
    for (const f of files) {
      try { const im = await loadImage(f); im.label = 0; LAB.cnnImages.push(im); } catch (e) { toast(e.message, 'error'); }
    }
    input.value = ''; renderCNN();
  };
  LAB.trainCNN = async function () {
    const labels = LAB.cnnImages.map(function (im) { return im.label; });
    const distinct = labels.filter(function (v, i, a) { return a.indexOf(v) === i; }).length;
    if (LAB.cnnImages.length < 4 || distinct < 2) { toast('Add at least 4 images with at least 2 different labels', 'error', 5000); return; }
    const btn = el('cnn-btn'); if (btn) { btn.disabled = true; btn.textContent = 'Training...'; }
    const task = LAB.cnnTask, n = task === 'binary' ? 2 : CONTENT_CLASSES.length;
    const xs = tf.stack(LAB.cnnImages.map(function (im) { return im.tensor; }));
    const ys = C.encodeLabels(tf, labels, task, n);
    try {
      if (LAB.cnn) LAB.cnn.model.dispose();
      const model = C.buildCNN(tf, { task: task, nClasses: n });
      await C.trainCNN(tf, model, xs, ys, { epochs: 25, batchSize: 8, onEpoch: function (e, total, logs) {
        setText('cnn-status', 'Epoch ' + (e + 1) + ' / ' + total + ' · loss ' + logs.loss.toFixed(4)); } });
      LAB.cnn = { model: model, task: task };
      logEvent('cnn_trained', { client: (client() || {}).id, images: labels.length, task: task });
      toast('CNN trained on ' + labels.length + ' images', 'success');
      await saveOther('cnn', { model_name: 'post_image_classifier', epochs: 25, train_rows: labels.length, architecture: { task: task, classes: n, params: model.countParams() }, summary: { headline: 'Learned to sort post images into ' + n + ' groups from ' + labels.length + ' labelled images.' }, activation_audit: A.auditModel(model, task) }, model);
    } catch (e) { toast('CNN training failed: ' + e.message, 'error', 6000); }
    finally { xs.dispose(); ys.dispose(); }
    renderCNN();
  };
  LAB.predictCNN = async function (input) {
    const f = input.files && input.files[0]; if (!f || !LAB.cnn) return;
    try {
      const im = await loadImage(f);
      const names = LAB.cnn.task === 'binary' ? null : CONTENT_CLASSES;
      const p = C.predictImage(tf, LAB.cnn.model, im.tensor, names);
      im.tensor.dispose(); URL.revokeObjectURL(im.url);
      setHTML('cnn-pred', LAB.cnn.task === 'binary'
        ? 'High-engagement probability: <strong>' + (p.probability * 100).toFixed(1) + '%</strong>'
        : 'Predicted class: <strong>' + esc(p.label) + '</strong> (' + (p.confidence * 100).toFixed(1) + '%)');
    } catch (e) { toast(e.message, 'error'); }
    input.value = '';
  };

  /* ---------- GAN ---------- */
  function renderGAN() {
    const c = client();
    const ready = platformsWithWeeks(c, 3);
    let h = '<div class="callout callout-info" style="margin-bottom:12px">A generative adversarial network learns what this client’s real weeks look like and generates realistic synthetic weeks for what-if planning. ' +
      'Generator and discriminator hidden layers use <strong>Leaky ReLU</strong>; the generator outputs <strong>sigmoid</strong> (values scaled 0..1), the discriminator outputs <strong>sigmoid</strong> (real vs fake). Synthetic rows are never saved into your real metrics.</div>';
    if (!ready.length) { setHTML('dl-pane', h + '<div class="callout callout-warn">Needs at least 3 weeks on one platform.</div>'); return; }
    h += '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center"><select id="gan-plat" class="form-input" style="max-width:200px">' +
      ready.map(function (pk) { return '<option value="' + pk + '">' + esc(platName(pk)) + '</option>'; }).join('') + '</select>' +
      '<button class="btn btn-primary" id="gan-btn" onclick="FYLab.trainGAN()">Train GAN &amp; generate</button>' +
      (LAB.ganRows.length ? '<button class="btn btn-ghost" onclick="FYLab.downloadGAN()">&#8681; CSV</button>' : '') +
      '<span class="card-sub" id="gan-status"></span></div>';
    if (LAB.ganRows.length) {
      const cols = G.COLUMNS;
      h += '<div class="card-sub" style="margin:14px 0 6px">10 synthetic weeks (' + esc(platName(LAB.ganPlat)) + ') — labelled synthetic</div><div style="overflow-x:auto"><table class="data-table"><thead><tr>' +
        cols.map(function (k) { return '<th>' + k + '</th>'; }).join('') + '</tr></thead><tbody>' +
        LAB.ganRows.map(function (r) { return '<tr>' + cols.map(function (k) { return '<td>' + r[k] + '</td>'; }).join('') + '</tr>'; }).join('') + '</tbody></table></div>';
      const st = G.compareStats(LAB.ganReal, LAB.ganRows);
      h += '<div class="card-sub" style="margin:14px 0 6px">Real vs synthetic averages</div><div style="overflow-x:auto"><table class="data-table"><thead><tr><th>Column</th><th>Real mean</th><th>Synthetic mean</th><th>Real spread</th><th>Synthetic spread</th></tr></thead><tbody>' +
        st.map(function (s) { return '<tr><td>' + s.column + '</td><td>' + s.realMean.toFixed(2) + '</td><td>' + s.synthMean.toFixed(2) + '</td><td>' + s.realStd.toFixed(2) + '</td><td>' + s.synthStd.toFixed(2) + '</td></tr>'; }).join('') + '</tbody></table></div>';
      h += auditHTML(LAB.gan.generator, 'bounded01', 'Generator layers') + auditHTML(LAB.gan.discriminator, 'binary', 'Discriminator layers');
    }
    setHTML('dl-pane', h);
    if (LAB.ganPlat && el('gan-plat')) el('gan-plat').value = LAB.ganPlat;
  }
  LAB.trainGAN = async function () {
    const c = client(), pk = el('gan-plat').value, btn = el('gan-btn');
    const rows = G.seriesToRows(c.series[pk]);
    if (btn) { btn.disabled = true; btn.textContent = 'Training...'; }
    try {
      if (LAB.gan) { LAB.gan.generator.dispose(); LAB.gan.discriminator.dispose(); }
      LAB.gan = await G.trainGAN(tf, rows, { steps: 500, onStep: function (s, n, d, g) {
        setText('gan-status', 'Step ' + s + ' / ' + n + ' · D ' + d.toFixed(3) + ' · G ' + g.toFixed(3)); } });
      LAB.ganRows = G.generate(tf, LAB.gan, 10); LAB.ganReal = rows; LAB.ganPlat = pk;
      logEvent('gan_trained', { client: c.id, platform: pk, real_rows: rows.length });
      toast('Generated 10 synthetic weeks', 'success');
      await saveOther('gan', { model_name: 'synthetic_weeks', platform: pk, epochs: 500, train_rows: rows.length, architecture: { params: LAB.gan.generator.countParams() + LAB.gan.discriminator.countParams() }, forecast: LAB.ganRows, summary: { headline: 'Generated 10 realistic what-if weeks for ' + platName(pk) + ' from ' + rows.length + ' real weeks. These are never mixed into your real results.' } }, null);
    } catch (e) { toast('GAN training failed: ' + e.message, 'error', 6000); }
    renderGAN();
  };
  LAB.downloadGAN = function () {
    const cols = G.COLUMNS;
    const csv = ['synthetic,platform,' + cols.join(',')].concat(LAB.ganRows.map(function (r) { return 'yes,' + LAB.ganPlat + ',' + cols.map(function (k) { return r[k]; }).join(','); })).join('\n');
    download(new Blob([csv], { type: 'text/csv' }), 'synthetic-weeks_' + (client() || {}).id + '_' + todayISO() + '.csv');
  };

  /* ---------- Backup & restore ---------- */
  function connected() { return !!(sb && SESSION && SESSION.user); }

  async function makeBackup(ids) {
    if (connected()) {
      try { return await FYBackup.exportClients(sb, ids); }
      catch (e) { console.warn('database export failed, using the copy on screen', e); }
    }
    const list = DB.clients.filter(function (c) { return ids.indexOf(c.id) > -1; });
    return FYBackup.buildBackup(FYBackup.fromLegacyClients(list), { source: 'on-screen copy (database not reachable)' });
  }
  async function saveBackup(ids) {
    try {
      const bk = await makeBackup(ids);
      download(new Blob([JSON.stringify(bk, null, 2)], { type: 'application/json' }), FYBackup.backupFileName(bk, todayISO()));
      const n = Object.keys(bk.counts).map(function (k) { return bk.counts[k]; }).reduce(function (a, b) { return a + b; }, 0);
      toast('Backup saved: ' + bk.client_ids.length + ' client(s), ' + n + ' rows', 'success', 4500);
      logEvent('backup_exported', { clients: bk.client_ids, counts: bk.counts });
    } catch (e) { toast('Backup failed: ' + e.message, 'error', 6000); }
  }
  window.fyBackupClient = function (id, ev) { if (ev) ev.stopPropagation(); saveBackup([id]); };
  window.fyBackupAll = function () { saveBackup(DB.clients.map(function (c) { return c.id; })); };
  window.fyChooseBackup = function () {
    showModal('Back up a client',
      '<p style="font-size:13px;color:var(--text2);margin-bottom:14px">Saves every row for the client (profile, weekly metrics, posts, snapshots, document records, credential labels) into one file. Keep it somewhere safe; if the database is ever wiped, upload it under <strong>Restore from backup</strong>.</p>' +
      '<div class="mrow"><label>Client</label><select id="bk-client">' + DB.clients.map(function (c) { return '<option value="' + esc(c.id) + '">' + esc(c.name) + '</option>'; }).join('') + '</select></div>',
      '<button class="btn btn-ghost" onclick="closeModal()">Cancel</button><button class="btn btn-primary" onclick="fyBackupClient(document.getElementById(\'bk-client\').value);closeModal()">&#8681; Save backup</button>');
  };
  window.fyOpenRestore = function () {
    showModal('Restore from backup',
      '<p style="font-size:13px;color:var(--text2);margin-bottom:14px">Upload a <code>fy-backup_...json</code> file. Each row is written back to the database; rows that already exist are updated, never duplicated. The file is checked first and rejected if it was edited or damaged.</p>' +
      '<div class="mrow"><input type="file" id="restore-file" accept="application/json,.json"/></div><div id="restore-status" style="font-size:12px;color:var(--text2);min-height:18px"></div>',
      '<button class="btn btn-ghost" onclick="closeModal()">Close</button><button class="btn btn-primary" onclick="fyRunRestore()">Restore</button>');
  };
  window.fyRunRestore = async function () {
    const f = el('restore-file') && el('restore-file').files[0];
    const st = el('restore-status');
    if (!f) { st.textContent = 'Choose a backup file first.'; return; }
    if (!connected()) { st.textContent = 'Not connected to the database. Reload the page and try again.'; st.style.color = 'var(--red)'; return; }
    st.textContent = 'Checking file...'; st.style.color = 'var(--text2)';
    try {
      const res = await FYBackup.restoreFromText(sb, await f.text(), SESSION.user.id, {
        onProgress: function (t, done, total) { st.textContent = 'Restoring ' + t + ': ' + done + ' / ' + total; } });
      const parts = Object.keys(res.written).map(function (k) { return res.written[k] + ' ' + k; });
      st.style.color = 'var(--green)';
      st.textContent = 'Restored ' + res.client_names.join(', ') + ' — ' + parts.join(', ');
      logEvent('backup_restored', { clients: res.client_ids, written: res.written });
      await DB.load(); renderClients(); await renderDashboard();
      toast('Backup restored', 'success');
    } catch (e) { st.style.color = 'var(--red)'; st.textContent = e.message; }
  };

  /* Re-render the lab whenever AI Insights renders; add Backup buttons to client cards. */
  function wrap(name, after) {
    const orig = window[name]; if (typeof orig !== 'function') return;
    window[name] = function () {
      const r = orig.apply(this, arguments);
      if (r && typeof r.then === 'function') return r.then(function (v) { after(); return v; });
      after(); return r;
    };
  }
  function addBackupButtons() {
    document.querySelectorAll('#clients-grid .client-card').forEach(function (card) {
      if (card.querySelector('.fy-bk')) return;
      const m = /switchClient\('([^']+)'\)/.exec(card.getAttribute('onclick') || '');
      if (!m) return;
      const b = document.createElement('button');
      b.className = 'btn btn-ghost fy-bk'; b.style.cssText = 'font-size:11px;padding:4px 10px;margin-top:10px';
      b.innerHTML = '&#8681; Backup'; b.title = 'Download a restorable backup of this client';
      b.onclick = function (ev) { window.fyBackupClient(m[1], ev); };
      card.appendChild(b);
    });
  }
  wrap('renderAI', LAB.render);
  wrap('renderClients', addBackupButtons);
  document.addEventListener('DOMContentLoaded', function () { setTimeout(function () { LAB.render(); addBackupButtons(); }, 0); });
})();
