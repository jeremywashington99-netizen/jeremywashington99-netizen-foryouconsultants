/* ==========================================================================
   SEGMENT 12 - Site wiring: every training run lives in Supabase

   NET (the Train Model network):
     FYSync.saveNet(client)     writes the run to public.model_runs
     FYSync.restoreNet(client)  rebuilds the trained network from the last run
     FYSync.fillForecasts(c)    caches 6-week forecasts for the scorecard
     FYSync.ensureNet(client)   restore if saved, otherwise train (and save)

   Nothing here touches localStorage. If the database is unreachable the
   screen says so; it never keeps a quiet local copy instead.
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
    Object.keys(n || {}).forEach(function (pk) { out[pk] = {}; Object.keys(n[pk]).forEach(function (k) { out[pk][k] = mm(n[pk][k]); }); });
    return out;
  }

  /* ---------- save ---------- */
  Sync.saveNet = async function (c) {
    if (!NET.trained || !NET.reg) return { ok: false, error: 'nothing trained to save' };
    const reg0 = NET.reg;
    try {
      const reg = await S.serializeModel(tf, NET.reg);
      const clf = NET.clf ? await S.serializeModel(tf, NET.clf) : null;
      const rng = {};
      (c.platforms || []).forEach(function (pk) { try { const n = NET.norms[pk].er; rng[pk] = (n.hi - n.lo) || 1; } catch (e) {} });
      const pk0 = c.primary || (c.platforms || [])[0];
      const miss = pk0 && rng[pk0] ? NET.metrics.mae * rng[pk0] : null;
      const summary = X.netSummary({ r2: NET.metrics.r2, rows: NET.metrics.rows });
      summary.missPoints = miss;
      summary.guardrails = NET.guard || null;
      const r = await S.saveRun(sb, ownerId(), {
        client_id: c.id, kind: 'net', status: 'ok', model_name: 'engagement_network',
        backend: NET.metrics.backend, epochs: NET.metrics.epochs,
        loss_final: NET.lossHistory.length ? NET.lossHistory[NET.lossHistory.length - 1] : null,
        mae: NET.metrics.mae, rmse: NET.metrics.rmse, r_squared: NET.metrics.r2, train_rows: NET.metrics.rows,
        data_weeks: weeksOf(c), data_posts: postsOf(c),
        architecture: { layers: NET.describe(), params: NET.metrics.params },
        hyperparams: { optimizer: 'adam', hidden: 'relu', output: 'linear (engagement) / softmax (content type)' },
        metrics: NET.metrics, forecast: NET.forecasts || null, summary: summary,
        activation_audit: (window.FYDL && FYDL.activations) ? { regressor: FYDL.activations.auditModel(NET.reg, 'regression') } : null,
        weights: { fv: NET.FEATURE_VERSION, guard: NET.guard || null, classCounts: NET.classCounts || null, reg: reg, clf: clf, norms: plainNorms(NET.norms), classScores: NET.classScores,
                   lossHistory: NET.lossHistory, lastFit: NET.lastFit, nClasses: CONTENT_CLASSES.length }
      });
      Sync.lastSave = r.ok ? new Date() : Sync.lastSave;
      Sync.lastError = r.ok ? null : r.error;
      return r;
    } catch (e) {
      /* new data arrived mid-save and the network was rebuilt: the newer run will be saved instead */
      if (NET.reg !== reg0) return { ok: false, superseded: true, error: 'new data arrived while saving; the newer training run is saved instead' };
      Sync.lastError = e.message; return { ok: false, error: e.message };
    }
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

  /* ---------- plain-English panel ---------- */
  Sync.netPanelHTML = function (c) {
    if (!NET.trained) return '';
    const sm = X.netSummary({ r2: NET.metrics.r2, rows: NET.metrics.rows });
    const stale = S.staleness(NET.savedRun, weeksOf(c), postsOf(c));
    const col = sm.fitPct >= 55 ? 'var(--green)' : sm.fitPct >= 30 ? 'var(--amber)' : 'var(--red)';
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
      '<div class="card-sub" style="margin:6px 0">' + saved + '</div>' +
      (NET.guard && window.FYGuard ? FYGuard.html(NET.guard, esc) : '') +
      '<details style="margin-top:8px"><summary class="card-sub" style="cursor:pointer">Technical detail (for analysts)</summary>' +
      '<div style="overflow-x:auto;margin-top:8px"><table class="data-table"><thead><tr><th>Stage</th><th>What it does</th><th>Rule it uses</th><th>Layer</th></tr></thead><tbody>' + tech + '</tbody></table></div></details>';
  };
})();
