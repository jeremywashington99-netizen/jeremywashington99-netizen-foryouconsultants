/* ==========================================================================
   SEGMENT 15 - The Train Model network, as one module that runs on the server

   This is the training code for the engagement network (regressor), the
   content-type classifier, the ten accuracy safeguards and the forecast roll.
   It used to live inside index.html and ran in the visitor's browser. It is
   now the single copy, used by:

     - the Supabase edge function "train-net"   (where training really happens)
     - the browser, for features() and forecast() on a model restored from the
       database (reading a saved model, never training one)
     - the Node tests

   tf, the guardrails module and the list of content classes are passed in, so
   nothing here touches a window, a document or any browser storage.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FYNetCore = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FEATURE_VERSION = 3;   /* bump when the inputs change: older saved models would read them wrongly */

  const Stat = {
    mean: a => a.reduce((x, y) => x + y, 0) / a.length,
    min: a => Math.min.apply(null, a),
    max: a => Math.max.apply(null, a),
    mae(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; },
    rmse(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); },
    minmax(a) { const lo = Stat.min(a), hi = Stat.max(a); return { lo: lo, hi: hi, norm: v => hi === lo ? 0.5 : (v - lo) / (hi - lo), denorm: v => v * (hi - lo) + lo }; }
  };

  function create(deps) {
    const tf = deps.tf, G = deps.G || null, CLASSES = deps.classes || [];
    if (!tf) throw new Error('net-core needs tf');

    /* Feature vector for week i of a platform series: 7 inputs, all scaled into 0..1. */
    function features(s, i, norms) {
      const prevER = i > 0 ? s.er[i - 1] : s.er[0];
      const prevER2 = i > 1 ? s.er[i - 2] : prevER;
      /* running maximum: only weeks up to i, so no future week leaks in (guardrail 5) */
      const postRate = s.posts[i] / Math.max(1, Math.max.apply(null, s.posts.slice(0, i + 1)));
      const paidShare = s.paid[i] > 0 ? 1 : 0;
      const saveRate = s.saves[i] / Math.max(1, s.reach[i]) * 100;
      return [
        norms.er.norm(s.er[i]),
        norms.er.norm(prevER),
        norms.er.norm(prevER2),
        norms.fol.norm(s.followers[i]),
        postRate,
        paidShare,
        Math.min(1, saveRate / 5)
      ];
    }

    function buildRegressor() {
      /* Small on purpose: a few dozen weekly examples cannot support a big network without it memorising them. */
      const reg = () => tf.regularizers.l2({ l2: 0.004 });
      const m = tf.sequential();
      m.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [7], kernelInitializer: 'heNormal', kernelRegularizer: reg(), useBias: true, name: 'dense_relu_24' }));
      m.add(tf.layers.dropout({ rate: 0.15, name: 'dropout_15' }));
      m.add(tf.layers.dense({ units: 12, activation: 'relu', kernelInitializer: 'heNormal', kernelRegularizer: reg(), useBias: true, name: 'dense_relu_12' }));
      m.add(tf.layers.dense({ units: 1, activation: 'linear', kernelInitializer: 'glorotNormal', useBias: true, name: 'linear_out' }));
      m.compile({ optimizer: tf.train.adam(0.008), loss: 'meanSquaredError', metrics: ['mse'] });
      return m;
    }

    function buildClassifier(nClasses) {
      const m = tf.sequential();
      m.add(tf.layers.dense({ units: 24, activation: 'relu', inputShape: [7], kernelInitializer: 'heNormal', useBias: true, name: 'clf_relu_24' }));
      m.add(tf.layers.dense({ units: 16, activation: 'relu', kernelInitializer: 'heNormal', useBias: true, name: 'clf_relu_16' }));
      m.add(tf.layers.dense({ units: nClasses, activation: 'softmax', kernelInitializer: 'glorotNormal', useBias: true, name: 'softmax_out' }));
      m.compile({ optimizer: tf.train.adam(0.02), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
      return m;
    }

    /**
     * Train on every platform series of a client.
     * client = { platforms:[], primary, series:{pk:{er,followers,reach,posts,paid,saves,dates}}, posts_log:[{platform,cls,posted_at}] }
     * Returns the trained state (models are tf objects: the caller serialises or disposes them).
     */
    async function train(client, opts) {
      opts = opts || {};
      const onEpoch = opts.onEpoch || null;
      try { await tf.ready(); } catch (e) {}
      const state = {
        reg: null, clf: null, norms: null, lossHistory: [], classScores: {}, classCounts: null, guard: null,
        recencyWeighted: false, lastFit: { actual: [], predicted: [], labels: [] },
        metrics: { mae: 0, rmse: 0, r2: 0, epochs: 0, params: 0, backend: (tf.getBackend && tf.getBackend()) || 'cpu', rows: 0 }
      };

      const X = [], Y = [], XC = [], YC = [], TAG = [];
      const platNorms = {};
      const dataIssues = { issues: [], bad: {}, count: 0 }; const leak = { ok: true, leaks: [] };

      client.platforms.forEach(pk => {
        const s = client.series[pk]; if (!s) return;
        const norms = { er: Stat.minmax(s.er), fol: Stat.minmax(s.followers), reach: Stat.minmax(s.reach) };
        platNorms[pk] = norms;
        /* guardrails 4 + 5: skip impossible/missing weeks, and prove week i never sees week i+1 */
        const vs = G ? G.validateSeries(s) : { issues: [], bad: {} };
        vs.issues.forEach(x => dataIssues.issues.push(Object.assign({ platform: pk }, x)));
        if (G) { const lp = G.leakProbe((q, i, n) => features(q, i, n), s, norms); if (!lp.ok) { leak.ok = false; leak.leaks = leak.leaks.concat(lp.leaks); } }
        /* supervised pairs: features(week i) -> engagement(week i+1) */
        for (let i = 0; i < s.er.length - 1; i++) {
          if (vs.bad[i] || vs.bad[i + 1]) continue;
          const f = features(s, i, norms), y = norms.er.norm(s.er[i + 1]);
          if (!f.every(Number.isFinite) || !Number.isFinite(y)) continue;
          X.push(f); Y.push([y]); TAG.push({ pk: pk, i: i });
        }
        /* classifier: logged posts give (features, content class) pairs */
        (client.posts_log || []).filter(p => p.platform === pk).forEach(p => {
          let wi = s.er.length - 1;
          if (p.posted_at && s.dates) {
            const pd = String(p.posted_at).slice(0, 10);
            for (let q = 0; q < s.dates.length; q++) { if (s.dates[q] >= pd) { wi = q; break; } }
          }
          if (!(p.cls >= 0 && p.cls < CLASSES.length)) return;
          const oh = new Array(CLASSES.length).fill(0);
          oh[p.cls] = 1;
          XC.push(features(s, wi, platNorms[pk])); YC.push(oh);
        });
      });

      if (X.length < 3) throw new Error('Not enough data to train');

      const reg = buildRegressor();
      const clf = buildClassifier(CLASSES.length);
      state.reg = reg; state.clf = clf;
      const tensors = [];
      const T = t => { tensors.push(t); return t; };

      try {
        /* guardrail 9, acted on: if recent weeks sit away from older ones, the newest weeks are counted twice while learning */
        let driftState = 'unknown';
        try { const pk0 = client.primary && client.series[client.primary] ? client.primary : client.platforms[0]; driftState = G ? G.drift(client.series[pk0]).state : 'unknown'; } catch (e) {}
        const emph = (driftState === 'shifted' || driftState === 'moving');
        const recentIdx = []; if (emph) { const per = {}; TAG.forEach((t, k) => { (per[t.pk] = per[t.pk] || []).push(k); }); Object.keys(per).forEach(pk => per[pk].slice(-6).forEach(k => recentIdx.push(k))); }
        state.recencyWeighted = emph;
        const xs = T(tf.tensor2d(X.concat(recentIdx.map(k => X[k])))), ys = T(tf.tensor2d(Y.concat(recentIdx.map(k => Y[k]))));
        const EPOCHS = X.length < 8 ? 300 : 220;
        /* guardrails 1 + 5: when there are enough examples, hold back the NEWEST weeks of each platform (never shuffled in) */
        let holdErr = null, trainErr = null, holdN = 0;
        let bestVal = Infinity, sinceBest = 0, bestW = null;
        const cb = (watchVal, record) => ({ onEpochEnd: (e, logs) => {
          if (record !== false) state.lossHistory.push(logs.loss);     /* the curve shown/checked is the main learning run only */
          if (watchVal && logs.val_loss != null) {                    /* early stopping, keeping the weights from the best held-back moment */
            if (logs.val_loss < bestVal - 1e-6) { bestVal = logs.val_loss; sinceBest = 0; if (bestW) bestW.forEach(w => w.dispose()); bestW = reg.getWeights().map(w => w.clone()); }
            else if (++sinceBest >= 50) reg.stopTraining = true;
          }
          if (onEpoch && e % 10 === 0) onEpoch(e, EPOCHS, logs.loss);
          if (!Number.isFinite(logs.loss)) reg.stopTraining = true;   /* guardrail 7: halt if numbers go invalid */
        } });
        const plainFit = () => reg.fit(xs, ys, { epochs: EPOCHS, batchSize: 8, shuffle: true, verbose: 0, callbacks: [cb()] });
        if (G && X.length >= G.MIN_FOR_HOLDOUT) {
          const trIdx = [], hoIdx = [];
          const byPk = {}; TAG.forEach((t, k) => { (byPk[t.pk] = byPk[t.pk] || []).push(k); });
          Object.keys(byPk).forEach(pk => {
            const ks = byPk[pk];
            if (ks.length < G.MIN_FOR_HOLDOUT / 2) { ks.forEach(k => trIdx.push(k)); return; }
            const sp = G.timeSplit(ks.length); sp.train.forEach(j => trIdx.push(ks[j])); sp.hold.forEach(j => hoIdx.push(ks[j]));
          });
          if (hoIdx.length >= 2 && trIdx.length >= 6) {
            const holdSet = new Set(hoIdx), extra = recentIdx.filter(k => !holdSet.has(k));
            const trW = trIdx.concat(extra);
            const xT = T(tf.tensor2d(trW.map(k => X[k]))), yT = T(tf.tensor2d(trW.map(k => Y[k])));
            const xH = T(tf.tensor2d(hoIdx.map(k => X[k]))), yH = T(tf.tensor2d(hoIdx.map(k => Y[k])));
            await reg.fit(xT, yT, { epochs: EPOCHS, batchSize: 8, shuffle: true, verbose: 0, validationData: [xH, yH], callbacks: [cb(true)] });
            if (bestW) { reg.setWeights(bestW); bestW.forEach(w => w.dispose()); bestW = null; }   /* roll back to the best held-back moment */
            const oT = T(reg.predict(xT)), oH = T(reg.predict(xH));
            const pT = Array.from(await oT.data()), pH = Array.from(await oH.data());
            trainErr = Stat.mae(trW.map(k => Y[k][0]), pT); holdErr = Stat.mae(hoIdx.map(k => Y[k][0]), pH); holdN = hoIdx.length;
            /* the honest numbers are measured; now let it also learn from the newest weeks */
            await reg.fit(xs, ys, { epochs: 10, batchSize: 8, shuffle: true, verbose: 0, callbacks: [cb(false, false)] });
          } else await plainFit();
        } else await plainFit();
        if (bestW) { bestW.forEach(w => w.dispose()); bestW = null; }
        if (state.lossHistory.some(v => !Number.isFinite(v))) throw new Error('Learning broke down (invalid numbers appeared). Check the data for extreme values and retry.');

        if (XC.length >= 3) {
          const xc = T(tf.tensor2d(XC)), yc = T(tf.tensor2d(YC));
          const counts = new Array(CLASSES.length).fill(0); YC.forEach(r => { counts[r.indexOf(1)]++; });
          state.classCounts = counts;   /* guardrail 3: rare content types get extra weight */
          await clf.fit(xc, yc, { epochs: 180, batchSize: 4, shuffle: true, verbose: 0, classWeight: G ? G.classWeights(counts) : undefined });
        }

        /* Evaluate the regressor against real held values */
        const xsEval = T(tf.tensor2d(X));
        const preds = T(reg.predict(xsEval));
        const pv = Array.from(await preds.data());
        const av = Y.map(v => v[0]);
        const m = state.metrics;
        m.mae = Stat.mae(av, pv);
        m.rmse = Stat.rmse(av, pv);
        const my = Stat.mean(av);
        let ssRes = 0, ssTot = 0;
        for (let i = 0; i < av.length; i++) { ssRes += (av[i] - pv[i]) ** 2; ssTot += (av[i] - my) ** 2; }
        m.r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
        m.epochs = state.lossHistory.length;
        m.rows = X.length;
        m.params = reg.countParams() + (clf ? clf.countParams() : 0);
        state.lastFit = { actual: av, predicted: pv, labels: av.map((_, i) => 's' + (i + 1)) };
        state.norms = platNorms;

        /* Softmax class scores per platform */
        client.platforms.forEach(pk => {
          const s = client.series[pk]; if (!s) return;
          const t = tf.tensor2d([features(s, s.er.length - 1, platNorms[pk])]);
          const out = clf.predict(t);
          state.classScores[pk] = Array.from(out.dataSync());
          t.dispose(); out.dispose();
        });

        /* the ten accuracy safeguards, kept with this run */
        try {
          if (G) {
            const primary = client.primary && client.series[client.primary] ? client.primary : client.platforms[0];
            const health = G.weightsHealth(reg.getWeights().map(w => w.dataSync()));
            state.guard = G.audit({ history: state.lossHistory, trainErr: trainErr, holdErr: holdErr, holdN: holdN, r2: m.r2, rows: X.length,
              classCounts: XC.length >= 3 ? state.classCounts : null, dataIssues: dataIssues, leak: leak, health: health,
              drift: G.drift(client.series[primary]), adapted: state.recencyWeighted,
              holdPoints: (holdErr != null && platNorms[primary] && platNorms[primary].er) ? holdErr * ((platNorms[primary].er.hi - platNorms[primary].er.lo) || 1) : null,
              scale: G.scaling(X), weeks: client.series[primary].er.length });
          }
        } catch (e) { state.guard = null; state.guardError = e.message; }
      } catch (e) {
        release(state);
        throw e;
      } finally {
        tensors.forEach(t => { try { t.dispose(); } catch (_) {} });
      }
      return state;
    }

    /* Free a trained state. A model's optimiser keeps its own tensors, so it is released as well. */
    function release(state) {
      if (!state) return;
      ['reg', 'clf'].forEach(k => {
        const m = state[k]; if (!m) return;
        try { if (m.optimizer && m.optimizer.dispose) m.optimizer.dispose(); } catch (_) {}
        try { m.dispose(); } catch (_) {}
        state[k] = null;
      });
    }

    /* Roll the regressor forward n weeks for one platform */
    async function forecast(state, client, platformKey, nWeeks) {
      const s = client.series[platformKey];
      if (!s || !state || !state.reg || !state.norms || !state.norms[platformKey]) return [];
      const norms = state.norms[platformKey];
      const er = s.er.slice();
      const out = [];
      for (let k = 0; k < nWeeks; k++) {
        const i = er.length - 1;
        const rep = (arr, v) => arr.concat(new Array(k).fill(v));
        const f = features({
          er: er, followers: rep(s.followers, s.followers[s.followers.length - 1]),
          reach: rep(s.reach, s.reach[s.reach.length - 1]),
          posts: rep(s.posts, s.posts[s.posts.length - 1]),
          paid: rep(s.paid, 0),
          saves: rep(s.saves, s.saves[s.saves.length - 1])
        }, i, norms);   /* step k looks at the week it just predicted, not always the last real one */
        const t = tf.tensor2d([f]);
        const p = state.reg.predict(t);
        const raw = (await p.data())[0];
        t.dispose(); p.dispose();
        const val = +Math.max(0, norms.er.denorm(raw)).toFixed(2);
        out.push(val); er.push(val);
      }
      return out;
    }

    /* Layer table for the architecture panel */
    function describe(state) {
      const rows = [];
      const push = (model, tag) => {
        if (!model) return;
        model.layers.forEach(l => {
          const w = l.getWeights();
          rows.push({
            head: tag, name: l.name,
            activation: (l.getConfig && l.getConfig().activation) || (l.name.indexOf('dropout') > -1 ? 'dropout' : 'linear'),
            units: (l.getConfig && l.getConfig().units) || '-',
            weights: w.length ? w[0].shape.join(' x ') : '-', bias: w.length > 1 ? w[1].shape.join(' x ') : '-', params: l.countParams()
          });
        });
      };
      push(state.reg, 'regressor');
      push(state.clf, 'classifier');
      return rows;
    }

    function plainNorms(n) {
      const out = {};
      Object.keys(n || {}).forEach(pk => { out[pk] = {}; Object.keys(n[pk]).forEach(k => { out[pk][k] = { lo: n[pk][k].lo, hi: n[pk][k].hi }; }); });
      return out;
    }

    return { FEATURE_VERSION: FEATURE_VERSION, features: features, buildRegressor: buildRegressor, buildClassifier: buildClassifier,
             train: train, forecast: forecast, describe: describe, plainNorms: plainNorms, release: release };
  }

  /* Reject a payload that is not a sane client before spending any compute on it. */
  function validateClient(c, limits) {
    const L = Object.assign({ platforms: 8, weeks: 260, posts: 3000 }, limits || {});
    if (!c || typeof c !== 'object') return 'no client data';
    if (!Array.isArray(c.platforms) || !c.platforms.length || c.platforms.length > L.platforms) return 'platforms must be a list of 1-' + L.platforms;
    if (!c.series || typeof c.series !== 'object') return 'no weekly series';
    const keys = ['er', 'followers', 'reach', 'posts', 'paid', 'saves'];
    for (const pk of c.platforms) {
      if (typeof pk !== 'string' || pk.length > 40) return 'bad platform name';
      const s = c.series[pk]; if (!s) return 'missing series for ' + pk;
      const n = Array.isArray(s.er) ? s.er.length : 0;
      if (n < 1 || n > L.weeks) return pk + ': weeks must be 1-' + L.weeks;
      for (const k of keys) {
        if (!Array.isArray(s[k]) || s[k].length !== n) return pk + ': "' + k + '" must have ' + n + ' values';
        if (!s[k].every(v => typeof v === 'number' && Number.isFinite(v))) return pk + ': "' + k + '" has a value that is not a number';
      }
      if (s.dates != null && (!Array.isArray(s.dates) || s.dates.length !== n)) return pk + ': dates must match the weeks';
    }
    if (c.posts_log != null) {
      if (!Array.isArray(c.posts_log) || c.posts_log.length > L.posts) return 'too many posts';
    }
    if (c.primary != null && (typeof c.primary !== 'string' || !c.series[c.primary])) return 'primary platform not found';
    return null;
  }

  return { FEATURE_VERSION: FEATURE_VERSION, Stat: Stat, create: create, validateClient: validateClient };
});
