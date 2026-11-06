/* ==========================================================================
   SEGMENT 15 - The Train Model network, as one module that runs on the server

   This is the training code for the engagement network (regressor), the
   content-type classifier, the ten accuracy safeguards and the forecast roll.
   It used to live inside index.html and ran in the visitor's browser. It is
   now the single copy, used by:

     - the Supabase edge function "train-net"   (where training really happens)
     - the Node tests

   The networks are trained by js/mini-mlp.js (plain JavaScript, no library),
   which is what lets this finish well inside an edge function's compute
   allowance. The guardrails module, the content classes and the MLP trainer
   are passed in, so nothing here touches a window, a document or any browser
   storage.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FYNetCoreV5 = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FEATURE_VERSION = 5;   /* bump when the inputs change: older saved models would read them wrongly.
                                 v4 = FROZEN fixed transforms (fy_features_v1): nothing is fitted to data, ever.
                                 v5 = same frozen transforms, but the network predicts the CHANGE from this week
                                      (next = this week + change) and sees the weekly rhythm (recent moves and
                                      distance from the 6-week average). Fixed gain constants only, still never refit. */
  const GAIN = 10;       /* fixed: one ER point near 4% is ~0.04 scaled units; x10 puts weekly moves on a 0-to-1 footing */
  const SHORT = 4, LONG = 8;   /* fixed: weeks in the short and long recent averages (this week included) */

  /* ---- Frozen, deterministic transforms (identical to public.fy_scale in the database and FY.features in the page).
     Fixed constants only: the same raw value always maps to the same feature value, on every device, forever.
     The signed log compresses both tails, so a sudden audience gain or loss cannot dominate. ---- */
  const LN = Math.log;
  /* regressor settings (fixed): learning rate, weight penalty, dropout, epochs, early-stop patience, members, member size */
  let REG_LR = 0.01, REG_L2 = 0, REG_DROP = 0, REG_EPOCHS = 300, REG_PATIENCE = 80, ENSEMBLE = 5;
  /* No weight penalty or dropout per member: early stopping on the held-back newest weeks plus averaging five
     members does the regularising. Measured on the demo client over 25 runs: weeks studied 77-80% matched,
     held-back weeks beat "same as last week". */
  const H1 = 24, H2 = 12;
  const round6 = v => { const s = v < 0 ? -1 : 1; return s * Math.round(Math.abs(v) * 1e6 + 1e-9) / 1e6; };
  const slog = x => Math.sign(x) * LN(1 + Math.abs(x));
  const pos = x => Math.max(x, 0);
  const SCALE = {
    engagement_rate:   x => LN(1 + pos(x)) / LN(101),
    followers:         x => LN(1 + pos(x)) / LN(1 + 1e9),
    views:             x => LN(1 + pos(x)) / LN(1 + 1e10),
    count_small:       x => LN(1 + pos(x)) / LN(101),
    rate:              x => LN(1 + 100 * pos(x)) / LN(101),
    follower_delta:    x => slog(x) / slog(1e7),
    pct_change:        x => slog(x) / slog(1000)
  };
  function scale(kind, x) { if (x === null || x === undefined || !Number.isFinite(+x)) return 0; return round6(SCALE[kind](+x)); }
  const FROZEN = {
    er:    { frozen: true, kind: 'engagement_rate', lo: 0, hi: 100,  norm: v => scale('engagement_rate', v), denorm: y => Math.exp(y * LN(101)) - 1 },
    fol:   { frozen: true, kind: 'followers',       lo: 0, hi: 1e9,  norm: v => scale('followers', v),       denorm: y => Math.exp(y * LN(1 + 1e9)) - 1 },
    reach: { frozen: true, kind: 'views',           lo: 0, hi: 1e10, norm: v => scale('views', v),           denorm: y => Math.exp(y * LN(1 + 1e10)) - 1 }
  };
  /* ER points that one unit of scaled error corresponds to, at a given engagement level */
  function missPoints(maeScaled, meanER) {
    const m = Number.isFinite(meanER) ? meanER : 5;
    return Math.abs(FROZEN.er.denorm(FROZEN.er.norm(m) + maeScaled) - m);
  }

  const Stat = {
    mean: a => a.reduce((x, y) => x + y, 0) / a.length,
    min: a => Math.min.apply(null, a),
    max: a => Math.max.apply(null, a),
    mae(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / a.length; },
    rmse(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return Math.sqrt(s / a.length); },
    minmax(a) { const lo = Stat.min(a), hi = Stat.max(a); return { lo: lo, hi: hi, norm: v => hi === lo ? 0.5 : (v - lo) / (hi - lo), denorm: v => v * (hi - lo) + lo }; }
  };

  /* Feature vector for week i of a platform series: 9 inputs from FIXED transforms.
     Each input depends only on week i and the weeks before it, so no future week can leak in
     (guardrail 5), and adding new weeks never changes the inputs of an old week. The norms argument
     is kept for the callers' signature; the frozen scalers are always used.
       0  this week's engagement (frozen log scale)
       1  last move: this week minus last week          (x GAIN, squashed by tanh into -0.5..0.5)
       2  the move before that                          (same)
       3  distance from the 4-week average              (same)  -> together with 4, the weekly rhythm:
       4  distance from the 8-week average              (same)     rises that fall back, dips that recover
       5  follower growth this week, %                  (frozen pct_change scale)
       6  posts this week                               (frozen count scale)
       7  paid promotion this week                      (0 / 1)
       8  saves per person reached                      (frozen rate scale) */
  const squash = v => round6(0.5 * Math.tanh(GAIN * v));
  function erAt(s, j) { return scale('engagement_rate', s.er[Math.max(0, j)]); }
  function features(s, i, norms) {
    const e0 = erAt(s, i), e1 = erAt(s, i - 1), e2 = erAt(s, i - 2);
    const avg = w => { let sum = 0, n = 0; for (let j = Math.max(0, i - w + 1); j <= i; j++) { sum += erAt(s, j); n++; } return sum / n; };
    const f0 = Number(s.followers[i]) || 0, fp = i > 0 ? (Number(s.followers[i - 1]) || 0) : f0;
    const growth = fp > 0 ? 100 * (f0 - fp) / fp : 0;
    const reach = (s.reach || [])[i] || 0, saves = (s.saves || [])[i] || 0;
    return [
      e0,
      squash(e0 - e1),
      squash(e1 - e2),
      squash(e0 - avg(SHORT)),
      squash(e0 - avg(LONG)),
      Math.max(-0.5, Math.min(1.5, scale('pct_change', growth))),   /* fixed clamp: a freak week cannot swamp the rest */
      scale('count_small', (s.posts || [])[i] || 0),
      (s.paid || [])[i] > 0 ? 1 : 0,
      reach > 0 ? scale('rate', saves / reach) : 0
    ];
  }
  const N_INPUTS = 9;
  /* The network's output is the change from this week, in GAIN units. Turn it back into next week's level
     (frozen scaled units): level = this week + change / GAIN. */
  function decode(raw, f) { return f[0] + raw / GAIN; }
  function target(s, i) { return GAIN * (erAt(s, i + 1) - erAt(s, i)); }


  function create(deps) {
    const Mlp = deps.Mlp, G = deps.G || null, CLASSES = deps.classes || [];
    const rand = deps.rand || undefined;

    function needMlp() { if (!Mlp) throw new Error('net-core needs the mini-mlp trainer to train'); }
    function regSpec(k) {
      /* Each member is small on purpose: a few dozen weekly examples cannot support a big network without it memorising them.
         Early stopping keeps each one learning patterns; the penalty and dropout settings stay available. Hidden layers stay ReLU. */
      return { name: 'sequential_1', inputDim: N_INPUTS, loss: 'mse', lr: REG_LR,
        layers: [{ name: 'dense_relu_' + (H1 * k), units: H1 * k, act: 'relu', init: 'he', l2: REG_L2 },
                 { name: 'dense_relu_' + (H2 * k), units: H2 * k, act: 'relu', init: 'he', l2: REG_L2 },
                 { name: 'linear_out', units: 1, act: 'linear', init: 'glorot' }],
        dropout: REG_DROP > 0 ? { after: 0, rate: REG_DROP, name: 'dropout_' + Math.round(REG_DROP * 100) } : undefined };
    }
    function buildRegressor() { needMlp(); return Mlp.create(regSpec(1), rand); }

    /* k members -> one network with k times the hidden units: layer 1 side by side, layer 2 block-diagonal
       (member j's units only see member j's layer-1 units), output weights divided by k. Exactly the average. */
    function mergeEnsemble(members) {
      const k = members.length;
      if (k === 1) return members[0];
      const big = Mlp.create(regSpec(k), rand);
      const ws = members.map(m => m.getWeights());
      const K0 = new Float64Array(N_INPUTS * H1 * k), B0 = new Float64Array(H1 * k);
      const K1 = new Float64Array(H1 * k * H2 * k), B1 = new Float64Array(H2 * k);
      const K2 = new Float64Array(H2 * k), B2 = new Float64Array(1);
      ws.forEach((w, m) => {
        for (let i = 0; i < N_INPUTS; i++) for (let j = 0; j < H1; j++) K0[i * H1 * k + m * H1 + j] = w[0][i * H1 + j];
        for (let j = 0; j < H1; j++) B0[m * H1 + j] = w[1][j];
        for (let i = 0; i < H1; i++) for (let j = 0; j < H2; j++) K1[(m * H1 + i) * H2 * k + m * H2 + j] = w[2][i * H2 + j];
        for (let j = 0; j < H2; j++) B1[m * H2 + j] = w[3][j];
        for (let i = 0; i < H2; i++) K2[m * H2 + i] = w[4][i] / k;
        B2[0] += w[5][0] / k;
      });
      big.setWeights([K0, B0, K1, B1, K2, B2]);
      return big;
    }

    function buildClassifier(nClasses) {
      needMlp();
      return Mlp.create({ name: 'sequential_2', inputDim: N_INPUTS, loss: 'cce', lr: 0.02,
        layers: [{ name: 'clf_relu_24', units: 24, act: 'relu', init: 'he' },
                 { name: 'clf_relu_16', units: 16, act: 'relu', init: 'he' },
                 { name: 'softmax_out', units: nClasses, act: 'softmax', init: 'glorot' }] }, rand);
    }

    /**
     * Train on every platform series of a client.
     * client = { platforms:[], primary, series:{pk:{er,followers,reach,posts,paid,saves,dates}}, posts_log:[{platform,cls,posted_at}] }
     * Returns the trained state (models are tf objects: the caller serialises or disposes them).
     */
    async function train(client, opts) {
      opts = opts || {};
      const onEpoch = opts.onEpoch || null;
      /* parts lets a server with a short time limit train the two networks in separate requests; they are independent */
      const parts = opts.parts || ['reg', 'clf'];
      const doReg = parts.indexOf('reg') >= 0, doClf = parts.indexOf('clf') >= 0;
      const state = {
        reg: null, clf: null, norms: null, lossHistory: [], classScores: {}, classCounts: null, guard: null,
        recencyWeighted: false, lastFit: { actual: [], predicted: [], labels: [] },
        metrics: { mae: 0, rmse: 0, r2: 0, epochs: 0, params: 0, backend: 'mini-mlp (plain JavaScript)', rows: 0 }
      };

      const X = [], Y = [], XC = [], YC = [], TAG = [];
      const platNorms = {};
      const dataIssues = { issues: [], bad: {}, count: 0 }; const leak = { ok: true, leaks: [] };

      client.platforms.forEach(pk => {
        const s = client.series[pk]; if (!s) return;
        const norms = FROZEN;   /* fixed scaler: never fitted, never refit */
        platNorms[pk] = norms;
        /* guardrails 4 + 5: skip impossible/missing weeks, and prove week i never sees week i+1 */
        const vs = G ? G.validateSeries(s) : { issues: [], bad: {} };
        vs.issues.forEach(x => dataIssues.issues.push(Object.assign({ platform: pk }, x)));
        if (G) { const lp = G.leakProbe((q, i, n) => features(q, i, n), s, norms); if (!lp.ok) { leak.ok = false; leak.leaks = leak.leaks.concat(lp.leaks); } }
        /* supervised pairs: features(week i) -> engagement(week i+1) */
        for (let i = 0; i < s.er.length - 1; i++) {
          if (vs.bad[i] || vs.bad[i + 1]) continue;
          const f = features(s, i, norms), y = target(s, i);
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

      const clf = doClf ? buildClassifier(CLASSES.length) : null;
      state.clf = clf;
      let reg = null;

      try {
        let holdErr = null, trainErr = null, holdN = 0;
        state.norms = platNorms;
        if (XC.length >= 3) { const counts = new Array(CLASSES.length).fill(0); YC.forEach(r => { counts[r.indexOf(1)]++; }); state.classCounts = counts; }   /* guardrail 3: rare content types get extra weight */
        if (doReg) {
        /* guardrail 9, acted on: if recent weeks sit away from older ones, the newest weeks are counted twice while learning */
        let driftState = 'unknown';
        try { const pk0 = client.primary && client.series[client.primary] ? client.primary : client.platforms[0]; driftState = G ? G.drift(client.series[pk0]).state : 'unknown'; } catch (e) {}
        const emph = (driftState === 'shifted' || driftState === 'moving');
        const recentIdx = []; if (emph) { const per = {}; TAG.forEach((t, k) => { (per[t.pk] = per[t.pk] || []).push(k); }); Object.keys(per).forEach(pk => per[pk].slice(-6).forEach(k => recentIdx.push(k))); }
        state.recencyWeighted = emph;
        const xs = X.concat(recentIdx.map(k => X[k])), ys = Y.concat(recentIdx.map(k => Y[k]));
        const EPOCHS = X.length < 8 ? 300 : REG_EPOCHS;

        /* guardrails 1 + 5: when there are enough examples, hold back the NEWEST weeks of each platform (never shuffled in) */
        let trIdx = null, hoIdx = null;
        if (G && X.length >= G.MIN_FOR_HOLDOUT) {
          const tr = [], ho = [];
          const byPk = {}; TAG.forEach((t, k) => { (byPk[t.pk] = byPk[t.pk] || []).push(k); });
          Object.keys(byPk).forEach(pk => {
            const ks = byPk[pk];
            if (ks.length < G.MIN_FOR_HOLDOUT / 2) { ks.forEach(k => tr.push(k)); return; }
            const sp = G.timeSplit(ks.length); sp.train.forEach(j => tr.push(ks[j])); sp.hold.forEach(j => ho.push(ks[j]));
          });
          if (ho.length >= 2 && tr.length >= 6) { trIdx = tr; hoIdx = ho; }
        }
        const trW = trIdx ? trIdx.concat(recentIdx.filter(k => hoIdx.indexOf(k) < 0)) : null;

        /* An ensemble: ENSEMBLE small networks learn independently from different random starting points and the
           forecast is their average. One network trained on a few dozen weeks lands somewhere different every time;
           the average is steady from run to run and misses less. The members are merged into ONE network
           (side-by-side hidden units, output averaged), which gives exactly the same numbers as averaging them. */
        const members = [], memberHold = [], memberTrain = [];
        for (let m = 0; m < ENSEMBLE; m++) {
          const net = buildRegressor();
          let bestVal = Infinity, sinceBest = 0, bestW = null;
          const cb = (watchVal, record) => ((e, logs) => {
            if (record !== false && m === 0) state.lossHistory.push(logs.loss);   /* the curve shown is the first member's main learning run */
            if (watchVal && logs.val_loss != null) {                             /* early stopping, keeping the weights from the best held-back moment */
              if (logs.val_loss < bestVal - 1e-6) { bestVal = logs.val_loss; sinceBest = 0; bestW = net.getWeights(); }
              else if (++sinceBest >= REG_PATIENCE) net.stopTraining = true;
            }
            if (onEpoch && e % 10 === 0) onEpoch(m * EPOCHS + e, ENSEMBLE * EPOCHS, logs.loss);
            if (!Number.isFinite(logs.loss)) { net.stopTraining = true; state.lossHistory.push(logs.loss); }   /* guardrail 7: halt if numbers go invalid */
          });
          if (trIdx) {
            const xT = trW.map(k => X[k]), yT = trW.map(k => Y[k]);
            const xH = hoIdx.map(k => X[k]), yH = hoIdx.map(k => Y[k]);
            net.fit(xT, yT, { epochs: EPOCHS, batchSize: 8, shuffle: true, validation: { X: xH, Y: yH }, onEpochEnd: cb(true) });
            if (bestW) net.setWeights(bestW);   /* roll back to the best held-back moment */
            memberTrain.push(net.predict(xT).map((r, j) => decode(r[0], xT[j])));
            memberHold.push(net.predict(xH).map((r, j) => decode(r[0], xH[j])));
            /* the honest numbers are measured; now let it also learn from the newest weeks */
            net.fit(xs, ys, { epochs: 10, batchSize: 8, shuffle: true, onEpochEnd: cb(false, false) });
          } else net.fit(xs, ys, { epochs: EPOCHS, batchSize: 8, shuffle: true, onEpochEnd: cb() });
          members.push(net);
        }
        if (state.lossHistory.some(v => !Number.isFinite(v))) throw new Error('Learning broke down (invalid numbers appeared). Check the data for extreme values and retry.');
        if (trIdx) {
          const avgOf = rows => rows[0].map((_, j) => rows.reduce((t, r) => t + r[j], 0) / rows.length);
          const lvl = ks => ks.map(k => decode(Y[k][0], X[k]));                 /* real next-week level, frozen units */
          const pT = avgOf(memberTrain), pH = avgOf(memberHold);
          trainErr = Stat.mae(lvl(trW), pT); holdErr = Stat.mae(lvl(hoIdx), pH); holdN = hoIdx.length;
          state.holdLevels = { actual: lvl(hoIdx), predicted: pH, persist: hoIdx.map(k => X[k][0]) };
        }
        state.memberWeights = [].concat.apply([], members.map(m => m.getWeights()));
        reg = mergeEnsemble(members);
        state.reg = reg;
        state.ensemble = ENSEMBLE;
        state.holdout = { holdErr: holdErr, trainErr: trainErr, holdN: holdN };
        }

        if (doClf && XC.length >= 3) {
          clf.fit(XC, YC, { epochs: 180, batchSize: 4, shuffle: true, classWeights: G ? G.classWeights(state.classCounts) : undefined });
        }

        if (doReg) {
        /* Evaluate the regressor against real held values */
        const pv = reg.predict(X).map((r, j) => decode(r[0], X[j]));     /* next week's level, frozen units */
        const av = Y.map((v, j) => decode(v[0], X[j]));
        const m = state.metrics;
        m.mae = Stat.mae(av, pv);
        m.rmse = Stat.rmse(av, pv);
        const r2Of = (a, p) => { const my = Stat.mean(a); let ssRes = 0, ssTot = 0;
          for (let i = 0; i < a.length; i++) { ssRes += (a[i] - p[i]) ** 2; ssTot += (a[i] - my) ** 2; }
          return ssTot === 0 ? 0 : 1 - ssRes / ssTot; };
        m.r2_pooled = r2Of(av, pv);
        /* Scored PER PLATFORM: each platform's ups and downs measured against its own average, so a platform that simply
           runs higher than another earns nothing. The headline score is the LOWEST platform, the conservative one. */
        const toER = v => +Math.max(0, FROZEN.er.denorm(v)).toFixed(2);
        m.byPlatform = {}; const fitBy = {};
        client.platforms.forEach(pk => {
          const ks = []; TAG.forEach((t, k) => { if (t.pk === pk) ks.push(k); });
          if (!ks.length) return;
          const a = ks.map(k => av[k]), p = ks.map(k => pv[k]), s = client.series[pk];
          m.byPlatform[pk] = { r2: ks.length >= 4 ? r2Of(a, p) : null, rows: ks.length, mae: Stat.mae(a, p) };
          fitBy[pk] = { actual: a.map(toER), predicted: p.map(toER),
                        labels: ks.map(k => { const i = TAG[k].i + 1; return s.dates && s.dates[i] ? String(s.dates[i]).slice(5, 10) : 'wk ' + (i + 1); }) };
        });
        const scored = Object.keys(m.byPlatform).map(pk => m.byPlatform[pk].r2).filter(v => v != null);
        m.r2 = scored.length ? Math.min.apply(null, scored) : m.r2_pooled;
        m.epochs = state.lossHistory.length;
        m.rows = X.length;
        m.params = (reg ? reg.countParams() : 0) + (clf ? clf.countParams() : 0);
        state.lastFit = { actual: av.map(toER), predicted: pv.map(toER), labels: av.map((_, i) => 's' + (i + 1)), platforms: fitBy };
        }

        /* Softmax class scores per platform */
        if (doClf) client.platforms.forEach(pk => {
          const s = client.series[pk]; if (!s) return;
          state.classScores[pk] = clf.predict([features(s, s.er.length - 1, platNorms[pk])])[0];
        });

        /* the ten accuracy safeguards, kept with this run */
        try {
          if (G && doReg) {
            const primary = client.primary && client.series[client.primary] ? client.primary : client.platforms[0];
            const health = G.weightsHealth(state.memberWeights || reg.getWeights());   /* the members, not the merged copy's structural zeros */
            state.guard = G.audit({ history: state.lossHistory, trainErr: trainErr, holdErr: holdErr, holdN: holdN, r2: state.metrics.r2, rows: X.length,
              classCounts: XC.length >= 3 ? state.classCounts : null, dataIssues: dataIssues, leak: leak, health: health,
              drift: G.drift(client.series[primary]), adapted: state.recencyWeighted,
              holdPoints: holdErr != null ? missPoints(holdErr, Stat.mean(client.series[primary].er)) : null,
              scale: G.scaling(X), weeks: client.series[primary].er.length });
          }
        } catch (e) { state.guard = null; state.guardError = e.message; }
      } catch (e) {
        release(state);
        throw e;
      }
      return state;
    }

    /* Nothing to free: the networks are plain arrays. Kept so callers need not care how training is done. */
    function release(state) { if (state) { state.reg = null; state.clf = null; } }

    /* Roll the regressor forward n weeks for one platform */
    async function forecast_(state, client, platformKey, nWeeks) {
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
        const raw = state.reg.predict([f])[0][0];
        const val = +Math.max(0, norms.er.denorm(decode(raw, f))).toFixed(2);
        out.push(val); er.push(val);
      }
      return out;
    }

    /* Layer table for the architecture panel */
    function describe(state) {
      return (state.reg ? state.reg.describe('regressor') : []).concat(state.clf ? state.clf.describe('classifier') : []);
    }

    function plainNorms(n) {
      const out = {};
      Object.keys(n || {}).forEach(pk => { out[pk] = {}; Object.keys(n[pk]).forEach(k => {
        out[pk][k] = n[pk][k].frozen ? { frozen: true, kind: n[pk][k].kind, lo: n[pk][k].lo, hi: n[pk][k].hi } : { lo: n[pk][k].lo, hi: n[pk][k].hi }; }); });
      return out;
    }

    /* The model_runs row for a finished training run: what the server saves, built in one place so tests see the same thing. */
    async function runRow(st, c, ctx) {
      const MAX_B64 = 900 * 1024;   /* one database row must stay readable on every page load */
      const reg = st.reg.toBlob(), clf = st.clf ? st.clf.toBlob() : null;
      if (reg.weightData.length > MAX_B64 || (clf && clf.weightData.length > MAX_B64)) throw new Error('model weights are too large for one database row');
      const forecast = {};
      for (const pk of c.platforms) { const f = await forecast_(st, c, pk, 6); if (f.length) forecast[pk] = f; }
      const m = st.metrics, pk0 = c.primary || c.platforms[0];
      const s0 = c.series[pk0], meanER = s0 && s0.er.length ? Stat.mean(s0.er) : null;
      /* the plain-English wording is built in the page from these numbers */
      const summary = { trained_on: 'supabase', missPoints: missPoints(m.mae, meanER), feature_spec: 'fy_features_v1', guardrails: st.guard || null };
      const weeks = Math.max.apply(null, c.platforms.map(pk => c.series[pk].er.length));
      return {
        owner_id: ctx.owner_id, client_id: ctx.client_id, kind: 'net', status: 'ok', model_name: 'engagement_network', framework: 'fy-mlp (plain JavaScript)',
        backend: 'supabase edge function', architecture: { layers: describe(st), params: m.params },
        hyperparams: { optimizer: 'adam', hidden: 'relu', output: 'linear (engagement) / softmax (content type)', trained_on: 'supabase' },
        epochs: m.epochs, loss_final: st.lossHistory.length ? st.lossHistory[st.lossHistory.length - 1] : null,
        mae: m.mae, rmse: m.rmse, r_squared: m.r2, train_rows: m.rows, data_weeks: weeks, data_posts: (c.posts_log || []).length,
        metrics: m, forecast: forecast, summary: summary, activation_audit: null,
        weights: { fv: FEATURE_VERSION, guard: st.guard || null, classCounts: st.classCounts || null, reg: reg, clf: clf,
                   norms: plainNorms(st.norms), classScores: st.classScores, lossHistory: st.lossHistory, lastFit: st.lastFit, nClasses: CLASSES.length }
      };
    }

    return { FEATURE_VERSION: FEATURE_VERSION, features: features, decode: decode, runRow: runRow, buildRegressor: buildRegressor, mergeEnsemble: mergeEnsemble, buildClassifier: buildClassifier,
             train: train, forecast: forecast_, describe: describe, plainNorms: plainNorms, release: release };
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

  return { FEATURE_VERSION: FEATURE_VERSION, Stat: Stat, features: features, decode: decode, target: target, GAIN: GAIN, N_INPUTS: N_INPUTS, create: create, validateClient: validateClient,
           FROZEN: FROZEN, scale: scale, slog: slog, missPoints: missPoints };
});
