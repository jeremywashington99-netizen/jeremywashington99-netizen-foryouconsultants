/* ==========================================================================
   SEGMENT 3 - RNN for sequential data (weekly metrics)

   Use in the site: read the last N weeks of a platform (engagement rate,
   followers, reach, posts, saves, paid boost) and predict next week.
     regression -> next week's engagement rate        (linear)
     binary     -> will engagement rise next week      (sigmoid)

   Architecture:
     LSTM 32  (cell activation ReLU, gates sigmoid)  -> hidden, ReLU
     Dense 16 -> ReLU / Leaky ReLU                    -> hidden
     Dropout
     Dense out (activation from the task)

   Why the gates stay sigmoid: inside an LSTM the input/forget/output gates
   are 0..1 switches that decide how much memory to keep. That is part of the
   cell's maths, not a hidden-layer activation, so the policy applies to the
   cell activation (ReLU) and every Dense hidden layer.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./dl-activations.js'));
  else { root.FYDL = root.FYDL || {}; root.FYDL.rnn = factory(root.FYDL.activations); }
})(typeof self !== 'undefined' ? self : this, function (act) {
  'use strict';

  const FEATURES = ['er', 'followers', 'reach', 'posts', 'saves', 'paid'];
  const TARGET = 'er';

  /** client.series[platform] -> rows of [er, followers, reach, posts, saves, paid] */
  function seriesToMatrix(series, cols) {
    const c = cols || FEATURES;
    const n = (series[TARGET] || []).length;
    const rows = [];
    for (let i = 0; i < n; i++) {
      rows.push(c.map(function (k) {
        const v = series[k] ? Number(series[k][i]) : 0;
        if (k === 'paid') return v > 0 ? 1 : 0;
        return isFinite(v) ? v : 0;
      }));
    }
    return rows;
  }

  /** Min-max scaler per column, so every input sits in 0..1. */
  function fitScaler(matrix) {
    const f = matrix[0].length;
    const lo = new Array(f).fill(Infinity), hi = new Array(f).fill(-Infinity);
    matrix.forEach(function (r) { r.forEach(function (v, j) { if (v < lo[j]) lo[j] = v; if (v > hi[j]) hi[j] = v; }); });
    return { lo: lo, hi: hi };
  }
  function scaleValue(v, scaler, j) { const d = scaler.hi[j] - scaler.lo[j]; return d === 0 ? 0.5 : (v - scaler.lo[j]) / d; }
  function unscaleValue(v, scaler, j) { const d = scaler.hi[j] - scaler.lo[j]; return d === 0 ? scaler.lo[j] : v * d + scaler.lo[j]; }
  function scaleMatrix(matrix, scaler) { return matrix.map(function (r) { return r.map(function (v, j) { return scaleValue(v, scaler, j); }); }); }

  /** Sliding windows: lookback weeks in, the following week's target out. */
  function makeWindows(scaled, lookback, targetCol, task) {
    const X = [], Y = [];
    for (let i = 0; i + lookback < scaled.length; i++) {
      X.push(scaled.slice(i, i + lookback));
      const next = scaled[i + lookback][targetCol];
      if (task === 'binary') Y.push(next > scaled[i + lookback - 1][targetCol] ? 1 : 0);
      else Y.push(next);
    }
    return { X: X, Y: Y };
  }

  /** Up to 4 weeks of history per window, never more than the data allows. */
  function chooseLookback(nWeeks) { return Math.max(1, Math.min(4, nWeeks - 2)); }

  const DEFAULTS = { task: 'regression', cell: 'lstm', units: 32, cellActivation: 'relu',
                     hidden: 'relu', denseUnits: 16, dropout: 0.1, learningRate: 0.005 };

  function buildRNN(tf, options) {
    const o = Object.assign({}, DEFAULTS, options || {});
    if (!o.timesteps || !o.features) throw new Error('timesteps and features are required');
    act.checkHidden(o.hidden);
    const out = act.outputSpec(o.task, 1);
    const m = tf.sequential({ name: 'rnn_' + o.task });
    const cellCfg = { units: o.units, activation: o.cellActivation, recurrentActivation: 'sigmoid',
                      inputShape: [o.timesteps, o.features], kernelInitializer: act.HIDDEN_INIT,
                      returnSequences: false, name: o.cell + '_' + o.units };
    m.add(o.cell === 'gru' ? tf.layers.gru(cellCfg) : tf.layers.lstm(cellCfg));
    act.addHidden(tf, m, function (a) {
      return tf.layers.dense({ units: o.denseUnits, activation: a, kernelInitializer: act.HIDDEN_INIT, name: 'dense_' + o.denseUnits });
    }, o.hidden);
    if (o.dropout > 0) m.add(tf.layers.dropout({ rate: o.dropout, name: 'dropout' }));
    m.add(tf.layers.dense({ units: 1, activation: out.activation, name: 'out_' + out.activation }));
    m.compile({ optimizer: tf.train.adam(o.learningRate), loss: out.loss, metrics: out.metrics });
    m.fyTask = o.task;
    return m;
  }

  /**
   * Train one model on every platform series of a client (pooled windows,
   * each platform scaled on its own range).
   * seriesByPlatform: { tiktok: client.series.tiktok, ... }
   */
  async function trainRNN(tf, seriesByPlatform, options) {
    const o = Object.assign({ epochs: 150, batchSize: 8, onEpoch: null }, DEFAULTS, options || {});
    const targetCol = FEATURES.indexOf(TARGET);
    const platforms = Object.keys(seriesByPlatform).filter(function (p) {
      const s = seriesByPlatform[p]; return s && s[TARGET] && s[TARGET].length >= 3;
    });
    if (!platforms.length) throw new Error('Need at least 3 weeks on one platform to train the RNN');
    const minWeeks = Math.min.apply(null, platforms.map(function (p) { return seriesByPlatform[p][TARGET].length; }));
    const lookback = o.lookback || chooseLookback(minWeeks);
    const scalers = {}, X = [], Y = [];
    platforms.forEach(function (p) {
      const mat = seriesToMatrix(seriesByPlatform[p]);
      const sc = fitScaler(mat); scalers[p] = sc;
      const w = makeWindows(scaleMatrix(mat, sc), lookback, targetCol, o.task);
      Array.prototype.push.apply(X, w.X); Array.prototype.push.apply(Y, w.Y);
    });
    if (X.length < 2) throw new Error('Need more weeks: only ' + X.length + ' training window(s)');
    const model = buildRNN(tf, Object.assign({}, o, { timesteps: lookback, features: FEATURES.length }));
    const xs = tf.tensor3d(X), ys = tf.tensor2d(Y, [Y.length, 1]);
    const history = [];
    try {
      await model.fit(xs, ys, { epochs: o.epochs, batchSize: o.batchSize, shuffle: true, verbose: 0,
        callbacks: { onEpochEnd: function (e, logs) { history.push(logs.loss); if (o.onEpoch) o.onEpoch(e, o.epochs, logs); } } });
      const pred = tf.tidy(function () { return Array.from(model.predict(xs).dataSync()); });
      let err = 0;
      pred.forEach(function (p, i) { err += Math.abs(p - Y[i]); });
      return { model: model, scalers: scalers, lookback: lookback, platforms: platforms, history: history,
               windows: X.length, task: o.task, maeScaled: err / pred.length };
    } finally { xs.dispose(); ys.dispose(); }
  }

  /** Roll the model forward `steps` weeks for one platform. Returns real-unit values. */
  function forecastRNN(tf, trained, series, platform, steps) {
    if (trained.task !== 'regression') throw new Error('forecast needs a regression model');
    const sc = trained.scalers[platform];
    if (!sc) throw new Error('model was not trained on ' + platform);
    const targetCol = FEATURES.indexOf(TARGET);
    const rows = scaleMatrix(seriesToMatrix(series), sc);
    const out = [];
    for (let k = 0; k < steps; k++) {
      const win = rows.slice(rows.length - trained.lookback);
      const v = tf.tidy(function () { return trained.model.predict(tf.tensor3d([win])).dataSync()[0]; });
      out.push(+unscaleValue(v, sc, targetCol).toFixed(2));
      const next = rows[rows.length - 1].slice(); next[targetCol] = v; rows.push(next);
    }
    return out;
  }

  return { FEATURES: FEATURES, TARGET: TARGET, DEFAULTS: DEFAULTS, seriesToMatrix: seriesToMatrix, fitScaler: fitScaler,
           scaleValue: scaleValue, unscaleValue: unscaleValue, scaleMatrix: scaleMatrix, makeWindows: makeWindows,
           chooseLookback: chooseLookback, buildRNN: buildRNN, trainRNN: trainRNN, forecastRNN: forecastRNN };
});
