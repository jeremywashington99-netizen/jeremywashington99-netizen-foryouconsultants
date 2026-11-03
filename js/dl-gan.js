/* ==========================================================================
   SEGMENT 4 - GAN for data generation

   Use in the site: learn what a client's real weeks look like and generate
   realistic synthetic weeks for what-if planning and for padding small
   training sets. Synthetic rows are clearly labelled and are never written
   into the real metrics tables.

   Generator      z(8) -> Dense 32 LeakyReLU -> Dense 32 LeakyReLU
                  -> Dense n_features  SIGMOID   (data is min-max scaled 0..1)
   Discriminator  x -> Dense 32 LeakyReLU -> Dropout -> Dense 16 LeakyReLU
                  -> Dense 1  SIGMOID            (binary: real vs fake)

   Leaky ReLU is the usual GAN hidden choice: it keeps gradients flowing for
   negative inputs, which stops the generator from stalling.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./dl-activations.js'));
  else { root.FYDL = root.FYDL || {}; root.FYDL.gan = factory(root.FYDL.activations); }
})(typeof self !== 'undefined' ? self : this, function (act) {
  'use strict';

  const COLUMNS = ['er', 'followers', 'reach', 'posts', 'likes', 'saves', 'shares', 'comments'];
  const DEFAULTS = { latentDim: 8, genUnits: [32, 32], discUnits: [32, 16], hidden: 'leaky', alpha: 0.2,
                     dropout: 0.2, learningRate: 0.002, beta1: 0.5, steps: 600, batchSize: 16, realLabel: 0.9 };

  /** client.series[platform] -> rows of COLUMNS (only weeks with an engagement value). */
  function seriesToRows(series, cols) {
    const c = cols || COLUMNS;
    const n = (series.er || []).length;
    const rows = [];
    for (let i = 0; i < n; i++) rows.push(c.map(function (k) { const v = Number(series[k] ? series[k][i] : 0); return isFinite(v) ? v : 0; }));
    return rows;
  }

  function fitScaler(rows) {
    const f = rows[0].length, lo = new Array(f).fill(Infinity), hi = new Array(f).fill(-Infinity);
    rows.forEach(function (r) { r.forEach(function (v, j) { lo[j] = Math.min(lo[j], v); hi[j] = Math.max(hi[j], v); }); });
    return { lo: lo, hi: hi };
  }
  function scaleRows(rows, sc) {
    return rows.map(function (r) { return r.map(function (v, j) { const d = sc.hi[j] - sc.lo[j]; return d === 0 ? 0.5 : (v - sc.lo[j]) / d; }); });
  }
  function unscaleRows(rows, sc) {
    return rows.map(function (r) { return r.map(function (v, j) { const d = sc.hi[j] - sc.lo[j]; return d === 0 ? sc.lo[j] : v * d + sc.lo[j]; }); });
  }

  function buildGenerator(tf, nFeatures, o) {
    const out = act.outputSpec('bounded01', nFeatures);
    const g = tf.sequential({ name: 'generator' });
    o.genUnits.forEach(function (u, i) {
      act.addHidden(tf, g, function (a) {
        const cfg = { units: u, activation: a, kernelInitializer: act.HIDDEN_INIT, name: 'g_dense' + (i + 1) + '_' + u };
        if (i === 0) cfg.inputShape = [o.latentDim];
        return tf.layers.dense(cfg);
      }, o.hidden, o.alpha);
    });
    g.add(tf.layers.dense({ units: out.units, activation: out.activation, name: 'g_out_' + out.activation }));
    return g;
  }

  function buildDiscriminator(tf, nFeatures, o) {
    const out = act.outputSpec('binary');
    const d = tf.sequential({ name: 'discriminator' });
    o.discUnits.forEach(function (u, i) {
      act.addHidden(tf, d, function (a) {
        const cfg = { units: u, activation: a, kernelInitializer: act.HIDDEN_INIT, name: 'd_dense' + (i + 1) + '_' + u };
        if (i === 0) cfg.inputShape = [nFeatures];
        return tf.layers.dense(cfg);
      }, o.hidden, o.alpha);
      if (i === 0 && o.dropout > 0) d.add(tf.layers.dropout({ rate: o.dropout, name: 'd_dropout' }));
    });
    d.add(tf.layers.dense({ units: 1, activation: out.activation, name: 'd_out_' + out.activation }));
    return d;
  }

  function bce(tf, labels, preds) {
    const p = preds.clipByValue(1e-7, 1 - 1e-7);
    return tf.neg(tf.add(labels.mul(p.log()), tf.sub(1, labels).mul(tf.sub(1, p).log()))).mean();
  }

  function variablesOf(model) { return model.trainableWeights.map(function (w) { return w.val; }); }

  /**
   * Train a GAN on real rows (array of arrays, real units).
   * Returns { generator, discriminator, scaler, columns, dLoss[], gLoss[] }.
   */
  async function trainGAN(tf, rows, options) {
    const o = Object.assign({}, DEFAULTS, options || {});
    if (!rows || rows.length < 3) throw new Error('Need at least 3 real rows to train the GAN');
    act.checkHidden(o.hidden);
    const nF = rows[0].length;
    const scaler = fitScaler(rows);
    const real = tf.tensor2d(scaleRows(rows, scaler));
    const G = buildGenerator(tf, nF, o), D = buildDiscriminator(tf, nF, o);
    const dOpt = tf.train.adam(o.learningRate, o.beta1), gOpt = tf.train.adam(o.learningRate, o.beta1);
    const dVars = variablesOf(D), gVars = variablesOf(G);
    const dLoss = [], gLoss = [];
    const b = o.batchSize;
    try {
      for (let step = 0; step < o.steps; step++) {
        const dl = dOpt.minimize(function () {
          const idx = tf.randomUniform([b], 0, rows.length, 'int32');
          const realBatch = tf.gather(real, idx);
          const fake = G.apply(tf.randomNormal([b, o.latentDim]), { training: true });
          const pr = D.apply(realBatch, { training: true }), pf = D.apply(fake, { training: true });
          return tf.add(bce(tf, tf.fill([b, 1], o.realLabel), pr), bce(tf, tf.zeros([b, 1]), pf));
        }, true, dVars);
        const gl = gOpt.minimize(function () {
          const fake = G.apply(tf.randomNormal([b, o.latentDim]), { training: true });
          return bce(tf, tf.ones([b, 1]), D.apply(fake, { training: true }));
        }, true, gVars);
        dLoss.push(dl.dataSync()[0]); gLoss.push(gl.dataSync()[0]); dl.dispose(); gl.dispose();
        if (o.onStep && step % 20 === 0) { o.onStep(step, o.steps, dLoss[step], gLoss[step]); await tf.nextFrame(); }
      }
    } finally { real.dispose(); dOpt.dispose(); gOpt.dispose(); }
    return { generator: G, discriminator: D, scaler: scaler, latentDim: o.latentDim, dLoss: dLoss, gLoss: gLoss };
  }

  /** Generate n synthetic rows in real units. Columns other than 'er' are rounded counts. */
  function generate(tf, trained, n, columns) {
    const cols = columns || COLUMNS;
    const scaled = tf.tidy(function () { return trained.generator.predict(tf.randomNormal([n, trained.latentDim])).arraySync(); });
    return unscaleRows(scaled, trained.scaler).map(function (r) {
      const o = { synthetic: true };
      r.forEach(function (v, j) { o[cols[j]] = cols[j] === 'er' ? +Math.max(0, v).toFixed(2) : Math.max(0, Math.round(v)); });
      return o;
    });
  }

  /** Mean / std per column, real vs synthetic - how close the generator got. */
  function compareStats(realRows, synthRows, columns) {
    const cols = columns || COLUMNS;
    function ms(a) { const m = a.reduce(function (x, y) { return x + y; }, 0) / a.length;
                     return { mean: m, std: Math.sqrt(a.reduce(function (s, v) { return s + (v - m) * (v - m); }, 0) / a.length) }; }
    return cols.map(function (c, j) {
      const r = ms(realRows.map(function (x) { return x[j]; })), s = ms(synthRows.map(function (x) { return x[c]; }));
      return { column: c, realMean: r.mean, synthMean: s.mean, realStd: r.std, synthStd: s.std };
    });
  }

  return { COLUMNS: COLUMNS, DEFAULTS: DEFAULTS, seriesToRows: seriesToRows, fitScaler: fitScaler, scaleRows: scaleRows,
           unscaleRows: unscaleRows, buildGenerator: buildGenerator, buildDiscriminator: buildDiscriminator,
           trainGAN: trainGAN, generate: generate, compareStats: compareStats };
});
