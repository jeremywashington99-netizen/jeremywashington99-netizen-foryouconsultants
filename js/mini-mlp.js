/* ==========================================================================
   SEGMENT 16 - A small neural-network trainer that needs nothing but JavaScript

   Dense layers with ReLU / linear / softmax, dropout, an L2 weight penalty,
   the Adam optimiser, mean-squared-error and cross-entropy losses. It does the
   same job as the TensorFlow.js layers the engagement network used before, but
   with no library to load, so it trains in a fraction of a second inside a
   Supabase edge function (TensorFlow.js on the edge's processor does not fit
   within an edge function's compute allowance, even for a network this small).

   Trained weights are written out in the exact layout TensorFlow.js reads
   ("fy-model-v1"), so the browser can load a model trained here and use it to
   predict, with the tests proving both give the same numbers.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FYMlp = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  /* seeded generator for tests; Math.random in production */
  function seeded(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); const v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function truncNormal(r, std) { for (;;) { const z = gauss(r); if (Math.abs(z) <= 2) return z * std; } }   /* re-draw beyond 2 standard deviations, as TensorFlow does */

  function toB64(buf) {
    const bytes = new Uint8Array(buf);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  /**
   * spec = { name, inputDim, loss:'mse'|'cce', lr, layers:[{name, units, act:'relu'|'linear'|'softmax', init:'he'|'glorot', l2?}],
   *          dropout?:{ after:<layer index>, rate, name } }
   */
  function create(spec, rand) {
    rand = rand || Math.random;
    const L = spec.layers, nL = L.length;
    const dims = [spec.inputDim].concat(L.map(l => l.units));
    const W = [], B = [];
    L.forEach((l, k) => {
      const fi = dims[k], fo = dims[k + 1];
      const std = l.init === 'he' ? Math.sqrt(2 / fi) : Math.sqrt(2 / (fi + fo));
      const w = new Float64Array(fi * fo);
      for (let i = 0; i < w.length; i++) w[i] = truncNormal(rand, std);
      W.push(w); B.push(new Float64Array(fo));
    });
    const zerosLike = a => a.map(x => new Float64Array(x.length));
    let mW = zerosLike(W), vW = zerosLike(W), mB = zerosLike(B), vB = zerosLike(B), step = 0;
    const drop = spec.dropout || null;
    const model = { spec: spec, stopTraining: false, optimizerDisposed: false };

    function forward(x, train) {
      const acts = [x], zs = [], masks = [];
      let a = x;
      for (let k = 0; k < nL; k++) {
        const fi = dims[k], fo = dims[k + 1], z = new Float64Array(fo), w = W[k];
        for (let j = 0; j < fo; j++) z[j] = B[k][j];
        for (let i = 0; i < fi; i++) { const ai = a[i]; if (ai !== 0) { const row = i * fo; for (let j = 0; j < fo; j++) z[j] += ai * w[row + j]; } }
        let out;
        if (L[k].act === 'relu') { out = new Float64Array(fo); for (let j = 0; j < fo; j++) out[j] = z[j] > 0 ? z[j] : 0; }
        else if (L[k].act === 'softmax') {
          let mx = -Infinity; for (let j = 0; j < fo; j++) if (z[j] > mx) mx = z[j];
          out = new Float64Array(fo); let sum = 0;
          for (let j = 0; j < fo; j++) { out[j] = Math.exp(z[j] - mx); sum += out[j]; }
          for (let j = 0; j < fo; j++) out[j] /= sum;
        } else out = z.slice();
        let mask = null;
        if (train && drop && drop.after === k) {
          const keep = 1 - drop.rate; mask = new Float64Array(fo);
          for (let j = 0; j < fo; j++) { mask[j] = rand() < keep ? 1 / keep : 0; out[j] *= mask[j]; }
        }
        zs.push(z); masks.push(mask); acts.push(out); a = out;
      }
      return { acts: acts, zs: zs, masks: masks };
    }

    function predict(rows) { return rows.map(r => Array.from(forward(r, false).acts[nL])); }
    function mseOf(X, Y) { let s = 0; for (let i = 0; i < X.length; i++) { const p = forward(X[i], false).acts[nL][0], e = p - Y[i][0]; s += e * e; } return s / X.length; }

    /**
     * Train. o = { epochs, batchSize, shuffle, sampleWeights?, classWeights?, validation?:{X,Y}, onEpochEnd?(e, logs) }
     * A callback (or the caller) may set model.stopTraining = true to end early.
     */
    function fit(X, Y, o) {
      o = o || {};
      const n = X.length, bs = o.batchSize || 8, epochs = o.epochs || 1;
      const lr = spec.lr, b1 = 0.9, b2 = 0.999, eps = 1e-7;
      const idx = Array.from({ length: n }, (_, i) => i);
      const history = { loss: [], val_loss: [] };
      model.stopTraining = false;
      const sw = i => {
        if (o.sampleWeights) return o.sampleWeights[i];
        if (o.classWeights) { const c = Y[i].indexOf(1); const cw = o.classWeights[c]; return cw == null ? 1 : cw; }
        return 1;
      };
      for (let e = 0; e < epochs; e++) {
        if (o.shuffle !== false) for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; }
        let lossSum = 0;
        for (let b0 = 0; b0 < n; b0 += bs) {
          const batch = idx.slice(b0, b0 + bs), nb = batch.length;
          const gW = zerosLike(W), gB = zerosLike(B);
          let bl = 0;
          for (const s of batch) {
            const f = forward(X[s], true), out = f.acts[nL], w = sw(s);
            let d;
            if (spec.loss === 'cce') {
              d = new Float64Array(out.length);
              for (let j = 0; j < out.length; j++) { d[j] = (out[j] - Y[s][j]) * w / nb; bl -= w * Y[s][j] * Math.log(Math.max(out[j], 1e-7)) / nb; }
            } else { const err = out[0] - Y[s][0]; d = Float64Array.of(2 * err * w / nb); bl += w * err * err / nb; }
            for (let k = nL - 1; k >= 0; k--) {
              const fi = dims[k], fo = dims[k + 1], a = f.acts[k], z = f.zs[k], mk = f.masks[k];
              if (mk) for (let j = 0; j < fo; j++) d[j] *= mk[j];
              if (L[k].act === 'relu') for (let j = 0; j < fo; j++) if (z[j] <= 0) d[j] = 0;
              const dprev = k > 0 ? new Float64Array(fi) : null;
              for (let j = 0; j < fo; j++) gB[k][j] += d[j];
              for (let i = 0; i < fi; i++) {
                const row = i * fo, ai = a[i];
                if (ai !== 0) for (let j = 0; j < fo; j++) gW[k][row + j] += ai * d[j];
                if (dprev) { let acc = 0; for (let j = 0; j < fo; j++) acc += W[k][row + j] * d[j]; dprev[i] = acc; }
              }
              d = dprev;
            }
          }
          for (let k = 0; k < nL; k++) if (L[k].l2) {
            let sq = 0; const w = W[k];
            for (let i = 0; i < w.length; i++) { sq += w[i] * w[i]; gW[k][i] += 2 * L[k].l2 * w[i]; }
            bl += L[k].l2 * sq;
          }
          step++;
          const lrT = lr * Math.sqrt(1 - Math.pow(b2, step)) / (1 - Math.pow(b1, step));
          for (let k = 0; k < nL; k++) {
            const upd = (p, g, m, v) => { for (let i = 0; i < p.length; i++) { m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i]; p[i] -= lrT * m[i] / (Math.sqrt(v[i]) + eps); } };
            upd(W[k], gW[k], mW[k], vW[k]); upd(B[k], gB[k], mB[k], vB[k]);
          }
          lossSum += bl * nb;
        }
        const logs = { loss: lossSum / n };
        if (o.validation) logs.val_loss = mseOf(o.validation.X, o.validation.Y);
        history.loss.push(logs.loss); if (logs.val_loss != null) history.val_loss.push(logs.val_loss);
        if (o.onEpochEnd) o.onEpochEnd(e, logs);
        if (model.stopTraining) break;
      }
      return { history: history };
    }

    /* weights as plain arrays [kernel0, bias0, kernel1, bias1, ...] */
    function getWeights() { const out = []; for (let k = 0; k < nL; k++) { out.push(W[k].slice()); out.push(B[k].slice()); } return out; }
    function setWeights(ws) { for (let k = 0; k < nL; k++) { W[k].set(ws[2 * k]); B[k].set(ws[2 * k + 1]); } }
    function countParams() { let n = 0; for (let k = 0; k < nL; k++) n += W[k].length + B[k].length; return n; }

    /* ---- the layout TensorFlow.js loads ---- */
    function topology() {
      const layers = [];
      L.forEach((l, k) => {
        const cfg = {
          units: l.units, activation: l.act, use_bias: true,
          kernel_initializer: l.init === 'he'
            ? { class_name: 'VarianceScaling', config: { scale: 2, mode: 'fan_in', distribution: 'normal', seed: null } }
            : { class_name: 'VarianceScaling', config: { scale: 1, mode: 'fan_avg', distribution: 'normal', seed: null } },
          bias_initializer: { class_name: 'Zeros', config: {} },
          kernel_regularizer: l.l2 ? { class_name: 'L1L2', config: { l1: 0, l2: l.l2 } } : null,
          bias_regularizer: null, activity_regularizer: null, kernel_constraint: null, bias_constraint: null,
          name: l.name, trainable: true
        };
        if (k === 0) { cfg.batch_input_shape = [null, spec.inputDim]; cfg.dtype = 'float32'; }
        layers.push({ class_name: 'Dense', config: cfg });
        if (drop && drop.after === k) layers.push({ class_name: 'Dropout', config: { rate: drop.rate, noise_shape: null, seed: null, name: drop.name, trainable: true } });
      });
      return { class_name: 'Sequential', config: { name: spec.name || 'sequential_1', layers: layers }, keras_version: 'tfjs-layers 4.17.0', backend: 'tensor_flow.js' };
    }
    function toBlob() {
      const specs = [], parts = [];
      L.forEach((l, k) => {
        specs.push({ name: l.name + '/kernel', shape: [dims[k], dims[k + 1]], dtype: 'float32' });
        specs.push({ name: l.name + '/bias', shape: [dims[k + 1]], dtype: 'float32' });
        parts.push(Float32Array.from(W[k]), Float32Array.from(B[k]));
      });
      const total = parts.reduce((s, p) => s + p.length, 0), all = new Float32Array(total);
      let o = 0; parts.forEach(p => { all.set(p, o); o += p.length; });
      return { format: 'fy-model-v1', topology: topology(), weightSpecs: specs, weightData: toB64(all.buffer) };
    }
    /* what the architecture table shows */
    function describe(tag) {
      const rows = [];
      L.forEach((l, k) => {
        rows.push({ head: tag, name: l.name, activation: l.act, units: l.units, weights: dims[k] + ' x ' + dims[k + 1], bias: String(dims[k + 1]), params: W[k].length + B[k].length });
        if (drop && drop.after === k) rows.push({ head: tag, name: drop.name, activation: 'dropout', units: '-', weights: '-', bias: '-', params: 0 });
      });
      return rows;
    }

    Object.assign(model, { forward: forward, predict: predict, fit: fit, getWeights: getWeights, setWeights: setWeights, countParams: countParams,
                           topology: topology, toBlob: toBlob, describe: describe, mse: mseOf });
    return model;
  }

  return { create: create, seeded: seeded };
});
