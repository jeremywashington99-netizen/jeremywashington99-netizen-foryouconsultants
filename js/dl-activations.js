/* ==========================================================================
   SEGMENT 1 - Activation policy (shared by the CNN, RNN and GAN modules)

   Rule 1  Hidden layers  -> ReLU or Leaky ReLU, uniform across the network.
   Rule 2  Output layer   -> matches the analytics target:
             regression  (engagement rate, reach)     -> linear   + MSE
             positive    (counts that cannot go < 0)  -> softplus + MSE
             bounded01   (values min-max scaled 0..1) -> sigmoid  + MSE
             binary      (yes / no, real / fake)      -> sigmoid  + binary cross-entropy
             multiclass  (one content class of many)  -> softmax  + categorical cross-entropy
             multilabel  (several tags at once)       -> sigmoid  + binary cross-entropy

   Works in the browser (window.FYDL.activations) and in Node
   (require('./js/dl-activations.js')) so it can be unit tested in VS Code.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FYDL = root.FYDL || {}; root.FYDL.activations = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const HIDDEN_KINDS = ['relu', 'leaky'];
  const DEFAULT_LEAKY_ALPHA = 0.2;

  const OUTPUT_SPECS = {
    regression: { activation: 'linear',   loss: 'meanSquaredError',        metrics: ['mae'] },
    positive:   { activation: 'softplus', loss: 'meanSquaredError',        metrics: ['mae'] },
    bounded01:  { activation: 'sigmoid',  loss: 'meanSquaredError',        metrics: ['mae'] },
    binary:     { activation: 'sigmoid',  loss: 'binaryCrossentropy',      metrics: ['accuracy'] },
    multiclass: { activation: 'softmax',  loss: 'categoricalCrossentropy', metrics: ['accuracy'] },
    multilabel: { activation: 'sigmoid',  loss: 'binaryCrossentropy',      metrics: ['accuracy'] }
  };

  /** Output layer spec for an analytics task. units = size of the output layer. */
  function outputSpec(task, units) {
    const spec = OUTPUT_SPECS[task];
    if (!spec) throw new Error('Unknown task "' + task + '". Use one of: ' + Object.keys(OUTPUT_SPECS).join(', '));
    let n = units == null ? 1 : units;
    if (task === 'binary') n = 1;
    if (task === 'multiclass' && n < 2) throw new Error('multiclass needs at least 2 classes');
    if (!Number.isInteger(n) || n < 1) throw new Error('units must be a positive integer');
    return { units: n, activation: spec.activation, loss: spec.loss, metrics: spec.metrics.slice() };
  }

  function checkHidden(kind) {
    if (HIDDEN_KINDS.indexOf(kind) < 0) throw new Error('Hidden activation must be "relu" or "leaky", got "' + kind + '"');
    return kind;
  }

  /**
   * Add a hidden layer that follows the policy.
   *   relu  -> layer with activation 'relu'
   *   leaky -> layer with activation 'linear' followed by a LeakyReLU layer
   * makeLayer(activation) must return a tf layer (Dense, Conv2D ...).
   */
  function addHidden(tf, model, makeLayer, kind, alpha) {
    checkHidden(kind);
    if (kind === 'relu') { model.add(makeLayer('relu')); return; }
    model.add(makeLayer('linear'));
    model.add(tf.layers.leakyReLU({ alpha: alpha == null ? DEFAULT_LEAKY_ALPHA : alpha }));
  }

  /** He initialisation is the matching initializer for ReLU-family layers. */
  const HIDDEN_INIT = 'heNormal';

  /**
   * Audit a built tf.Sequential model against the policy.
   * Returns { ok, hidden:[...], output:{...}, problems:[...] }.
   * Layers that carry weights (dense, conv2d, lstm, gru, simplernn) are checked.
   * A weighted hidden layer passes when its activation is relu, or linear with
   * a LeakyReLU layer directly after it. Recurrent cells pass when the cell
   * activation is relu; their gates stay sigmoid because gates are 0..1
   * switches inside the cell, not a hidden-layer activation.
   */
  function auditModel(model, task) {
    const layers = model.layers;
    const problems = [];
    const hidden = [];
    const weighted = [];
    layers.forEach(function (l, i) {
      const cls = l.getClassName();
      if (/^(Dense|Conv1D|Conv2D|LSTM|GRU|SimpleRNN)$/.test(cls)) weighted.push(i);
    });
    if (!weighted.length) return { ok: false, hidden: [], output: null, problems: ['model has no weighted layers'] };
    const outIdx = weighted[weighted.length - 1];
    weighted.forEach(function (i) {
      if (i === outIdx) return;
      const l = layers[i];
      const cfg = l.getConfig();
      const act = cfg.activation;
      const next = layers[i + 1];
      let kind = null;
      if (act === 'relu') kind = 'relu';
      else if (act === 'linear' && next && next.getClassName() === 'LeakyReLU') kind = 'leaky';
      hidden.push({ name: l.name, type: l.getClassName(), activation: kind || act });
      if (!kind) problems.push(l.name + ' uses "' + act + '" - hidden layers must be relu or leaky relu');
    });
    const kinds = hidden.map(function (h) { return h.activation; }).filter(function (a, i, arr) { return arr.indexOf(a) === i; });
    if (kinds.length > 1) problems.push('hidden layers mix ' + kinds.join(' and ') + ' - keep them uniform');
    const outLayer = layers[outIdx];
    const outAct = outLayer.getConfig().activation;
    const output = { name: outLayer.name, activation: outAct };
    if (task) {
      const want = outputSpec(task, outLayer.getConfig().units).activation;
      output.expected = want;
      if (outAct !== want) problems.push('output ' + outLayer.name + ' uses "' + outAct + '" but task "' + task + '" needs "' + want + '"');
    }
    return { ok: problems.length === 0, hidden: hidden, output: output, problems: problems };
  }

  return {
    HIDDEN_KINDS: HIDDEN_KINDS,
    DEFAULT_LEAKY_ALPHA: DEFAULT_LEAKY_ALPHA,
    HIDDEN_INIT: HIDDEN_INIT,
    TASKS: Object.keys(OUTPUT_SPECS),
    outputSpec: outputSpec,
    checkHidden: checkHidden,
    addHidden: addHidden,
    auditModel: auditModel
  };
});
