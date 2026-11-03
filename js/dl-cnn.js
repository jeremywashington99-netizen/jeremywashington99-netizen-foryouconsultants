/* ==========================================================================
   SEGMENT 2 - CNN for image processing

   Use in the site: learn from post images (thumbnails / screenshots).
     multiclass -> which content class an image belongs to   (softmax)
     binary     -> will this image be a high-engagement post   (sigmoid)
     regression -> expected engagement rate for the image      (linear)

   Architecture (hidden = ReLU or Leaky ReLU, uniform):
     Conv2D 16 3x3 -> act -> MaxPool 2x2
     Conv2D 32 3x3 -> act -> MaxPool 2x2
     Flatten -> Dense 64 -> act -> Dropout
     Dense out  (activation from the task)

   tf is passed in, so the same code runs in the browser (window.tf) and in
   Node tests (require('@tensorflow/tfjs')).
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./dl-activations.js'));
  else { root.FYDL = root.FYDL || {}; root.FYDL.cnn = factory(root.FYDL.activations); }
})(typeof self !== 'undefined' ? self : this, function (act) {
  'use strict';

  const DEFAULTS = {
    inputShape: [64, 64, 3],
    task: 'multiclass',
    nClasses: 2,
    hidden: 'relu',
    filters: [16, 32],
    kernelSize: 3,
    denseUnits: 64,
    dropout: 0.25,
    learningRate: 0.001
  };

  function buildCNN(tf, options) {
    const o = Object.assign({}, DEFAULTS, options || {});
    act.checkHidden(o.hidden);
    const out = act.outputSpec(o.task, o.task === 'multiclass' || o.task === 'multilabel' ? o.nClasses : 1);
    const m = tf.sequential({ name: 'cnn_' + o.task });
    o.filters.forEach(function (f, i) {
      act.addHidden(tf, m, function (a) {
        const cfg = { filters: f, kernelSize: o.kernelSize, padding: 'same', activation: a,
                      kernelInitializer: act.HIDDEN_INIT, name: 'conv' + (i + 1) + '_' + f };
        if (i === 0) cfg.inputShape = o.inputShape;
        return tf.layers.conv2d(cfg);
      }, o.hidden);
      m.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2, name: 'pool' + (i + 1) }));
    });
    m.add(tf.layers.flatten({ name: 'flatten' }));
    act.addHidden(tf, m, function (a) {
      return tf.layers.dense({ units: o.denseUnits, activation: a, kernelInitializer: act.HIDDEN_INIT, name: 'dense_' + o.denseUnits });
    }, o.hidden);
    if (o.dropout > 0) m.add(tf.layers.dropout({ rate: o.dropout, name: 'dropout' }));
    m.add(tf.layers.dense({ units: out.units, activation: out.activation, kernelInitializer: 'glorotUniform', name: 'out_' + out.activation }));
    m.compile({ optimizer: tf.train.adam(o.learningRate), loss: out.loss, metrics: out.metrics });
    m.fyTask = o.task;
    return m;
  }

  /**
   * Turn a [h, w, 3] pixel tensor (0..255) into a model-ready [size, size, 3]
   * float tensor scaled 0..1. In the browser get the pixels with
   * tf.browser.fromPixels(imgElement).
   */
  function preprocessPixels(tf, pixels, size) {
    const s = size || DEFAULTS.inputShape[0];
    return tf.tidy(function () {
      let t = pixels;
      if (t.shape.length !== 3) throw new Error('expected a [height, width, channels] tensor');
      if (t.shape[2] === 4) t = t.slice([0, 0, 0], [-1, -1, 3]);
      if (t.shape[2] === 1) t = t.tile([1, 1, 3]);
      return tf.image.resizeBilinear(t.toFloat(), [s, s]).div(255);
    });
  }

  /** Labels -> target tensor for the task. */
  function encodeLabels(tf, labels, task, nClasses) {
    if (task === 'multiclass') {
      labels.forEach(function (l) {
        if (!Number.isInteger(l) || l < 0 || l >= nClasses) throw new Error('class label ' + l + ' outside 0..' + (nClasses - 1));
      });
      return tf.oneHot(tf.tensor1d(labels, 'int32'), nClasses).toFloat();
    }
    if (task === 'binary') {
      labels.forEach(function (l) { if (l !== 0 && l !== 1) throw new Error('binary labels must be 0 or 1'); });
      return tf.tensor2d(labels, [labels.length, 1]);
    }
    return tf.tensor2d(labels.map(Number), [labels.length, 1]);
  }

  /** Mirror every image left-right to double a small training set. */
  function augmentFlip(tf, xs, ys) {
    return tf.tidy(function () {
      const flipped = tf.reverse(xs, 2);
      return { xs: tf.concat([xs, flipped], 0), ys: tf.concat([ys, ys], 0) };
    });
  }

  async function trainCNN(tf, model, xs, ys, options) {
    const o = Object.assign({ epochs: 20, batchSize: 8, validationSplit: 0, augment: true, onEpoch: null }, options || {});
    let x = xs, y = ys, made = false;
    if (o.augment) { const a = augmentFlip(tf, xs, ys); x = a.xs; y = a.ys; made = true; }
    const history = [];
    try {
      await model.fit(x, y, {
        epochs: o.epochs, batchSize: o.batchSize, shuffle: true, verbose: 0,
        validationSplit: o.validationSplit,
        callbacks: { onEpochEnd: function (e, logs) { history.push(logs.loss); if (o.onEpoch) o.onEpoch(e, o.epochs, logs); } }
      });
    } finally { if (made) { x.dispose(); y.dispose(); } }
    return history;
  }

  /** Predict one preprocessed image ([size,size,3]) and explain the result. */
  function predictImage(tf, model, image, classNames) {
    const task = model.fyTask;
    const values = tf.tidy(function () { return Array.from(model.predict(image.expandDims(0)).dataSync()); });
    if (task === 'multiclass') {
      let best = 0;
      values.forEach(function (v, i) { if (v > values[best]) best = i; });
      return { task: task, classIndex: best, label: classNames ? classNames[best] : String(best), confidence: values[best], probabilities: values };
    }
    if (task === 'binary') return { task: task, probability: values[0], label: values[0] >= 0.5 ? 'yes' : 'no' };
    return { task: task, value: values[0] };
  }

  return { DEFAULTS: DEFAULTS, buildCNN: buildCNN, preprocessPixels: preprocessPixels, encodeLabels: encodeLabels,
           augmentFlip: augmentFlip, trainCNN: trainCNN, predictImage: predictImage };
});
