/* ==========================================================================
   SEGMENT 10 - Keep trained models in Supabase, not in the browser

   Every training run is written to public.model_runs:
     metrics, forecast, plain-language summary, activation audit, and for the
     network and RNN the trained weights themselves. Opening the AI screen on
     any device then restores the last run from the database instead of
     retraining or relying on anything stored locally.

   Weights travel as base64 of the raw float32 buffer plus the layer topology,
   which is lossless: a restored model predicts exactly what the original did
   (the tests assert this to 6 decimal places).

   tf and the Supabase client are passed in, so this runs in the browser and
   in Node tests.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FYStore = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const KINDS = ['net', 'rnn', 'cnn', 'gan'];
  /* A row this large would make every AI-screen load slow; refuse it loudly
     rather than store something the page then struggles to read. */
  const MAX_WEIGHTS_BYTES = 900 * 1024;

  /* ---------- base64 <-> ArrayBuffer (browser and Node) ---------- */
  function toB64(buf) {
    const bytes = new Uint8Array(buf);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function fromB64(b64) {
    if (typeof Buffer !== 'undefined') {
      const b = Buffer.from(b64, 'base64');
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    }
    const bin = atob(b64), out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out.buffer;
  }

  /** Capture a tf.LayersModel as plain JSON. */
  async function serializeModel(tf, model) {
    let artifacts = null;
    await model.save(tf.io.withSaveHandler(async function (a) {
      artifacts = a;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' } };
    }));
    if (!artifacts) throw new Error('model produced no artifacts to save');
    const out = {
      format: 'fy-model-v1',
      topology: artifacts.modelTopology,
      weightSpecs: artifacts.weightSpecs,
      weightData: toB64(artifacts.weightData)
    };
    if (out.weightData.length > MAX_WEIGHTS_BYTES) {
      throw new Error('model weights are ' + Math.round(out.weightData.length / 1024) + ' KB, over the ' +
                      Math.round(MAX_WEIGHTS_BYTES / 1024) + ' KB limit for one database row');
    }
    return out;
  }

  /** Rebuild a model from serializeModel output. Caller compiles if it needs to train. */
  async function restoreModel(tf, blob) {
    if (!blob || blob.format !== 'fy-model-v1') throw new Error('not a saved Four You model');
    return tf.loadLayersModel(tf.io.fromMemory({
      modelTopology: blob.topology,
      weightSpecs: blob.weightSpecs,
      weightData: fromB64(blob.weightData)
    }));
  }

  /* ---------- database ---------- */

  function check(run) {
    if (!run || !run.client_id) throw new Error('a run needs a client_id');
    if (KINDS.indexOf(run.kind) < 0) throw new Error('kind must be one of ' + KINDS.join(', ') + ', got "' + run.kind + '"');
  }

  /**
   * Insert one training run. Returns { ok, id } or { ok:false, error } - it
   * never throws, because a failed save must not hide a successful training.
   * The caller shows the error; nothing is silently kept locally instead.
   */
  async function saveRun(sb, ownerId, run) {
    try {
      check(run);
      if (!sb) throw new Error('not connected to the database');
      if (!ownerId) throw new Error('no signed-in account');
      const row = {
        owner_id: ownerId,
        client_id: run.client_id,
        kind: run.kind,
        platform: run.platform || null,
        status: run.status || 'ok',
        model_name: run.model_name || (run.kind + (run.platform ? '_' + run.platform : '')),
        framework: 'tensorflow.js',
        backend: run.backend || null,
        architecture: run.architecture || {},
        hyperparams: run.hyperparams || null,
        epochs: run.epochs == null ? null : run.epochs,
        loss_final: run.loss_final == null ? null : run.loss_final,
        mae: run.mae == null ? null : run.mae,
        rmse: run.rmse == null ? null : run.rmse,
        r_squared: run.r_squared == null ? null : run.r_squared,
        train_rows: run.train_rows == null ? null : run.train_rows,
        data_weeks: run.data_weeks == null ? null : run.data_weeks,
        data_posts: run.data_posts == null ? null : run.data_posts,
        metrics: run.metrics || {},
        forecast: run.forecast || null,
        summary: run.summary || null,
        activation_audit: run.activation_audit || null,
        weights: run.weights || null,
        error_message: run.error_message || null
      };
      const r = await sb.from('model_runs').insert(row).select('id').maybeSingle();
      if (r && r.error) throw new Error(r.error.message);
      return { ok: true, id: r && r.data ? r.data.id : null };
    } catch (e) {
      return { ok: false, error: e.message || String(e) };
    }
  }

  /** The newest successful run of a kind for a client (optionally one platform). */
  async function latest(sb, clientId, kind, platform) {
    try {
      if (!sb) return { ok: false, error: 'not connected to the database' };
      let q = sb.from('model_runs').select('*').eq('client_id', clientId).eq('kind', kind).eq('status', 'ok');
      if (platform) q = q.eq('platform', platform);
      const r = await q.order('trained_at', { ascending: false }).limit(1);
      if (r.error) return { ok: false, error: r.error.message };
      const row = (r.data || [])[0] || null;
      return { ok: true, run: row };
    } catch (e) {
      return { ok: false, error: e.message || String(e) };
    }
  }

  /** Newest runs, any kind, for the history list. */
  async function history(sb, clientId, limit) {
    try {
      const r = await sb.from('model_runs')
        .select('id,kind,platform,status,trained_at,data_weeks,data_posts,mae,r_squared,train_rows,summary,error_message')
        .eq('client_id', clientId).order('trained_at', { ascending: false }).limit(limit || 10);
      if (r.error) return { ok: false, error: r.error.message, runs: [] };
      return { ok: true, runs: r.data || [] };
    } catch (e) {
      return { ok: false, error: e.message || String(e), runs: [] };
    }
  }

  /**
   * A saved run is only good while the data it learned from is unchanged.
   * "fingerprint" is how many weeks and posts it saw; if the client has more
   * now, the screen says the saved result is out of date rather than quietly
   * presenting it as current.
   */
  function staleness(run, weeks, posts) {
    if (!run) return { stale: false, reason: null };
    const w = run.data_weeks == null ? null : run.data_weeks;
    const p = run.data_posts == null ? null : run.data_posts;
    if (w != null && weeks > w) return { stale: true, reason: (weeks - w) + ' new week' + (weeks - w === 1 ? '' : 's') + ' added since this was trained' };
    if (p != null && posts > p) return { stale: true, reason: (posts - p) + ' new post' + (posts - p === 1 ? '' : 's') + ' logged since this was trained' };
    return { stale: false, reason: null };
  }

  return { KINDS: KINDS, MAX_WEIGHTS_BYTES: MAX_WEIGHTS_BYTES, toB64: toB64, fromB64: fromB64,
           serializeModel: serializeModel, restoreModel: restoreModel,
           saveRun: saveRun, latest: latest, history: history, staleness: staleness };
});
