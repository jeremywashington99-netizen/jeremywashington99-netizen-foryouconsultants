/* ==========================================================================
   SEGMENT 14 - Walk-forward accuracy test ("backtest")

   A forecast is only worth showing if it has been right before. This replays
   history: stand at week k knowing only weeks 1..k, forecast the next 4 weeks,
   then compare with what really happened. Repeat for several k. Every method
   (the neural network and simple ones) takes the same test; the one with the
   smallest measured error is the one used. The size of its real misses also
   becomes the "likely range" - measured, not assumed.

   Pure functions; a forecaster is  async (history, h, k) -> [h numbers].
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FYBacktest = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const HORIZON = 4;          // weeks ahead that must be accurate
  const MIN_TRAIN = 8;        // weeks a method may use before its first test
  const MIN_FOLDS = 4;        // tests needed before anything is called verified
  const MAX_FOLDS = 6;        // newest tests only (keeps it quick, and relevant)
  const VERIFIED = 0.20;      // typical 4-week miss within 20% of the usual rate
  const DIRECTIONAL = 0.35;

  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / (a.length || 1); }
  function r2(v) { return Math.round(v * 100) / 100; }
  function quantile(a, q) {
    const s = a.slice().sort(function (x, y) { return x - y; });
    if (!s.length) return 0;
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }

  /* ---------- simple forecasters ---------- */
  function naive(hist, h) { const v = hist[hist.length - 1]; return new Array(h).fill(v); }
  function recentMean(hist, h) { return new Array(h).fill(mean(hist.slice(-6))); }
  /** Holt's damped trend: level + a trend that fades, so it never runs away. */
  function dampedTrend(hist, h) {
    const a = 0.5, b = 0.2, phi = 0.85;
    let l = hist[0], t = hist.length > 1 ? hist[1] - hist[0] : 0;
    for (let i = 1; i < hist.length; i++) {
      const prev = l;
      l = a * hist[i] + (1 - a) * (l + phi * t);
      t = b * (l - prev) + (1 - b) * phi * t;
    }
    const out = []; let damp = 0;
    for (let k = 1; k <= h; k++) { damp += Math.pow(phi, k); out.push(Math.max(0, l + damp * t)); }
    return out;
  }
  const SIMPLE = { 'last week repeated': naive, 'recent average': recentMean, 'damped trend': dampedTrend };

  /* ---------- the test ---------- */

  /**
   * values       weekly engagement rates, oldest first
   * forecasters  { name: async (history, h, k) -> [h] }
   * returns      { ok, folds, origins, results: { name: { perHorizon, mae, errors } } } or { ok:false, need }
   */
  async function run(values, forecasters, opts) {
    const o = opts || {}, h = o.horizon || HORIZON, minTrain = o.minTrain || MIN_TRAIN, maxFolds = o.maxFolds || MAX_FOLDS;
    const n = values.length;
    const last = n - h;                                  // latest origin that still has h real weeks after it
    const firstPossible = minTrain;
    if (last < firstPossible) return { ok: false, folds: 0, need: (minTrain + h) - n, weeks: n };
    const origins = [];
    for (let k = Math.max(firstPossible, last - maxFolds + 1); k <= last; k++) origins.push(k);
    const results = {};
    for (const name of Object.keys(forecasters)) {
      const errors = [];                                  // errors[fold][horizon]
      let failed = null;
      for (const k of origins) {
        try {
          const f = await forecasters[name](values.slice(0, k), h, k);
          if (!f || f.length < h || f.slice(0, h).some(function (v) { return !isFinite(v); })) throw new Error('invalid forecast');
          errors.push(f.slice(0, h).map(function (v, j) { return Math.abs(v - values[k + j]); }));
        } catch (e) { failed = e.message; break; }
      }
      if (failed || errors.length !== origins.length) { results[name] = { failed: failed || 'incomplete' }; continue; }
      const perHorizon = [];
      for (let j = 0; j < h; j++) perHorizon.push(r2(mean(errors.map(function (e) { return e[j]; }))));
      results[name] = { perHorizon: perHorizon, mae: r2(mean(perHorizon)), errors: errors };
    }
    return { ok: true, folds: origins.length, origins: origins, horizon: h, weeks: n, results: results };
  }

  /** Lowest measured error wins; on a near-tie (within 3%) the simpler method wins. */
  function choose(bt, order) {
    const names = (order || Object.keys(bt.results)).filter(function (n) { return bt.results[n] && !bt.results[n].failed; });
    if (!names.length) return null;
    let best = names[0];
    names.forEach(function (n) { if (bt.results[n].mae < bt.results[best].mae * 0.97) best = n; });
    return best;
  }

  /** Measured likely range for each week ahead: the 80th-percentile real miss at that distance. */
  function bands(res, horizonNeeded) {
    const h = res.perHorizon.length, out = [];
    for (let j = 0; j < horizonNeeded; j++) {
      if (j < h) out.push(r2(quantile(res.errors.map(function (e) { return e[j]; }), 0.8)));
      else out.push(r2(out[h - 1] * (1 + 0.15 * (j - h + 1))));      // beyond the tested distance: widen, and say untested
    }
    return out;
  }

  /** Plain-language verdict. */
  function verdict(bt, bestName, values) {
    if (!bt.ok) {
      return { status: 'learning', label: 'Still learning', tone: 'amber',
        headline: 'Not enough history to test the forecast yet.',
        why: ['It tests itself by forecasting weeks it has not seen, then checking. That needs ' + (MIN_TRAIN + HORIZON) + '+ weeks; you have ' + bt.weeks + '. About ' + bt.need + ' more.'],
        verified: false };
    }
    if (bt.folds < MIN_FOLDS) {
      const need = MIN_FOLDS - bt.folds;
      return { status: 'learning', label: 'Still learning', tone: 'amber',
        headline: 'Only ' + bt.folds + ' accuracy test' + (bt.folds === 1 ? '' : 's') + ' so far; it needs ' + MIN_FOLDS + ' to be called verified.',
        why: ['Add about ' + need + ' more week' + (need === 1 ? '' : 's') + ' and it will have enough tests.'], verified: false };
    }
    const res = bt.results[bestName], usual = mean(values.slice(-8)) || 1;
    const miss4 = res.perHorizon[res.perHorizon.length - 1], rel = miss4 / usual;
    const nv = bt.results['last week repeated'], beat = nv && !nv.failed ? Math.round((1 - res.mae / nv.mae) * 100) : null;
    const why = ['Across ' + bt.folds + ' past tests, the ' + bt.horizon + '-week-ahead forecast was typically ' + res.perHorizon[res.perHorizon.length - 1] + ' engagement points off (' + Math.round(rel * 100) + '% of your usual rate).'];
    if (beat != null) why.push(beat > 0 ? 'That is ' + beat + '% closer than just repeating last week.' : 'It is no closer than just repeating last week, so treat it as a level, not a trend.');
    if (rel <= VERIFIED) return { status: 'verified', label: 'Verified', tone: 'green', headline: 'Tested against real past weeks and accurate enough to plan with.', why: why, verified: true, relMiss: r2(rel), miss4: miss4 };
    if (rel <= DIRECTIONAL) return { status: 'directional', label: 'Directional only', tone: 'amber', headline: 'Right about the direction, loose on the exact number.', why: why.concat('Plan with the range, not the single figure.'), verified: false, relMiss: r2(rel), miss4: miss4 };
    return { status: 'unreliable', label: 'Not reliable yet', tone: 'red', headline: 'Past tests were too far off to trust a number.', why: why.concat('Your results swing more than the pattern it can find. More weeks or steadier posting will help.'), verified: false, relMiss: r2(rel), miss4: miss4 };
  }

  return { HORIZON: HORIZON, MIN_TRAIN: MIN_TRAIN, MIN_FOLDS: MIN_FOLDS, MAX_FOLDS: MAX_FOLDS, VERIFIED: VERIFIED, DIRECTIONAL: DIRECTIONAL,
           SIMPLE: SIMPLE, naive: naive, recentMean: recentMean, dampedTrend: dampedTrend,
           run: run, choose: choose, bands: bands, verdict: verdict, quantile: quantile };
});
