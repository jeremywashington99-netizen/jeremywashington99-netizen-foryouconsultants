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
  /** A quick-moving level: new weeks count for 60%, so it follows a step change within a couple of weeks. */
  function fastLevel(hist, h) {
    let l = hist[0];
    for (let i = 1; i < hist.length; i++) l = 0.6 * hist[i] + 0.4 * l;
    return new Array(h).fill(l);
  }
  /**
   * Looks for a change in level: the latest point where the weeks since sit clearly away
   * (more than 2 normal swings) from the 6 weeks before it, and stay on that side.
   * If it finds one, it forecasts from the new level only, so old weeks stop dragging it back.
   */
  function sinceShift(hist, h) {
    const n = hist.length;
    for (let k = n - 3; k >= 6; k--) {
      const before = hist.slice(k - 6, k), after = hist.slice(k), mb = mean(before);
      const sd = Math.max(Math.sqrt(mean(before.map(function (v) { return (v - mb) * (v - mb); }))), 0.05 * Math.abs(mb), 0.05);
      const side = Math.sign(mean(after) - mb);
      if (Math.abs(mean(after) - mb) > 2 * sd && after.every(function (v) { return Math.sign(v - mb) === side; })) return new Array(h).fill(mean(after));
    }
    return recentMean(hist, h);
  }
  /** Level that follows change, plus a fading trend: anticipates direction without running away. */
  function blend(hist, h) {
    const a = fastLevel(hist, h), b = dampedTrend(hist, h);
    return a.map(function (v, i) { return (v + b[i]) / 2; });
  }
  const SIMPLE = { 'last week repeated': naive, 'recent average': recentMean, 'damped trend': dampedTrend,
                   'fast-adapting level': fastLevel, 'since last shift': sinceShift, 'blend of level and trend': blend };

  /** The standard error measures, all on weeks the method had not seen. */
  function metrics(pairs) {
    const n = pairs.length, act = pairs.map(function (p) { return p[1]; }), m = mean(act);
    const sse = pairs.reduce(function (a, p) { return a + (p[0] - p[1]) * (p[0] - p[1]); }, 0);
    const sst = act.reduce(function (a, v) { return a + (v - m) * (v - m); }, 0);
    const pct = pairs.filter(function (p) { return p[1] > 0; }).map(function (p) { return Math.abs(p[0] - p[1]) / p[1] * 100; });
    return { rmse: r2(Math.sqrt(sse / n)), mape: r2(mean(pct)), r2: sst > 0 ? r2(1 - sse / sst) : null, n: n };
  }

  /**
   * Did the method stay accurate THROUGH a change in the audience? Compares its misses on the
   * latest tests (which include the recent change) with the earlier tests.
   */
  function shiftTest(bt, name) {
    const r = bt && bt.ok && bt.results[name];
    if (!r || r.failed || !r.foldMae || r.foldMae.length < MIN_FOLDS) return null;
    const f = r.foldMae, cut = Math.ceil(f.length / 2), earlier = mean(f.slice(0, cut)), recent = mean(f.slice(cut));
    return { recent: r2(recent), earlier: r2(earlier), ratio: earlier > 0 ? r2(recent / earlier) : null, folds: f.length };
  }

  /** Does this method add anything over "next week = this week"? */
  function skill(bt, name) {
    const a = bt.results[name], b0 = bt.results['last week repeated'];
    if (!a || a.failed || !b0 || b0.failed || !(b0.mae > 0)) return null;
    return Math.round((1 - a.mae / b0.mae) * 100);
  }

  /**
   * Plain assessment table - the five things a reviewer asks:
   * continuity, trend, volatility, data sufficiency, verified accuracy.
   */
  function assess(bt, best, values, forecast, v) {
    const n = values.length, rows = [];
    const diffs = values.slice(1).map(function (x, i) { return Math.abs(x - values[i]); });
    const typical = mean(diffs) || 0.01;
    const jump = Math.abs(forecast[0] - values[n - 1]);
    rows.push({ category: 'Forecast continuity', rating: jump <= typical * 1.5 ? 'Good' : 'Check',
      note: jump <= typical * 1.5 ? 'Starts within a normal week-to-week move of the last real value.' : 'First forecast week jumps further than your usual weekly move.' });
    const recent = values.slice(-4), slope = (recent[recent.length - 1] - recent[0]) / (recent.length - 1), fslope = (forecast[3] - forecast[0]) / 3;
    const agree = Math.abs(slope) < typical * 0.15 || Math.abs(fslope) < typical * 0.15 || slope * fslope > 0;
    rows.push({ category: 'Trend consistency', rating: agree ? 'Reasonable' : 'Check',
      note: agree ? 'Direction fits the last 4 weeks (or both are flat).' : 'Heads the opposite way from the last 4 weeks; see the tests below for why.' });
    const sd = function (a) { const m = mean(a); return Math.sqrt(mean(a.map(function (x) { return (x - m) * (x - m); }))); };
    const ratio = sd(values) > 0 ? sd(forecast) / sd(values) : 1;
    rows.push({ category: 'Volatility', rating: ratio < 0.5 ? 'Smoother than history' : 'Similar to history',
      note: ratio < 0.5 ? 'Expected: a forecast is the most likely level, not a replay of random swings. The likely range shows how far a real week can stray.' : 'Moves about as much as your real weeks do.' });
    rows.push({ category: 'Data sufficiency', rating: n >= 20 ? 'Good' : n >= 12 ? 'Limited' : 'Insufficient',
      note: n + ' weeks. ' + (n >= 20 ? 'Enough for repeated testing.' : n >= 12 ? 'Enough to test, more will tighten it.' : 'Too few to test; treat as a rough guide.') });
    rows.push({ category: 'Verified accuracy', rating: v.label,
      note: bt.ok ? 'Tested on ' + bt.folds + ' past stretches; ' + (v.why[0] || '') : 'Not tested yet.' });
    return rows;
  }

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
      const errors = [], pairs = [];                      // errors[fold][horizon]; pairs of [forecast, actual]
      let failed = null;
      for (const k of origins) {
        try {
          const f = await forecasters[name](values.slice(0, k), h, k);
          if (!f || f.length < h || f.slice(0, h).some(function (v) { return !isFinite(v); })) throw new Error('invalid forecast');
          errors.push(f.slice(0, h).map(function (v, j) { pairs.push([v, values[k + j]]); return Math.abs(v - values[k + j]); }));
        } catch (e) { failed = e.message; break; }
      }
      if (failed || errors.length !== origins.length) { results[name] = { failed: failed || 'incomplete' }; continue; }
      const perHorizon = [];
      for (let j = 0; j < h; j++) perHorizon.push(r2(mean(errors.map(function (e) { return e[j]; }))));
      results[name] = Object.assign({ perHorizon: perHorizon, mae: r2(mean(perHorizon)), errors: errors,
        foldMae: errors.map(function (e) { return r2(mean(e)); }) }, metrics(pairs));
    }
    return { ok: true, folds: origins.length, origins: origins, horizon: h, weeks: n, results: results };
  }

  /** Lowest measured error wins; on a near-tie (within 3%) the simpler method wins. */
  function score(res, focusRecent) {
    if (!focusRecent || !res.foldMae || res.foldMae.length < MIN_FOLDS) return res.mae;
    const f = res.foldMae, recent = mean(f.slice(Math.floor(f.length / 2)));
    return 0.5 * res.mae + 0.5 * recent;            // when the audience is changing, how a method did lately matters as much as its whole record
  }
  function choose(bt, order, focusRecent) {
    const names = (order || Object.keys(bt.results)).filter(function (n) { return bt.results[n] && !bt.results[n].failed; });
    if (!names.length) return null;
    let best = names[0];
    names.forEach(function (n) { if (score(bt.results[n], focusRecent) < score(bt.results[best], focusRecent) * 0.97) best = n; });
    return best;
  }

  /** Measured likely range for each week ahead: the 80th-percentile real miss at that distance. */
  function bands(res, horizonNeeded, focusRecent) {
    const h = res.perHorizon.length, out = [];
    /* when the audience is changing, the range is built from the latest half of the tests so it widens with current volatility */
    const errs = focusRecent && res.errors.length >= MIN_FOLDS ? res.errors.slice(Math.floor(res.errors.length / 2)) : res.errors;
    for (let j = 0; j < horizonNeeded; j++) {
      if (j < h) out.push(r2(quantile(errs.map(function (e) { return e[j]; }), 0.8)));
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
           shiftTest: shiftTest, fastLevel: fastLevel, sinceShift: sinceShift, blend: blend, metrics: metrics, skill: skill, assess: assess, SIMPLE: SIMPLE, naive: naive, recentMean: recentMean, dampedTrend: dampedTrend,
           run: run, choose: choose, bands: bands, verdict: verdict, quantile: quantile };
});
