/* ==========================================================================
   SEGMENT 11 - The deep-learning results, in marketing language

   Turns a training run into what a social media marketer needs: where
   engagement is heading, how far to trust it, what it learned from, and what
   to do next. Layer names, activations, loss and parameter counts move into a
   collapsed "technical detail" section instead of being the headline.

   Rules this module keeps:
     - every statement is computed from the run and the client's own entries
     - "typical miss" is converted back into engagement points, the unit the
       marketer actually reads, rather than shown as a scaled number
     - a back-test on weeks the model has already seen flatters it, so the
       reliability wording says so
     - advice (best content type, best day, cadence) appears only when the
       client's own posts support it; otherwise it is left out, not guessed
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FYExplain = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  const FLAT_BAND = 0.05;          /* within +/-5% of today counts as steady */
  const EARLY_WEEKS = 8;           /* under this much history is an early read */
  const WIDEN_PER_WEEK = 0.30;     /* the possible miss grows by 30% for each week further out */

  function num(v) { const x = Number(v); return isFinite(x) ? x : null; }
  function r1(v) { return +Number(v).toFixed(1); }
  function r2(v) { return +Number(v).toFixed(2); }
  function median(a) {
    const s = a.slice().sort(function (x, y) { return x - y; });
    if (!s.length) return null;
    const i = Math.floor(s.length / 2);
    return s.length % 2 ? s[i] : (s[i - 1] + s[i]) / 2;
  }
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }

  /* ------------------------------------------------------------------
     Words for the technical parts
     ------------------------------------------------------------------ */

  const ACTIVATION_PLAIN = {
    relu: 'ignores weak or negative signals and passes strong ones through',
    leaky: 'passes strong signals through and keeps a faint trace of weak ones',
    linear: 'reports the number directly, with no cap',
    sigmoid: 'reports a likelihood between 0% and 100%',
    softmax: 'splits 100% between the options so they add up',
    softplus: 'reports a number that can never go below zero'
  };
  const ACTIVATION_LABEL = { relu: 'ReLU', leaky: 'Leaky ReLU', linear: 'Linear', sigmoid: 'Sigmoid', softmax: 'Softmax', softplus: 'Softplus' };

  /** One layer row -> what it does for the business. */
  function layerPlain(layer) {
    const type = String(layer.type || '').toLowerCase();
    const role = layer.role || 'hidden';
    if (/lstm|gru|simplernn/.test(type)) {
      return { stage: 'Memory', does: 'Reads the last few weeks in order and remembers the pattern - for example that engagement climbs after a run of saves.' };
    }
    if (/conv/.test(type)) {
      return { stage: 'Visual features', does: 'Scans a post image for shapes, colours and layout that tend to go with certain results.' };
    }
    if (role === 'output') {
      const a = layer.activation;
      const what = a === 'linear' ? 'the predicted engagement rate'
        : a === 'softmax' ? 'the most likely content type, as shares that add to 100%'
        : a === 'sigmoid' ? 'a yes / no likelihood'
        : a === 'softplus' ? 'a count that cannot be negative' : 'the prediction';
      return { stage: 'Prediction', does: 'Gives ' + what + '.' };
    }
    return { stage: 'Pattern finder', does: 'Combines signals such as reach, saves and posting frequency to spot what moves engagement.' };
  }

  function activationPlain(a) {
    const k = a === 'leaky' || a === 'Leaky ReLU' ? 'leaky' : String(a || '').toLowerCase().replace(/\s*relu$/, 'relu');
    return { label: ACTIVATION_LABEL[k] || a, plain: ACTIVATION_PLAIN[k] || '' };
  }

  /* ------------------------------------------------------------------
     Evidence from the client's own posts
     ------------------------------------------------------------------ */

  function bestContentType(posts, classNames) {
    const by = {};
    (posts || []).forEach(function (p) {
      const er = num(p.er); if (er == null || p.cls == null) return;
      (by[p.cls] = by[p.cls] || []).push(er);
    });
    const rows = Object.keys(by).map(function (k) { return { cls: Number(k), med: median(by[k]), n: by[k].length }; });
    if (rows.length < 2) return null;
    rows.sort(function (a, b) { return b.med - a.med; });
    /* one post proves nothing: only a type seen at least twice can be called strongest */
    const best = rows.filter(function (r) { return r.n >= 2; })[0];
    if (!best) return null;
    const rest = rows.filter(function (r) { return r !== best; });
    const others = median(rest.map(function (r) { return r.med; }));
    return { name: (classNames && classNames[best.cls]) || ('type ' + best.cls), med: r2(best.med), others: r2(others), n: best.n,
             lift: others > 0 ? Math.round((best.med / others - 1) * 100) : null };
  }

  function bestDay(posts) {
    const by = {};
    (posts || []).forEach(function (p) {
      const er = num(p.er); if (er == null || p.dow == null) return;
      (by[p.dow] = by[p.dow] || []).push(er);
    });
    const rows = Object.keys(by).map(function (k) { return { dow: Number(k), med: median(by[k]), n: by[k].length }; });
    if (rows.length < 2) return null;
    rows.sort(function (a, b) { return b.med - a.med; });
    const top = rows.filter(function (r) { return r.n >= 2; })[0];   // a day seen once is not a pattern
    if (!top) return null;
    return { day: DAYS[top.dow], med: r2(top.med), n: top.n, daysSeen: rows.length };
  }

  function cadence(series, benchPerWeek) {
    const posts = (series && series.posts || []).filter(function (v) { return num(v) != null; });
    if (!posts.length) return null;
    const avg = mean(posts);
    return { avg: r1(avg), bench: benchPerWeek == null ? null : benchPerWeek,
             below: benchPerWeek != null && avg < benchPerWeek * 0.8 };
  }

  /* ------------------------------------------------------------------
     The forecast, explained
     ------------------------------------------------------------------ */

  /**
   * in: {
   *   platform, platformLabel,
   *   series            the client's series for this platform
   *   forecast          [w1..w4] engagement rate %
   *   missPoints        typical miss in engagement points (already unscaled)
   *   posts             client's posts on this platform
   *   classNames        content type names
   *   benchPostsPerWeek published cadence benchmark (optional)
   *   source            'rnn' | 'net'
   * }
   */
  function explainForecast(inp) {
    const s = inp.series || {}, er = (s.er || []).filter(function (v) { return num(v) != null; });
    const label = inp.platformLabel || inp.platform;
    const f = (inp.forecast || []).map(Number).filter(isFinite);
    if (!er.length || !f.length) {
      return { ok: false, headline: 'Not enough to forecast ' + label + ' yet.', doNext: ['Add a few weeks of results for ' + label + ' and this fills in.'] };
    }
    const last = er[er.length - 1];
    const end = f[f.length - 1];
    const peak = Math.max.apply(null, f);
    const rel = last > 0 ? (end - last) / last : 0;
    const direction = Math.abs(rel) <= FLAT_BAND ? 'flat' : rel > 0 ? 'up' : 'down';
    const miss = inp.missPoints == null ? null : Math.max(0, inp.missPoints);

    const rows = f.map(function (v, i) {
      const spread = miss == null ? null : miss * (1 + WIDEN_PER_WEEK * i);
      return { week: i + 1, expected: r2(v),
               low: spread == null ? null : r2(Math.max(0, v - spread)),
               high: spread == null ? null : r2(v + spread) };
    });

    /* How far to trust it. */
    const avgEr = mean(er.slice(-6));
    const ratio = miss != null && avgEr > 0 ? miss / avgEr : null;
    let level = ratio == null ? 'unknown' : ratio < 0.10 ? 'strong' : ratio < 0.25 ? 'useful' : 'rough';
    const early = er.length < EARLY_WEEKS;
    let label2 = level === 'strong' ? 'Strong guide' : level === 'useful' ? 'Useful guide' : level === 'rough' ? 'Rough guide' : 'Unrated';
    if (early) label2 = 'Early read';
    const why = [];
    if (miss != null) why.push('On the weeks it replayed, it was typically ' + r2(miss) + ' engagement point' + (r2(miss) === 1 ? '' : 's') + ' off.');
    if (early) why.push('With ' + plural(er.length, 'week') + ' of history this is an early read - it firms up past ' + EARLY_WEEKS + ' weeks.');
    why.push('That replay uses weeks the model already learned from, so real accuracy will be a little worse than it suggests.');

    /* What to do. Each line is tied to something in the client's data. */
    const doNext = [];
    if (direction === 'up') {
      doNext.push('Momentum is building. Keep the current rhythm and the content that is working rather than changing the formula.');
    } else if (direction === 'down') {
      doNext.push('Engagement is expected to cool. Refresh the creative - new hooks, a different format - before the dip arrives, not after.');
    } else {
      doNext.push('Engagement looks steady. A good moment to test one new format against your usual, since a clear result will stand out.');
    }
    const bc = bestContentType(inp.posts, inp.classNames);
    if (bc) {
      doNext.push('Your strongest content type here is ' + bc.name + ' (median ' + bc.med + '% engagement' +
        (bc.lift != null && bc.lift > 0 ? ', ' + bc.lift + '% above your other types' : '') + ', from ' + plural(bc.n, 'post') + '). Lean on it.');
    }
    const bd = bestDay(inp.posts);
    if (bd) doNext.push(bd.day + ' has been your best posting day on ' + label + ' (median ' + bd.med + '%, across ' + bd.daysSeen + ' days compared).');
    const cad = cadence(s, inp.benchPostsPerWeek);
    if (cad && cad.below) {
      doNext.push('You average ' + cad.avg + ' posts a week on ' + label + '; the typical account posts about ' + cad.bench +
        '. More consistent posting is one of the cheapest ways to lift reach.');
    }

    const verb = direction === 'up' ? 'climb' : direction === 'down' ? 'dip' : 'hold steady';
    const headline = direction === 'flat'
      ? label + ' engagement is likely to hold steady around ' + r2(end) + '% over the next ' + f.length + ' weeks.'
      : label + ' engagement is likely to ' + verb + ' from ' + r2(last) + '% to about ' + r2(end) + '% over the next ' + f.length + ' weeks.';

    return {
      ok: true, platform: inp.platform, label: label, direction: direction, headline: headline,
      lastWeek: r2(last), expectedEnd: r2(end), peak: r2(peak), changePct: Math.round(rel * 100),
      rows: rows, missPoints: miss == null ? null : r2(miss),
      reliability: { level: early ? 'early' : level, label: label2, why: why },
      learnedFrom: 'It studied ' + plural(er.length, 'week') + ' of ' + label + ' results' +
        (inp.posts && inp.posts.length ? ' and ' + plural(inp.posts.length, 'logged post') : '') +
        ', repeatedly reading a few weeks in a row and checking what happened the week after.',
      doNext: doNext, evidence: { bestContentType: bc, bestDay: bd, cadence: cad }
    };
  }

  /** Convert a scaled mean error back to engagement points for one platform. */
  function missInPoints(maeScaled, scaler, erIndex) {
    if (maeScaled == null || !scaler) return null;
    const i = erIndex == null ? 0 : erIndex;
    const range = scaler.hi[i] - scaler.lo[i];
    return range > 0 ? maeScaled * range : null;
  }

  /* ------------------------------------------------------------------
     The "Train Model" panel
     ------------------------------------------------------------------ */

  function netSummary(m) {
    const r2v = num(m.r2), rows = num(m.rows);
    const fit = r2v == null ? null : Math.max(0, Math.min(1, r2v));
    /* same cut-offs as the accuracy safeguards, so the headline and the checks never disagree */
    const verdict = fit == null ? 'Not rated'
      : fit >= 0.85 ? 'Follows your past results closely'
      : fit >= 0.55 ? 'Follows the broad pattern'
      : fit >= 0.30 ? 'Finds part of the pattern'
      : 'Struggling to find a pattern';
    return {
      verdict: verdict,
      fitPct: fit == null ? null : Math.round(fit * 100),
      line: fit == null ? 'No rating yet.'
        : 'When replayed over weeks it had already seen, it matched ' + Math.round(fit * 100) + '% of the ups and downs in your engagement.',
      learnedFrom: rows == null ? '' : 'It learned from ' + plural(rows, 'week-to-week example') +
        ' - each one a week of results paired with what happened the week after.',
      caveat: 'This is a replay of weeks it learned from, so treat it as a ceiling, not a promise. Fresh weeks will score a little lower.',
      weak: fit != null && fit < 0.3
        ? 'The pattern is weak, usually because the weeks vary a lot or there are few of them. Keep adding weeks before relying on the forecast.' : null
    };
  }

  /** What each chart title means to a marketer. */
  const CHART_TITLES = {
    loss: { title: 'How its accuracy improved while learning', sub: 'The line should fall and then flatten. Falling means it is getting better at matching your results.' },
    fit: { title: 'Forecast check: what it predicted vs what happened', sub: 'Green is what really happened. Gold is what the model expected. The closer they sit, the more you can trust it.' },
    card: { title: 'How the forecast engine works', sub: null }
  };

  return { ACTIVATION_PLAIN: ACTIVATION_PLAIN, CHART_TITLES: CHART_TITLES,
           layerPlain: layerPlain, activationPlain: activationPlain,
           bestContentType: bestContentType, bestDay: bestDay, cadence: cadence,
           explainForecast: explainForecast, missInPoints: missInPoints, netSummary: netSummary,
           EARLY_WEEKS: EARLY_WEEKS, WIDEN_PER_WEEK: WIDEN_PER_WEEK };
});
