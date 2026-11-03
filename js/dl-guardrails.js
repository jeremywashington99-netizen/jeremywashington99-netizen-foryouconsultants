/* ==========================================================================
   SEGMENT 13 - Guardrails against the 10 classic neural-network accuracy problems

     1 Overfitting            held-back weeks (never trained on) vs trained weeks
     2 Underfitting           how well it matches even the weeks it learned from
     3 Class imbalance        class weights + warning when one content type dominates
     4 Noisy / bad data       range, duplicate, missing and outlier checks
     5 Data leakage           time-ordered split, and a probe proving week N never sees week N+1
     6 Bias / variance        read off checks 1 and 2 together
     7 Gradient problems      non-finite or dead weights, exploding loss
     8 Learning rate          diverging, bouncing or stuck loss curve
     9 Distribution shift     recent weeks vs the weeks it learned from
    10 Feature scaling        every input on a comparable 0..1 scale

   Pure functions. Each check returns
     { id, n, title, status: ok | watch | problem | unknown, plain, fix }
   written for a marketer, not an engineer.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.FYGuard = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const MIN_FOR_HOLDOUT = 12;        // pairs needed before a held-back check is meaningful
  const HOLDOUT_FRACTION = 0.25;
  const SKEW_SHARE = 0.60;           // one class above this share is "dominant"
  const MAX_CLASS_WEIGHT = 5;

  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / (a.length || 1); }
  function sd(a) { const m = mean(a); return Math.sqrt(mean(a.map(function (v) { return (v - m) * (v - m); }))); }
  function median(a) { const s = a.slice().sort(function (x, y) { return x - y; }), n = s.length; return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : 0; }
  function r2(v) { return Math.round(v * 100) / 100; }
  function chk(id, n, title, status, plain, fix) { return { id: id, n: n, title: title, status: status, plain: plain, fix: fix || '' }; }

  /* ---------- helpers used while training ---------- */

  /** Oldest weeks train, newest weeks are held back. Never shuffled across the boundary. */
  function timeSplit(n, frac) {
    const hold = Math.max(2, Math.floor(n * (frac == null ? HOLDOUT_FRACTION : frac)));
    const cut = n - hold, train = [], held = [];
    for (let i = 0; i < n; i++) (i < cut ? train : held).push(i);
    return { train: train, hold: held, cut: cut };
  }

  /** Inverse-frequency weights so rare content types are not ignored. Absent classes get 0. */
  function classWeights(counts) {
    const total = counts.reduce(function (a, b) { return a + b; }, 0);
    const present = counts.filter(function (c) { return c > 0; }).length;
    const out = {};
    counts.forEach(function (c, i) { out[i] = c > 0 ? Math.min(MAX_CLASS_WEIGHT, total / (present * c)) : 0; });
    return out;
  }

  /** Check every week and field; returns issues and the set of week indexes to leave out. */
  function validateSeries(s) {
    const issues = [], bad = {};
    if (!s) return { issues: issues, bad: bad, count: 0 };
    const n = (s.er || []).length;
    const fieldRules = { er: [0, 100], followers: [0, Infinity], reach: [0, Infinity], posts: [0, Infinity], saves: [0, Infinity] };
    Object.keys(fieldRules).forEach(function (f) {
      (s[f] || []).forEach(function (v, i) {
        const rule = fieldRules[f];
        if (v == null || !isNum(Number(v))) { issues.push({ field: f, week: i, problem: 'missing or not a number' }); bad[i] = 1; }
        else if (Number(v) < rule[0] || Number(v) > rule[1]) { issues.push({ field: f, week: i, problem: 'outside the possible range (' + v + ')' }); bad[i] = 1; }
      });
    });
    if (s.dates) {
      const seen = {};
      s.dates.forEach(function (d, i) { if (seen[d] != null) { issues.push({ field: 'dates', week: i, problem: 'same week entered twice' }); bad[i] = 1; } seen[d] = i; });
    }
    const er = (s.er || []).map(Number).filter(isNum);
    if (er.length >= 6) {
      const m = median(er), mad = Math.max(median(er.map(function (v) { return Math.abs(v - m); })), 0.15 * Math.abs(m), 0.05);   /* floor stops tiny samples flagging normal weeks */
      (s.er || []).forEach(function (v, i) {
        if (isNum(Number(v)) && Math.abs(0.6745 * (v - m) / mad) > 5) issues.push({ field: 'er', week: i, problem: 'far from every other week (' + v + '%) - typo or a one-off spike' });
      });
    }
    return { issues: issues, bad: bad, count: n };
  }

  /** Prove a week's inputs never include later weeks: change the future, features must not move. */
  function leakProbe(featuresFn, s, norms) {
    const n = (s.er || []).length;
    const keys = ['er', 'followers', 'reach', 'posts', 'paid', 'saves'];
    const leaks = [];
    for (let i = 1; i < n - 1; i++) {
      const base = featuresFn(s, i, norms);
      const alt = {};
      keys.forEach(function (k) { alt[k] = (s[k] || []).map(function (v, j) { return j > i ? (Number(v) || 0) * 1.7 + 11 : v; }); });
      Object.keys(s).forEach(function (k) { if (alt[k] === undefined) alt[k] = s[k]; });
      const moved = featuresFn(alt, i, norms);
      for (let c = 0; c < base.length; c++) if (Math.abs(base[c] - moved[c]) > 1e-12) { leaks.push({ week: i, feature: c }); break; }
    }
    return { ok: leaks.length === 0, leaks: leaks };
  }

  /** Weights that are NaN/Infinity, or mostly zero (dead units), mean learning broke. */
  function weightsHealth(arrays) {
    let total = 0, bad = 0, zero = 0;
    arrays.forEach(function (a) { for (let i = 0; i < a.length; i++) { total++; if (!isNum(a[i])) bad++; else if (Math.abs(a[i]) < 1e-8) zero++; } });
    return { ok: bad === 0, nonFinite: bad, deadShare: total ? zero / total : 0, total: total };
  }

  function lrDiagnosis(h) {
    if (!h || h.length < 10) return { state: 'unknown' };
    if (h.some(function (v) { return !isNum(v); })) return { state: 'exploded' };
    const min = Math.min.apply(null, h), first = h[0], last = h[h.length - 1];
    if (Math.max.apply(null, h) > first * 10 && first > 0) return { state: 'exploded' };
    const tail = h.slice(-30); let flips = 0;
    for (let i = 2; i < tail.length; i++) if ((tail[i] - tail[i - 1]) * (tail[i - 1] - tail[i - 2]) < 0) flips++;
    if (tail.length >= 20 && flips / (tail.length - 2) > 0.65 && (Math.max.apply(null, tail) - Math.min.apply(null, tail)) > 0.2 * Math.abs(mean(tail))) return { state: 'bouncing' };
    if (min > 0 && last > min * 2 && h.indexOf(min) < h.length * 0.8) return { state: 'diverging' };
    if (h.length >= 50 && first > 0 && (first - last) / first < 0.10) return { state: 'stuck' };
    return { state: 'ok', drop: first > 0 ? (first - last) / first : 0 };
  }

  /** Recent weeks vs the weeks before: has the audience's behaviour moved? */
  function drift(s, recent) {
    const k = recent || 3, er = (s && s.er || []).map(Number).filter(isNum);
    if (er.length < 8) return { state: 'unknown', n: er.length };
    const old = er.slice(0, -k), now = er.slice(-k), sdv = sd(old) || 0.0001;
    const z = (mean(now) - mean(old)) / sdv;
    return { state: Math.abs(z) > 2 ? 'shifted' : Math.abs(z) > 1.2 ? 'moving' : 'stable', z: r2(z), before: r2(mean(old)), now: r2(mean(now)) };
  }

  function scaling(X) {
    if (!X || !X.length) return { state: 'unknown' };
    const cols = X[0].length, constant = [], wide = [];
    for (let c = 0; c < cols; c++) {
      const col = X.map(function (r) { return r[c]; });
      if (col.some(function (v) { return !isNum(v); })) return { state: 'broken' };
      const lo = Math.min.apply(null, col), hi = Math.max.apply(null, col);
      if (hi - lo < 1e-9) constant.push(c);
      if (lo < -0.5 || hi > 1.5) wide.push(c);
    }
    return { state: wide.length ? 'unscaled' : constant.length ? 'flat' : 'ok', constant: constant, wide: wide, cols: cols };
  }

  /* ---------- the audit ---------- */

  /**
   * i: {
   *   history       loss per epoch
   *   trainErr      mean absolute error on trained weeks (scaled 0..1) or null
   *   holdErr       mean absolute error on held-back weeks, or null
   *   holdN         how many held-back examples
   *   r2            match on trained weeks (0..1)
   *   rows          number of training examples
   *   classCounts   examples per content type, or null
   *   dataIssues    validateSeries output (merged over platforms) or null
   *   leak          leakProbe output or null
   *   health        weightsHealth output or null
   *   drift         drift() output or null
   *   scale         scaling() output or null
   *   weeks         weeks of history
   * }
   */
  function audit(i) {
    const c = [];
    const hasHold = i.holdErr != null && i.trainErr != null && i.holdN > 0;
    const ratio = hasHold ? (i.holdErr + 0.01) / (i.trainErr + 0.01) : null;

    // 1 overfitting
    if (!hasHold) {
      c.push(chk('overfit', 1, 'Memorising instead of learning', 'unknown',
        'With ' + (i.rows == null ? 'this few' : i.rows) + ' examples there are too few to hold any back and test honestly (it needs ' + MIN_FOR_HOLDOUT + '). Its match score is a replay of weeks it studied, so treat it as a ceiling.',
        'Safeguards already on: random muting of 10% of its units while learning, and it stops early if it stops improving. A true test switches on at ' + MIN_FOR_HOLDOUT + ' examples.'));
    } else if (ratio > 2.5) {
      c.push(chk('overfit', 1, 'Memorising instead of learning', 'problem',
        'It is ' + r2(ratio) + 'x less accurate on weeks it never studied than on weeks it did. It is memorising.', 'Add more weeks; until then trust the range, not the single number.'));
    } else if (ratio > 1.6) {
      c.push(chk('overfit', 1, 'Memorising instead of learning', 'watch',
        'A little less accurate on weeks it never studied (' + r2(ratio) + 'x). Mild memorising.', 'More weeks will close the gap.'));
    } else {
      c.push(chk('overfit', 1, 'Memorising instead of learning', 'ok',
        'Held-back weeks scored about as well as studied weeks (' + r2(ratio) + 'x). It is learning patterns, not memorising.', ''));
    }

    // 2 underfitting
    if (i.r2 == null) c.push(chk('underfit', 2, 'Not learning enough', 'unknown', 'No match score yet.', ''));
    else if (i.r2 < 0.3) c.push(chk('underfit', 2, 'Not learning enough', 'problem', 'It matches only ' + Math.round(i.r2 * 100) + '% of the ups and downs even in weeks it studied. It is finding almost no pattern.', 'Log more weeks, and fill in saves, shares and posting counts so it has signals to work with.'));
    else if (i.r2 < 0.55) c.push(chk('underfit', 2, 'Not learning enough', 'watch', 'It matches ' + Math.round(i.r2 * 100) + '% of the ups and downs. The pattern is weak.', 'More weeks and more detail per week will help.'));
    else c.push(chk('underfit', 2, 'Not learning enough', 'ok', 'It matches ' + Math.round(i.r2 * 100) + '% of the ups and downs in the weeks it studied.', ''));

    // 3 class imbalance
    if (!i.classCounts || i.classCounts.reduce(function (a, b) { return a + b; }, 0) < 6) {
      c.push(chk('imbalance', 3, 'One content type dominating', 'unknown', 'Not enough logged posts to compare content types yet (needs 6).', 'Log each post with its content type.'));
    } else {
      const total = i.classCounts.reduce(function (a, b) { return a + b; }, 0), top = Math.max.apply(null, i.classCounts), share = top / total;
      c.push(share > SKEW_SHARE
        ? chk('imbalance', 3, 'One content type dominating', 'watch', Math.round(share * 100) + '% of your logged posts are one content type, so the engine could just guess that one and look accurate.', 'It now gives rarer types extra weight while learning. Posting a wider mix is the real fix.')
        : chk('imbalance', 3, 'One content type dominating', 'ok', 'Your logged posts are spread across content types (largest is ' + Math.round(share * 100) + '%). Rare types are also given extra weight while learning.', ''));
    }

    // 4 bad data
    if (!i.dataIssues) c.push(chk('data', 4, 'Typos and bad numbers', 'unknown', 'Not checked.', ''));
    else if (i.dataIssues.issues.length === 0) c.push(chk('data', 4, 'Typos and bad numbers', 'ok', 'Every week passed the checks: no missing numbers, impossible values, repeated weeks or wild outliers.', ''));
    else {
      const first = i.dataIssues.issues[0], n = i.dataIssues.issues.length;
      c.push(chk('data', 4, 'Typos and bad numbers', n > 2 ? 'problem' : 'watch',
        n + ' entr' + (n === 1 ? 'y looks' : 'ies look') + ' wrong. For example week ' + (first.week + 1) + ' (' + first.field + '): ' + first.problem + '.',
        'Weeks with impossible or missing values are left out of learning. Correct them in the data to bring them back.'));
    }

    // 5 leakage
    if (!i.leak) c.push(chk('leak', 5, 'Peeking at the answer', 'unknown', 'Not checked.', ''));
    else c.push(i.leak.ok
      ? chk('leak', 5, 'Peeking at the answer', 'ok', 'Verified: each week is predicted using only that week and earlier ones. Changing the future does not change the prediction. Held-back weeks are always the newest, never shuffled in. (The 0-to-1 rescaling ranges use every entered week, a minor effect.)', '')
      : chk('leak', 5, 'Peeking at the answer', 'problem', 'The prediction for week ' + (i.leak.leaks[0].week + 1) + ' changed when later weeks changed. It can see the future, so its accuracy is not real.', 'Training results should not be trusted until this is fixed.'));

    // 6 bias / variance
    if (!hasHold && i.r2 == null) c.push(chk('biasvar', 6, 'Too simple or too jumpy', 'unknown', 'Not enough information yet.', ''));
    else {
      const high_bias = i.r2 != null && i.r2 < 0.55, high_var = hasHold && ratio > 1.6;
      c.push(high_bias && high_var ? chk('biasvar', 6, 'Too simple or too jumpy', 'problem', 'Both: it misses the pattern and is unstable on new weeks.', 'More data is the only real fix.')
        : high_bias ? chk('biasvar', 6, 'Too simple or too jumpy', 'watch', 'Leaning too simple: it is not capturing enough of the pattern.', 'Add signals (saves, shares, posts per week) and more weeks.')
        : high_var ? chk('biasvar', 6, 'Too simple or too jumpy', 'watch', 'Leaning too jumpy: results swing on weeks it has not seen.', 'More weeks steady it.')
        : chk('biasvar', 6, 'Too simple or too jumpy', hasHold ? 'ok' : 'unknown', hasHold ? 'Balanced: complex enough to follow your results, steady enough on new weeks.' : 'It fits the weeks it studied; steadiness on new weeks cannot be tested yet.', ''));
    }

    // 7 gradients
    if (!i.health) c.push(chk('grad', 7, 'Learning breaking down', 'unknown', 'Not checked.', ''));
    else {
      const lr = lrDiagnosis(i.history);
      if (!i.health.ok || lr.state === 'exploded') c.push(chk('grad', 7, 'Learning breaking down', 'problem', 'The numbers inside the engine blew up (' + (i.health.nonFinite ? i.health.nonFinite + ' invalid values' : 'loss exploded') + '). Its output cannot be trusted.', 'Retrain; if it repeats, the data likely has an extreme value.'));
      else if (i.health.deadShare > 0.5) c.push(chk('grad', 7, 'Learning breaking down', 'watch', Math.round(i.health.deadShare * 100) + '% of its internal switches are switched off, so it is using little of its capacity.', 'Retrain; more varied data helps.'));
      else c.push(chk('grad', 7, 'Learning breaking down', 'ok', 'Its internal numbers stayed healthy throughout learning (ReLU units, careful starting values, and learning is halted if anything goes invalid).', ''));
    }

    // 8 learning speed
    {
      const lr = lrDiagnosis(i.history);
      const map = {
        unknown: ['unknown', 'Too few learning steps to judge.'],
        ok: ['ok', 'Its error fell ' + Math.round((lr.drop || 0) * 100) + '% and settled. The learning speed suited your data.'],
        exploded: ['problem', 'It learned too aggressively and overshot.'],
        diverging: ['problem', 'Its error started rising again; it was learning too fast.'],
        bouncing: ['watch', 'Its error bounced around instead of settling.'],
        stuck: ['watch', 'Its error barely moved; it was learning too slowly or has nothing to learn from.']
      };
      const m = map[lr.state] || map.unknown;
      c.push(chk('lr', 8, 'Learning speed', m[0], m[1], m[0] === 'ok' || m[0] === 'unknown' ? '' : 'Retrain. The engine uses an adaptive learning method (Adam) that adjusts itself, so repeating usually fixes it.'));
    }

    // 9 drift
    if (!i.drift || i.drift.state === 'unknown') c.push(chk('drift', 9, 'Your audience changing', 'unknown', 'Needs 8+ weeks to compare recent weeks with earlier ones.', ''));
    else if (i.drift.state === 'shifted') c.push(chk('drift', 9, 'Your audience changing', 'problem', 'Recent engagement (' + i.drift.now + '%) is well outside the earlier pattern (' + i.drift.before + '%). The engine learned an older audience.', 'Treat this forecast with care and retrain as new weeks arrive.'));
    else if (i.drift.state === 'moving') c.push(chk('drift', 9, 'Your audience changing', 'watch', 'Recent engagement (' + i.drift.now + '%) has moved from earlier weeks (' + i.drift.before + '%).', 'Retrain regularly; the engine refreshes whenever you add weeks.'));
    else c.push(chk('drift', 9, 'Your audience changing', 'ok', 'Recent weeks look like the earlier ones (' + i.drift.now + '% vs ' + i.drift.before + '%). Nothing has shifted.', ''));

    // 10 scaling
    if (!i.scale || i.scale.state === 'unknown') c.push(chk('scale', 10, 'Numbers on different scales', 'unknown', 'Not checked.', ''));
    else if (i.scale.state === 'broken') c.push(chk('scale', 10, 'Numbers on different scales', 'problem', 'Some inputs are not valid numbers.', 'Fix the data entries.'));
    else if (i.scale.state === 'unscaled') c.push(chk('scale', 10, 'Numbers on different scales', 'problem', 'Some inputs are not on a comparable scale, so big numbers (like reach) would drown out small ones (like saves).', 'Report this; scaling should be automatic.'));
    else if (i.scale.state === 'flat') c.push(chk('scale', 10, 'Numbers on different scales', 'watch', i.scale.constant.length + ' input' + (i.scale.constant.length === 1 ? ' never changes' : 's never change') + ' across your weeks, so it teaches the engine nothing.', 'Fill in varied values (posts, saves, paid boosts) per week.'));
    else c.push(chk('scale', 10, 'Numbers on different scales', 'ok', 'Every input is rescaled to the same 0-to-1 range, so reach, followers and saves compete fairly.', ''));

    const problems = c.filter(function (x) { return x.status === 'problem'; }).length;
    const watches = c.filter(function (x) { return x.status === 'watch'; }).length;
    const unknown = c.filter(function (x) { return x.status === 'unknown'; }).length;
    return {
      checks: c, problems: problems, watches: watches, unknown: unknown,
      status: problems ? 'problem' : watches ? 'watch' : 'ok',
      headline: problems ? problems + ' safeguard' + (problems === 1 ? '' : 's') + ' flagged a problem - read before relying on this forecast.'
        : watches ? watches + ' thing' + (watches === 1 ? '' : 's') + ' to keep an eye on.'
        : (10 - unknown) + ' of 10 safeguards passed' + (unknown ? '; ' + unknown + ' need more data to test.' : '.')
    };
  }

  function html(a, esc) {
    esc = esc || function (s) { return String(s).replace(/[&<>"]/g, function (m) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[m]; }); };
    const icon = { ok: '<span style="color:var(--green)">&#10003;</span>', watch: '<span style="color:var(--amber)">&#9888;</span>', problem: '<span style="color:var(--red)">&#10007;</span>', unknown: '<span style="color:var(--text3)">&#9675;</span>' };
    const col = a.status === 'problem' ? 'var(--red)' : a.status === 'watch' ? 'var(--amber)' : 'var(--green)';
    return '<details style="margin-top:10px"' + (a.status === 'ok' ? '' : ' open') + '><summary style="cursor:pointer"><strong>Accuracy safeguards</strong> <span class="card-sub" style="color:' + col + '">' + esc(a.headline) + '</span></summary>' +
      '<div style="margin-top:8px">' + a.checks.map(function (x) {
        return '<div style="display:flex;gap:10px;padding:6px 0;border-top:1px solid var(--border)"><div style="width:16px">' + icon[x.status] + '</div><div><div style="font-weight:600;font-size:13px">' + x.n + '. ' + esc(x.title) + '</div>' +
          '<div class="card-sub">' + esc(x.plain) + '</div>' + (x.fix ? '<div class="card-sub" style="opacity:.8">' + esc(x.fix) + '</div>' : '') + '</div></div>';
      }).join('') + '</div></details>';
  }

  return { MIN_FOR_HOLDOUT: MIN_FOR_HOLDOUT, HOLDOUT_FRACTION: HOLDOUT_FRACTION, timeSplit: timeSplit, classWeights: classWeights,
           validateSeries: validateSeries, leakProbe: leakProbe, weightsHealth: weightsHealth, lrDiagnosis: lrDiagnosis,
           drift: drift, scaling: scaling, audit: audit, html: html };
});
