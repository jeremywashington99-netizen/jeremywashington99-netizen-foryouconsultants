/* ==========================================================================
   SEGMENT 7 - Instant metrics from a single entry

   One post or one week is enough. Everything here is arithmetic on figures
   the owner entered, so the moment a row is saved the full derived set is
   available: no waiting for four weeks of history.

   Every returned metric carries:
     value   the number, or null when the input it needs was not entered
     basis   'measured'  taken straight from what was entered
             'derived'   arithmetic on entered figures only
             'benchmark' compared against a published average (Segment 6)
     needs   which raw field is missing, when value is null

   A null is never shown as 0. "Not tracked" and "zero" are different facts
   and conflating them is how analytics tools end up lying to clients.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./mkt-benchmarks.js'));
  else { root.FYMkt = root.FYMkt || {}; root.FYMkt.metrics = factory(root.FYMkt.bench); }
})(typeof self !== 'undefined' ? self : this, function (bench) {
  'use strict';

  function n(v) { const x = Number(v); return isFinite(x) ? x : null; }
  function pos(v) { const x = n(v); return x != null && x > 0 ? x : null; }
  function rate(part, whole, dp) {
    const p = n(part), w = pos(whole);
    if (p == null || w == null) return null;
    return +((p / w) * 100).toFixed(dp == null ? 2 : dp);
  }
  function round(v, dp) { return v == null ? null : +Number(v).toFixed(dp == null ? 2 : dp); }

  /** Interaction total. Platform definitions differ, so the parts are kept visible. */
  function engagements(e) {
    const parts = ['likes', 'comments', 'shares', 'saves'].map(function (k) { return n(e[k]); });
    if (parts.every(function (p) { return p == null; })) return null;
    return parts.reduce(function (s, p) { return s + (p || 0); }, 0);
  }

  const M = function (value, basis, needs) {
    return { value: value, basis: value == null ? 'missing' : basis, needs: value == null ? (needs || null) : null };
  };

  /**
   * Derive every available metric from one entry.
   * entry: a post record or a week row. Recognised fields:
   *   reach, impressions, followers, likes, comments, shares, saves,
   *   views_3s, completion_rate, avg_watch_seconds, new_follows,
   *   profile_visits, link_clicks, conversions, spend, revenue,
   *   sentiment_pos, sentiment_neg, posts
   * followersAt: follower count at the time, when the entry itself has none
   *   (a post record usually does not carry one).
   */
  function derive(entry, opts) {
    const e = entry || {}, o = opts || {};
    const followers = pos(e.followers) || pos(o.followersAt);
    const reach = pos(e.reach);
    const impressions = pos(e.impressions) || reach;
    const eng = engagements(e);
    const spend = n(e.spend) != null ? n(e.spend) : n(e.paid_spend);
    const clicks = n(e.link_clicks) != null ? n(e.link_clicks) : n(e.clicks);

    const m = {
      engagements:        M(eng, 'derived', 'likes, comments, shares or saves'),
      er_reach:           M(eng != null && reach ? rate(eng, reach) : null, 'derived', 'reach and interaction counts'),
      er_follower:        M(eng != null && followers ? rate(eng, followers) : null, 'derived', 'a follower count for this platform'),
      reach_rate:         M(reach && followers ? rate(reach, followers) : null, 'derived', 'reach and followers'),
      save_rate:          M(rate(e.saves, reach), 'derived', 'saves and reach'),
      share_rate:         M(rate(e.shares, reach), 'derived', 'shares and reach'),
      comment_rate:       M(rate(e.comments, reach), 'derived', 'comments and reach'),
      like_rate:          M(rate(e.likes, reach), 'derived', 'likes and reach'),
      virality_rate:      M(rate(e.shares, impressions), 'derived', 'shares and impressions'),
      hook_rate:          M(rate(e.views_3s, reach), 'derived', '3-second views (field "Views held 3s+")'),
      completion_rate:    M(n(e.completion_rate), 'measured', 'completion rate from the platform'),
      avg_watch_seconds:  M(n(e.avg_watch_seconds), 'measured', 'average watch time'),
      follow_rate:        M(rate(e.new_follows, reach), 'derived', 'new follows and reach'),
      profile_visit_rate: M(rate(e.profile_visits, reach), 'derived', 'profile visits and reach'),
      ctr:                M(rate(clicks, reach), 'derived', 'link clicks and reach'),
      conversion_rate:    M(rate(e.conversions, clicks), 'derived', 'conversions and link clicks'),
      cpm:                M(spend != null && reach ? round((spend / reach) * 1000) : null, 'derived', 'spend and reach'),
      cpe:                M(spend != null && eng ? round(spend / eng, 3) : null, 'derived', 'spend and interaction counts'),
      cpc:                M(spend != null && pos(clicks) ? round(spend / clicks, 3) : null, 'derived', 'spend and link clicks'),
      roas:               M(n(e.revenue) != null && pos(spend) ? round(n(e.revenue) / spend, 2) : null, 'derived', 'revenue and spend'),
      sentiment_ratio:    M((function () {
                            const p = n(e.sentiment_pos), g = n(e.sentiment_neg);
                            if (p == null || g == null || p + g <= 0) return null;
                            return rate(p, p + g);
                          })(), 'derived', 'positive and negative comment counts')
    };

    /* Engagement rate per impression - the definition LinkedIn benchmarks use. */
    m.er_impression = M(eng != null && impressions ? rate(eng, impressions) : null, 'derived', 'impressions and interaction counts');

    const missing = Object.keys(m).filter(function (k) { return m[k].value == null; })
      .map(function (k) { return { key: k, needs: m[k].needs }; });
    const have = Object.keys(m).filter(function (k) { return m[k].value != null; });
    return { metrics: m, have: have, missing: missing, followersUsed: followers, reachUsed: reach };
  }

  /** Flat value lookup, for callers that just want the number. */
  function value(derived, key) {
    const r = derived && derived.metrics && derived.metrics[key];
    return r ? r.value : null;
  }

  /**
   * Instant scorecard for ONE entry: each available metric graded against a
   * published benchmark where one exists, otherwise against the client's own
   * baseline, otherwise left ungraded and labelled as such.
   * The engagement rate is graded on the definition the platform's own
   * benchmark uses, so the comparison is like for like.
   */
  const ER_KEY_FOR = { linkedin: 'er_impression' };
  const GRADEABLE = ['er_follower', 'er_impression', 'er_reach', 'reach_rate'];

  function scorecard(entry, opts) {
    const o = opts || {};
    const platform = o.platform || entry.platform || 'instagram';
    const d = derive(entry, o);
    const erKey = ER_KEY_FOR[platform] || 'er_follower';
    const rows = [];
    GRADEABLE.forEach(function (key) {
      const v = value(d, key);
      if (v == null) return;
      const g = bench.grade(platform, key, v, {
        followers: d.followersUsed, format: o.format,
        baseline: o.baseline && o.baseline[key] != null ? o.baseline[key] : null
      });
      const fw = bench.frameworkEntry(key);
      rows.push({ key: key, label: fw ? fw.label : key, value: v, headline: key === erKey, grade: g });
    });
    /* Rates with no published benchmark still get a baseline comparison when
       the client has one, so they are never left without context. */
    ['save_rate', 'share_rate', 'comment_rate', 'follow_rate', 'ctr', 'hook_rate', 'completion_rate']
      .forEach(function (key) {
        const v = value(d, key);
        if (v == null) return;
        const base = o.baseline && o.baseline[key] != null ? o.baseline[key] : null;
        const fw = bench.frameworkEntry(key);
        rows.push({ key: key, label: fw ? fw.label : key, value: v, headline: false,
                    grade: bench.grade(platform, key, v, { baseline: base, followers: d.followersUsed }) });
      });
    return { platform: platform, derived: d, rows: rows,
             headline: rows.find(function (r) { return r.headline; }) || rows[0] || null,
             missing: d.missing };
  }

  /**
   * The client's own baseline: the median of each rate across their posts.
   * The median, not the mean, because one viral post would otherwise move the
   * yardstick and make every later post look like a failure.
   */
  function median(a) {
    const s = a.slice().sort(function (x, y) { return x - y; });
    if (!s.length) return null;
    const i = Math.floor(s.length / 2);
    return s.length % 2 ? s[i] : +((s[i - 1] + s[i]) / 2).toFixed(4);
  }

  const BASELINE_KEYS = ['er_reach', 'er_follower', 'reach_rate', 'save_rate', 'share_rate',
                         'comment_rate', 'like_rate', 'follow_rate', 'ctr', 'profile_visit_rate',
                         'hook_rate', 'completion_rate'];

  function baseline(entries, opts) {
    const o = opts || {};
    const buckets = {};
    BASELINE_KEYS.forEach(function (k) { buckets[k] = []; });
    (entries || []).forEach(function (e) {
      const d = derive(e, o);
      BASELINE_KEYS.forEach(function (k) {
        const v = value(d, k);
        if (v != null) buckets[k].push(v);
      });
    });
    const out = { n: (entries || []).length, counts: {} };
    BASELINE_KEYS.forEach(function (k) { out[k] = median(buckets[k]); out.counts[k] = buckets[k].length; });
    return out;
  }

  /**
   * Which framework metrics this client can already see, and the single field
   * that would unlock each one they cannot. This is what turns "not enough
   * data" into a concrete instruction.
   */
  function coverage(entries, opts) {
    const list = entries || [];
    const seen = {};
    list.forEach(function (e) {
      const d = derive(e, opts || {});
      d.have.forEach(function (k) { seen[k] = true; });
    });
    return bench.framework().map(function (f) {
      const available = f.needs.length === 0 ? list.length > 0 : !!seen[f.key] ||
        f.needs.every(function (field) { return list.some(function (e) { return e[field] != null && e[field] !== ''; }); });
      return { key: f.key, when: f.when, group: f.group, label: f.label, formula: f.formula,
               why: f.why, available: available, needs: available ? null : f.needs };
    });
  }

  return { derive: derive, value: value, scorecard: scorecard, baseline: baseline,
           coverage: coverage, median: median, engagements: engagements,
           BASELINE_KEYS: BASELINE_KEYS, GRADEABLE: GRADEABLE, ER_KEY_FOR: ER_KEY_FOR };
});
