/* ==========================================================================
   SEGMENT 6 - Published marketing benchmarks + grading

   Why this exists: a client with ONE week of data has nothing to trend
   against. Published platform averages give an outside yardstick, so the
   very first figure entered can be graded immediately instead of waiting
   four weeks for the model to have enough history.

   EVERY number in BENCHMARKS is a published industry average, NOT this
   client's data. Anything derived from it carries basis:'benchmark' so the
   UI can label it. Nothing here is presented as measured.

   Sources (fetched 2026-11-02, figures are the latest published):
     [SI-B]  Socialinsider, Social Media Benchmarks
             https://www.socialinsider.io/social-media-benchmarks
             Engagement rate measured PER FOLLOWER per post.
     [SI-L]  Socialinsider, LinkedIn Benchmarks
             https://www.socialinsider.io/social-media-benchmarks/linkedin
             LinkedIn engagement rate measured PER IMPRESSION.
     [SI-R]  Socialinsider, Social Media Reach Statistics
             https://www.socialinsider.io/blog/social-media-reach/
             Reach rate = total reach / total followers * 100.
     [SG-H]  Scott Graffius, Lifespan (Half-Life) of Social Media Posts, 2026
             https://www.scottgraffius.com/blog/files/lifespan-halflife-of-social-media-posts-update-2026.html
             Half-life = time to reach half of a post's total engagement.
             Based on 5.6M+ posts, data period 2025.
     [HS-M]  Hootsuite, Social media metrics to track
             https://blog.hootsuite.com/social-media-metrics/
             Used for metric definitions and formulas, not for averages.

   Caveat recorded deliberately: the Socialinsider pages label themselves
   2026 while stating the underlying data is 2025, and quote both annual and
   Q3 figures. Where both exist this module keeps the more conservative
   (lower) value as `value` and the other as `alt`, and says so in `note`.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.FYMkt = root.FYMkt || {}; root.FYMkt.bench = factory(); }
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const FETCHED = '2026-11-02';

  /* Follower brackets used by the reach-rate and growth tables. */
  const BRACKETS = [
    { max: 5000, key: '1-5K' },
    { max: 10000, key: '5-10K' },
    { max: 50000, key: '10-50K' },
    { max: 100000, key: '50-100K' },
    { max: Infinity, key: '100K-1M' }
  ];
  function bracketOf(followers) {
    const f = Number(followers) || 0;
    for (const b of BRACKETS) if (f <= b.max) return b.key;
    return '100K-1M';
  }

  /* --------------------------------------------------------------------
     BENCHMARKS[platform][metric] = {
       value, unit, basis:'benchmark', source, defn, note?, alt?,
       bySize?: { bracket: value }, byFormat?: { format: value }
     }
     A missing entry is NOT an error: grade() falls back to the client's own
     baseline and says so. Never invent a figure to fill a gap.
     -------------------------------------------------------------------- */
  const PCT = '%';
  const BENCHMARKS = {
    tiktok: {
      er_follower: { value: 2.50, alt: 3.70, unit: PCT, source: 'SI-B',
        defn: '(likes + comments + shares + saves) / followers x 100',
        note: 'Q3 2026 quarterly average; the 2025 annual average was 3.70%.' },
      posts_per_week: { value: 3.5, alt: 5, unit: 'posts', source: 'SI-B',
        defn: 'posts published per week',
        note: 'From ~15 posts/month. The same report also states 5/week; the monthly figure is used here.' }
    },
    instagram: {
      er_follower: { value: 0.45, alt: 0.48, unit: PCT, source: 'SI-B',
        defn: '(likes + comments) / followers x 100',
        note: 'Q3 2026 quarterly average; the 2025 annual average was 0.48%. Instagram counts likes and comments only.' },
      er_reach: { value: 3.20, unit: PCT, source: 'SI-R',
        defn: 'engagements / reach x 100' },
      reach_rate: { value: 4.90, unit: PCT, source: 'SI-R',
        defn: 'reach / followers x 100',
        note: 'Down about 14% year over year. Reach is an estimated metric on Instagram.',
        bySize: { '1-5K': 6.65, '5-10K': 5.75, '10-50K': 5.50, '50-100K': 4.50, '100K-1M': 3.50 },
        byFormat: { carousel: 4.50, reel: 4.10, image: 4.00 } },
      posts_per_week: { value: 4.6, alt: 5, unit: 'posts', source: 'SI-B',
        defn: 'posts published per week', note: 'From ~20 posts/month.' }
    },
    facebook: {
      er_follower: { value: 0.13, alt: 0.15, unit: PCT, source: 'SI-B',
        defn: '(reactions + comments + shares) / followers x 100' },
      er_reach: { value: 1.25, unit: PCT, source: 'SI-R', defn: 'engagements / reach x 100' },
      reach_rate: { value: 1.80, unit: PCT, source: 'SI-R', defn: 'reach / followers x 100',
        bySize: { '1-5K': 4.35, '5-10K': 3.40, '10-50K': 2.70, '50-100K': 1.50, '100K-1M': 0.50 },
        byFormat: { image: 1.20, album: 1.15, status: 0.80, reel: 0.55, link: 0.20 } },
      posts_per_week: { value: 5.5, unit: 'posts', source: 'SI-B', defn: 'posts published per week',
        note: 'From ~24 posts/month.' }
    },
    x: {
      er_follower: { value: 0.10, alt: 0.12, unit: PCT, source: 'SI-B',
        defn: '(likes + reposts) / followers x 100' },
      posts_per_week: { value: 16, unit: 'posts', source: 'SI-B', defn: 'posts published per week',
        note: 'From ~70 posts/month.' }
    },
    linkedin: {
      /* LinkedIn is the exception: the published rate is per IMPRESSION, not
         per follower. Grading a per-follower figure against it would be wrong,
         so the key is named for what it actually measures. */
      er_impression: { value: 5.20, unit: PCT, source: 'SI-L',
        defn: '(reactions + comments + shares) / impressions x 100',
        note: '2025 average. Q3 2026 format figures run lower.',
        byFormat: { document: 7.00, multi_image: 6.45, video: 6.00, image: 5.30, text: 4.50, poll: 4.20, link: 3.25 } },
      follower_growth_annual: { value: 24.50, unit: PCT, source: 'SI-L',
        defn: 'followers gained in the year / followers at the start x 100',
        note: 'LinkedIn company pages only. Published per page-size bracket.',
        bySize: { '1-5K': 24.50, '5-10K': 31.00, '10-50K': 21.30, '50-100K': 16.25, '100K-1M': 6.40 } },
      posts_per_week: { value: 4, unit: 'posts', source: 'SI-L',
        defn: 'posts published per week', note: 'From the 2025 per-format monthly counts (~17/month all formats).' }
    }
  };

  /* Half-life: minutes to collect half of a post's lifetime engagement. [SG-H] */
  const HALF_LIFE_MIN = {
    instagram: 1096,
    facebook: 86,
    x: 52,
    linkedin: 1393,
    youtube: 15276,
    pinterest: 174894,
    tiktok: null   /* published as ~0 minutes, which cannot drive a decay curve. */
  };
  const HALF_LIFE_NOTE = {
    tiktok: 'No usable published half-life for TikTok: the source reports roughly 0 minutes, ' +
            'meaning engagement arrives immediately, while TikTok is also known for long tails. ' +
            'Log the same post a second time and the app measures its actual decay instead of modelling it.',
    instagram: 'Instagram feed and Reels are reported together, not separately.'
  };

  const SOURCES = {
    'SI-B': { name: 'Socialinsider - Social Media Benchmarks', url: 'https://www.socialinsider.io/social-media-benchmarks' },
    'SI-L': { name: 'Socialinsider - LinkedIn Benchmarks', url: 'https://www.socialinsider.io/social-media-benchmarks/linkedin' },
    'SI-R': { name: 'Socialinsider - Social Media Reach Statistics', url: 'https://www.socialinsider.io/blog/social-media-reach/' },
    'SG-H': { name: 'Scott Graffius - Lifespan (Half-Life) of Social Media Posts, 2026', url: 'https://www.scottgraffius.com/blog/files/lifespan-halflife-of-social-media-posts-update-2026.html' },
    'HS-M': { name: 'Hootsuite - Social media metrics to track', url: 'https://blog.hootsuite.com/social-media-metrics/' }
  };

  /** Look up one benchmark, narrowed by account size and format where published. */
  function get(platform, metric, opts) {
    const o = opts || {};
    const p = BENCHMARKS[platform];
    const b = p && p[metric];
    if (!b) return null;
    let value = b.value, how = 'platform average';
    if (b.bySize && o.followers != null) {
      const k = bracketOf(o.followers);
      if (b.bySize[k] != null) { value = b.bySize[k]; how = 'average for ' + k + ' followers'; }
    }
    if (b.byFormat && o.format && b.byFormat[o.format] != null) {
      value = b.byFormat[o.format]; how = 'average for ' + String(o.format).replace(/_/g, ' ') + ' posts';
    }
    return { metric: metric, platform: platform, value: value, unit: b.unit, basis: 'benchmark',
             how: how, defn: b.defn, note: b.note || null, alt: b.alt != null ? b.alt : null,
             source: b.source, sourceName: SOURCES[b.source] ? SOURCES[b.source].name : b.source,
             sourceUrl: SOURCES[b.source] ? SOURCES[b.source].url : null, fetched: FETCHED };
  }

  /* Bands as a multiple of the benchmark. Deliberately wide: a single post
     varies enormously, so narrow bands would read as false precision. */
  const BANDS = [
    { upTo: 0.50, band: 'well below',  tone: 'bad'  },
    { upTo: 0.85, band: 'below',       tone: 'warn' },
    { upTo: 1.25, band: 'typical',     tone: 'ok'   },
    { upTo: 2.00, band: 'strong',      tone: 'good' },
    { upTo: Infinity, band: 'exceptional', tone: 'great' }
  ];

  /**
   * Grade one value. Falls back to the client's own baseline when no
   * benchmark is published, and says which yardstick was used.
   * baseline: optional number - this client's own median for the metric.
   */
  function grade(platform, metric, value, opts) {
    const o = opts || {};
    const v = Number(value);
    if (!isFinite(v)) return { ok: false, text: 'no value to grade' };
    const b = get(platform, metric, o);
    const ref = b ? b.value : (o.baseline != null && isFinite(o.baseline) && o.baseline > 0 ? Number(o.baseline) : null);
    if (ref == null || ref <= 0) {
      return { ok: false, value: v, against: 'none', basis: 'measured',
               text: 'No published benchmark is loaded for this metric and there is no baseline yet, so this figure stands on its own.' };
    }
    const ratio = v / ref;
    const row = BANDS.find(function (x) { return ratio <= x.upTo; });
    const against = b ? 'benchmark' : 'own baseline';
    const pct = Math.round((ratio - 1) * 100);
    const diff = pct === 0 ? 'in line with' : (pct > 0 ? '+' + pct + '% vs' : pct + '% vs');
    return {
      ok: true, value: v, ref: ref, ratio: +ratio.toFixed(3), band: row.band, tone: row.tone,
      against: against, basis: b ? 'benchmark' : 'measured',
      how: b ? b.how : 'this client\'s own median so far',
      source: b ? b.source : null, sourceName: b ? b.sourceName : null, sourceUrl: b ? b.sourceUrl : null,
      defn: b ? b.defn : null, note: b ? b.note : null,
      text: row.band + ' - ' + diff + ' the ' + (b ? b.how : 'client\'s own median') + ' of ' + ref + (b && b.unit === PCT ? '%' : '')
    };
  }

  /** Half-life in minutes, with the caveat text when there is one. */
  function halfLife(platform) {
    const m = HALF_LIFE_MIN[platform];
    return { platform: platform, minutes: m == null ? null : m,
             hours: m == null ? null : +(m / 60).toFixed(2),
             basis: m == null ? 'unavailable' : 'benchmark',
             note: HALF_LIFE_NOTE[platform] || null,
             source: 'SG-H', sourceName: SOURCES['SG-H'].name, sourceUrl: SOURCES['SG-H'].url,
             defn: 'time to collect half of a post\'s lifetime engagement' };
  }

  /* --------------------------------------------------------------------
     THE MARKETING FRAMEWORK
     Every metric a social marketer should be watching, in the three time
     frames the client asked for. `needs` lists the raw fields required, so
     the UI can say exactly which one field unlocks a missing metric instead
     of a vague "not enough data".
     -------------------------------------------------------------------- */
  const FRAMEWORK = [
    /* ---------- PAST: what already happened ---------- */
    { key: 'er_reach', when: 'past', group: 'Engagement', label: 'Engagement rate by reach',
      formula: '(likes + comments + shares + saves) / reach x 100', needs: ['reach'],
      why: 'The fairest read of content quality: of the people who saw it, how many acted.' },
    { key: 'er_follower', when: 'past', group: 'Engagement', label: 'Engagement rate by follower',
      formula: '(likes + comments + shares + saves) / followers x 100', needs: ['followers'],
      why: 'The figure every published benchmark uses, so it is the one to quote to a client.' },
    { key: 'reach_rate', when: 'past', group: 'Distribution', label: 'Reach rate',
      formula: 'reach / followers x 100', needs: ['reach', 'followers'],
      why: 'How much of the audience the platform actually served. Falling reach rate is the earliest warning sign there is.' },
    { key: 'save_rate', when: 'past', group: 'Engagement', label: 'Save rate',
      formula: 'saves / reach x 100', needs: ['saves', 'reach'],
      why: 'Saves signal lasting value and are weighted heavily by Instagram and TikTok ranking.' },
    { key: 'share_rate', when: 'past', group: 'Amplification', label: 'Share rate',
      formula: 'shares / reach x 100', needs: ['shares', 'reach'],
      why: 'Shares are how a post escapes the existing audience. The single best predictor of reach beyond followers.' },
    { key: 'comment_rate', when: 'past', group: 'Engagement', label: 'Comment rate',
      formula: 'comments / reach x 100', needs: ['comments', 'reach'],
      why: 'The most effortful action, so it reads intent rather than habit.' },
    { key: 'virality_rate', when: 'past', group: 'Amplification', label: 'Virality rate',
      formula: 'shares / impressions x 100', needs: ['shares', 'impressions'],
      why: 'Share of total delivery that came from people passing it on.' },
    { key: 'hook_rate', when: 'past', group: 'Video', label: 'Hook rate',
      formula: 'views of 3s or more / views x 100', needs: ['views_3s', 'reach'],
      why: 'Decides whether the rest of the video ever gets seen. The first thing to fix on a weak video.' },
    { key: 'completion_rate', when: 'past', group: 'Video', label: 'Completion rate',
      formula: 'completed views / views x 100', needs: ['completion_rate'],
      why: 'Watch-through is the strongest ranking input on TikTok and Reels.' },
    { key: 'follow_rate', when: 'past', group: 'Growth', label: 'Follower conversion',
      formula: 'new follows / reach x 100', needs: ['new_follows', 'reach'],
      why: 'Separates content that entertains from content that grows the account.' },
    { key: 'profile_visit_rate', when: 'past', group: 'Funnel', label: 'Profile visit rate',
      formula: 'profile visits / reach x 100', needs: ['profile_visits', 'reach'],
      why: 'The bridge between seeing a post and considering the business.' },
    { key: 'ctr', when: 'past', group: 'Funnel', label: 'Link click-through rate',
      formula: 'link clicks / reach x 100', needs: ['link_clicks', 'reach'],
      why: 'The first metric that connects social activity to anything off-platform.' },
    { key: 'conversion_rate', when: 'past', group: 'Funnel', label: 'Conversion rate',
      formula: 'conversions / link clicks x 100', needs: ['conversions', 'link_clicks'],
      why: 'Where social stops being awareness and starts being revenue.' },
    { key: 'cpm', when: 'past', group: 'Paid', label: 'CPM', formula: 'spend / reach x 1000',
      needs: ['spend', 'reach'], why: 'The price of attention, and the number to negotiate on.' },
    { key: 'cpe', when: 'past', group: 'Paid', label: 'Cost per engagement',
      formula: 'spend / engagements', needs: ['spend'],
      why: 'Compares paid efficiency across very different creative.' },
    { key: 'roas', when: 'past', group: 'Paid', label: 'ROAS', formula: 'revenue / spend',
      needs: ['revenue', 'spend'], why: 'The only paid figure a business owner really asks about.' },
    { key: 'sentiment_ratio', when: 'past', group: 'Brand', label: 'Sentiment split',
      formula: 'positive / (positive + negative) x 100', needs: ['sentiment_pos', 'sentiment_neg'],
      why: 'Volume without sentiment hides a crisis; a spike can be the bad kind.' },
    { key: 'posting_cadence', when: 'past', group: 'Operations', label: 'Posting cadence',
      formula: 'posts published per week', needs: ['posts'],
      why: 'Most underperforming accounts are not bad at content, they are inconsistent.' },
    { key: 'content_mix', when: 'past', group: 'Content', label: 'Performance by content type',
      formula: 'median engagement rate grouped by content type', needs: ['cls'],
      why: 'Tells you what to make more of, which is the decision the client actually has to take.' },
    { key: 'timing', when: 'past', group: 'Operations', label: 'Day and hour performance',
      formula: 'median engagement rate grouped by day of week and hour', needs: ['posted_at'],
      why: 'Cheapest available lift: same content, better slot.' },

    /* ---------- PRESENT: what is happening right now ---------- */
    { key: 'lifecycle_stage', when: 'present', group: 'Live posts', label: 'Where a post is in its life',
      formula: 'age vs published half-life, or measured from two readings', needs: ['posted_at'],
      why: 'Stops you judging a post before it has finished, which is the most common reporting mistake.' },
    { key: 'collected_share', when: 'present', group: 'Live posts', label: 'Share of lifetime engagement collected',
      formula: '1 - 0.5 ^ (age / half-life)', needs: ['posted_at'],
      why: 'Turns "it has 400 likes" into "it has 400 of an expected 520".' },
    { key: 'velocity', when: 'present', group: 'Live posts', label: 'Engagement velocity',
      formula: 'engagements / hours since posting', needs: ['posted_at'],
      why: 'Early velocity is what the ranking system reacts to, so it is the first signal a post will travel.' },
    { key: 'pace_vs_cadence', when: 'present', group: 'Operations', label: 'Pace against target cadence',
      formula: 'posts in the last 7 days vs target', needs: ['posted_at'],
      why: 'Catches a slipping schedule while there is still time in the week to fix it.' },
    { key: 'period_change', when: 'present', group: 'Trend', label: 'Latest period vs previous',
      formula: 'latest week / previous week - 1', needs: ['reach'],
      why: 'The client\'s real question is "better or worse than last week".' },
    { key: 'anomaly', when: 'present', group: 'Trend', label: 'Outlier check',
      formula: '(value - median) / median absolute deviation', needs: ['reach'],
      why: 'Separates a genuine change from normal week-to-week noise.' },

    /* ---------- FUTURE: what to expect ---------- */
    { key: 'lifetime_estimate', when: 'future', group: 'Live posts', label: 'Expected lifetime total',
      formula: 'current engagements / share of life collected', needs: ['posted_at'],
      why: 'Lets you report a post\'s likely final numbers on the day it goes out.' },
    { key: 'next_period', when: 'future', group: 'Forecast', label: 'Next period forecast',
      formula: 'benchmark anchor, then trend, then the trained model as history grows', needs: [],
      why: 'A forecast available from day one, honest about which method produced it.' },
    { key: 'similar_post', when: 'future', group: 'Forecast', label: 'Forecast for the next post like this one',
      formula: 'median of comparable posts by type and slot, widened by their spread', needs: ['cls'],
      why: 'Sets an expectation before publishing, so success is defined in advance.' },
    { key: 'growth_runway', when: 'future', group: 'Growth', label: 'Time to a follower milestone',
      formula: 'followers remaining / recent net weekly growth', needs: ['followers'],
      why: 'Converts growth into a date, which is what a client plans around.' },
    { key: 'cadence_target', when: 'future', group: 'Operations', label: 'Recommended cadence and slot',
      formula: 'benchmark cadence, adjusted to the client\'s own best day and hour', needs: [],
      why: 'The concrete instruction for next week.' },
    { key: 'scenario', when: 'future', group: 'Planning', label: 'Scenario: more posts per week',
      formula: 'median reach per post x planned posts, with the observed range', needs: ['reach'],
      why: 'Answers "what if we post five times instead of three" with the client\'s own numbers.' },
    { key: 'risk_flags', when: 'future', group: 'Risk', label: 'Early warnings',
      formula: 'falling reach rate, falling save rate, cadence gaps, single-platform dependence', needs: [],
      why: 'Reach rate falls weeks before followers do, so this is where intervention is still cheap.' }
  ];

  function framework(when) { return when ? FRAMEWORK.filter(function (f) { return f.when === when; }) : FRAMEWORK.slice(); }
  function frameworkEntry(key) { return FRAMEWORK.find(function (f) { return f.key === key; }) || null; }

  return { FETCHED: FETCHED, BENCHMARKS: BENCHMARKS, SOURCES: SOURCES, BANDS: BANDS,
           HALF_LIFE_MIN: HALF_LIFE_MIN, BRACKETS: BRACKETS, FRAMEWORK: FRAMEWORK,
           bracketOf: bracketOf, get: get, grade: grade, halfLife: halfLife,
           framework: framework, frameworkEntry: frameworkEntry };
});
