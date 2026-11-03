/* ==========================================================================
   SEGMENT 8 - Past, present and future for a post, and tiered forecasting

   Two questions this answers:

   1. "Look at an old post: how did it do, how is it doing now, how will it do?"
      A post keeps collecting engagement after it is published. Given when it
      went out and what it has now, the share of its lifetime already
      collected can be estimated, and from that its likely final total.
      Two ways, in order of preference:
        MEASURED  two readings of the same post at different times - the
                  decay is fitted from the client's own data. Always preferred.
        MODELLED  one reading plus the published half-life for the platform.
                  Clearly labelled, because the curve is an assumption.

   2. "Give me a forecast from the first entry, not after four weeks."
      forecast() always returns a number and always names its method:
        benchmark  no history at all - the published platform average
        anchored   one observation, pulled toward the benchmark
        trend      two or three observations - damped linear trend
        model      four or more - hand over to the trained network
                   (Segments 1-4); a statistical value is still returned so
                   the screen is never empty while the model trains.

   Saturating growth model used throughout:
       y(t) = L * (1 - 2^(-t/h))
   t = age, h = half-life, L = lifetime total. At t = h, y = L/2, which is
   the definition of half-life the published figures use.
   ========================================================================== */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./mkt-benchmarks.js'), require('./mkt-metrics.js'));
  else { root.FYMkt = root.FYMkt || {}; root.FYMkt.life = factory(root.FYMkt.bench, root.FYMkt.metrics); }
})(typeof self !== 'undefined' ? self : this, function (bench, met) {
  'use strict';

  const MIN_PER_HOUR = 60;
  /* Past six half-lives a post has collected 98.4% of its engagement. Treating
     it as finished past that point avoids dividing by a share near 1. */
  const CLOSED_AT_HALF_LIVES = 6;

  function num(v) { const x = Number(v); return isFinite(x) ? x : null; }
  function toTime(v) {
    if (v == null || v === '') return null;
    const t = v instanceof Date ? v.getTime() : Date.parse(v);
    return isFinite(t) ? t : null;
  }
  function collectedShare(ageMin, halfLifeMin) {
    if (!(halfLifeMin > 0) || !(ageMin >= 0)) return null;
    return 1 - Math.pow(2, -ageMin / halfLifeMin);
  }

  /**
   * Fit the half-life from two cumulative readings of the same post.
   * y1/y2 = (1 - 2^(-t1/h)) / (1 - 2^(-t2/h)) falls monotonically from 1
   * (h -> 0) to t1/t2 (h -> infinity), so one bisection finds h.
   * Returns { halfLifeMin, lifetime, basis:'measured' } or a reason it could not.
   */
  function fitFromReadings(r1, r2) {
    const t1 = num(r1.ageMin), t2 = num(r2.ageMin);
    const y1 = num(r1.value), y2 = num(r2.value);
    if (t1 == null || t2 == null || y1 == null || y2 == null) return { ok: false, why: 'both readings need an age and a value' };
    if (!(t2 > t1 && t1 >= 0)) return { ok: false, why: 'the second reading must be later than the first' };
    if (!(y2 > 0)) return { ok: false, why: 'the later reading has no engagement to work from' };
    if (y2 <= y1) {
      /* Nothing new arrived between the readings: the post has stopped. */
      return { ok: true, halfLifeMin: null, lifetime: y2, collected: 1, basis: 'measured',
               note: 'No new engagement between the two readings, so the post is treated as finished at ' + y2 + '.' };
    }
    const r = y1 / y2, floor = t1 / t2;
    if (r <= floor) {
      return { ok: false, why: 'growth between the readings was faster than it was before them, so a settling curve does not fit. ' +
                               'That usually means the post was still being pushed out. Read it again later.' };
    }
    let lo = 1e-4, hi = 1e8;
    for (let i = 0; i < 200; i++) {
      const h = Math.sqrt(lo * hi);
      const f = collectedShare(t1, h) / collectedShare(t2, h);
      if (f > r) lo = h; else hi = h;
    }
    const h = Math.sqrt(lo * hi);
    const share = collectedShare(t2, h);
    return { ok: true, halfLifeMin: +h.toFixed(2), lifetime: Math.round(y2 / share),
             collected: +share.toFixed(4), basis: 'measured',
             note: 'Half-life of ' + (h / MIN_PER_HOUR).toFixed(1) + ' hours measured from your two readings of this post.' };
  }

  /**
   * Where one post stands now.
   * post: { platform, posted_at, measured_at?, likes, comments, shares, saves, reach }
   * opts: { now?, readings?: [{at, likes...}|{at, value}] }
   */
  function lifecycle(post, opts) {
    const o = opts || {}, p = post || {};
    const now = toTime(o.now) || Date.now();
    const posted = toTime(p.posted_at);
    const platform = p.platform || 'instagram';
    const eng = met.engagements(p);
    const hl = bench.halfLife(platform);

    const out = { platform: platform, engagements: eng, now: new Date(now).toISOString(),
                  halfLife: hl, basis: null, stage: null, collected: null, lifetime: null,
                  remaining: null, velocity: null, ageHours: null, notes: [] };

    if (posted == null) {
      out.notes.push('No posting time recorded for this post, so its age is unknown. Add "Posted on" and the lifecycle fills in.');
      return out;
    }
    /* Figures were read off the platform at measured_at if given, else assume now. */
    const measured = toTime(p.measured_at) || now;
    const ageMin = Math.max(0, (measured - posted) / 60000);
    out.ageHours = +(ageMin / MIN_PER_HOUR).toFixed(2);
    out.ageHoursNow = +(Math.max(0, (now - posted) / 60000) / MIN_PER_HOUR).toFixed(2);
    if (eng != null && ageMin > 0) out.velocity = +(eng / (ageMin / MIN_PER_HOUR)).toFixed(2);

    /* --- preferred route: the client's own two readings --- */
    const readings = (o.readings || []).map(function (r) {
      const at = toTime(r.at != null ? r.at : r.measured_at);
      const value = r.value != null ? num(r.value) : met.engagements(r);
      return at == null || value == null ? null : { ageMin: Math.max(0, (at - posted) / 60000), value: value };
    }).filter(Boolean).sort(function (a, b) { return a.ageMin - b.ageMin; });

    if (readings.length >= 2) {
      const fit = fitFromReadings(readings[0], readings[readings.length - 1]);
      if (fit.ok) {
        out.basis = 'measured';
        out.halfLifeUsedMin = fit.halfLifeMin;
        out.collected = fit.collected;
        out.lifetime = fit.lifetime;
        out.notes.push(fit.note);
      } else {
        out.notes.push('Could not measure the decay: ' + fit.why);
      }
    }

    /* --- fallback: the published half-life --- */
    if (out.basis == null) {
      if (hl.minutes == null) {
        out.basis = 'unavailable';
        /* No curve is invented. But age is still a fact worth stating: a post
           weeks old has almost certainly stopped collecting, so what was
           entered is reported as a floor for its lifetime, not an estimate. */
        const ageDays = ageMin / 1440;
        out.ageDays = +ageDays.toFixed(1);
        out.lifetimeFloor = eng;
        out.stage = ageDays >= 14 ? 'likely finished' : ageDays >= 2 ? 'past its first days' : 'early';
        out.notes.push(hl.note || 'No published half-life for this platform.');
        if (ageDays >= 14 && eng != null) {
          out.notes.push('At ' + out.ageDays + ' days old this post has very likely stopped collecting, so the ' + eng +
                         ' interactions entered are the best available figure for its lifetime. Nothing has been extrapolated.');
        } else {
          out.notes.push('Log this post again in a day or two with its new figures and the app will measure its actual curve.');
        }
        return out;
      }
      const share = collectedShare(ageMin, hl.minutes);
      out.basis = 'modelled';
      out.halfLifeUsedMin = hl.minutes;
      out.collected = +share.toFixed(4);
      out.lifetime = eng != null && share > 0.02 ? Math.round(eng / share) : null;
      out.notes.push('Estimated from the published half-life of ' + hl.hours + ' hours for ' + platform +
                     ', not measured on this post. Log it a second time to replace this with a measured figure.');
      if (hl.note) out.notes.push(hl.note);
    }

    if (out.collected != null && eng != null && out.lifetime != null) out.remaining = Math.max(0, out.lifetime - eng);
    const hlUsed = out.halfLifeUsedMin;
    if (hlUsed > 0) {
      const lives = ageMin / hlUsed;
      out.halfLivesElapsed = +lives.toFixed(2);
      out.stage = lives < 0.5 ? 'surging' : lives < 1 ? 'climbing' : lives < 3 ? 'settling'
                : lives < CLOSED_AT_HALF_LIVES ? 'tail' : 'finished';
    } else if (out.collected === 1) {
      out.stage = 'finished';
    }
    return out;
  }

  /* ------------------------------------------------------------------
     Tiered forecasting
     ------------------------------------------------------------------ */

  /* Indicative spread by method. Stated as a plain range, never as a
     statistical confidence interval, because with one or two observations
     there is no distribution to be confident about. */
  const SPREAD = { benchmark: 0.60, anchored: 0.50, trend: 0.35, model: 0.25 };
  /* One observation is weak evidence, so it is pulled halfway to the
     published average rather than trusted on its own. */
  const ANCHOR_WEIGHT = 0.5;

  function mean(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }

  /**
   * Forecast the next value of one metric.
   * values: chronological numbers (oldest first). May be empty.
   * opts: { platform, metric, followers, format, modelValue }
   *   modelValue - the trained network's prediction, used when there are
   *   four or more observations.
   */
  function forecast(values, opts) {
    const o = opts || {};
    const metric = o.metric || 'er_follower';
    const vals = (values || []).map(num).filter(function (v) { return v != null; });
    const b = bench.get(o.platform || 'instagram', metric, { followers: o.followers, format: o.format });
    const out = { metric: metric, n: vals.length, method: null, basis: null, value: null,
                  low: null, high: null, confidence: null, why: null, benchmark: b };

    function finish(method, value, extraSpread, spreadTier) {
      const s = SPREAD[spreadTier || method] + (extraSpread || 0);
      out.method = method;
      out.value = +Number(value).toFixed(2);
      out.low = +Math.max(0, value * (1 - s)).toFixed(2);
      out.high = +(value * (1 + s)).toFixed(2);
      out.spread = +s.toFixed(2);
      out.basis = method === 'benchmark' ? 'benchmark' : (method === 'anchored' ? 'benchmark+measured' : 'measured');
      /* The confidence line must never claim a trained model when none ran. */
      out.confidence = method === 'benchmark' ? 'none - no data for this client yet'
        : method === 'anchored' ? 'low - one observation'
        : method === 'model' ? 'higher - ' + vals.length + ' observations and a trained model'
        : 'moderate - ' + vals.length + ' observation' + (vals.length === 1 ? '' : 's') +
          (vals.length >= 4 ? ', statistical trend only - the model has not been trained yet' : '');
      return out;
    }

    if (!vals.length) {
      if (!b) { out.why = 'No data for this client and no published benchmark for this metric, so no forecast can be made.'; return out; }
      out.why = 'Nothing logged yet, so this is the published ' + b.how + ' for ' + (o.platform || 'instagram') +
                ' (' + b.sourceName + '). It will switch to this client\'s own figures with the first entry.';
      return finish('benchmark', b.value);
    }

    if (vals.length === 1) {
      const seen = vals[0];
      if (!b) { out.why = 'One observation and no published benchmark, so the single figure is carried forward unchanged.'; return finish('anchored', seen, 0.10); }
      const v = ANCHOR_WEIGHT * seen + (1 - ANCHOR_WEIGHT) * b.value;
      out.why = 'One entry (' + seen + ') is weak evidence on its own, so it is averaged with the published ' +
                b.how + ' of ' + b.value + '. Expect the figure to move as more weeks are added.';
      return finish('anchored', v);
    }

    /* Damped linear trend. The last step is halved because short series
       over-extrapolate: two good weeks are not a trajectory. */
    const recent = vals.slice(-4);
    const last = recent[recent.length - 1];
    const step = (recent[recent.length - 1] - recent[0]) / (recent.length - 1);
    const trended = Math.max(0, last + step * 0.5);
    const avg = mean(recent);
    /* Pull toward the recent average so one spike does not set the forecast. */
    const v = 0.7 * trended + 0.3 * avg;
    const cv = Math.abs(avg) > 0 ? Math.sqrt(mean(recent.map(function (x) { return (x - avg) * (x - avg); }))) / Math.abs(avg) : 0;

    if (vals.length >= 4 && o.modelValue != null && isFinite(o.modelValue)) {
      out.why = 'Enough history for the trained network, so its prediction is used. The statistical trend for comparison is ' +
                trended.toFixed(2) + '.';
      out.statistical = +v.toFixed(2);
      return finish('model', o.modelValue, Math.min(0.2, cv * 0.5));
    }
    if (vals.length >= 4) {
      /* Enough history for the network, but it has not been trained in this
         session, so the figure stays a statistical trend and says so. */
      out.why = 'Four or more entries, so this is a damped trend on the last ' + recent.length +
                '. Train the model on the AI screen for a learned forecast.';
      out.modelReady = true;
      return finish('trend', v, Math.min(0.2, cv * 0.5), 'model');
    }
    out.why = vals.length + ' entries, so this is a damped trend: the recent direction, halved, blended with the recent average. ' +
              'It firms up at four entries.';
    return finish('trend', v, Math.min(0.25, cv * 0.5));
  }

  /**
   * Forecast for the next post like a given one: the median of comparable
   * posts, widened by how much they actually varied.
   * Comparable = same platform, then same content class, then same day part.
   */
  function similarPostForecast(post, posts, opts) {
    const o = opts || {};
    const p = post || {};
    const all = (posts || []).filter(function (x) { return x && x.platform === p.platform && x.id !== p.id; });
    const sameClass = all.filter(function (x) { return x.cls === p.cls; });
    const pool = sameClass.length >= 3 ? sameClass : all;
    const basisText = sameClass.length >= 3
      ? sameClass.length + ' earlier ' + (o.className || 'same-type') + ' posts'
      : all.length + ' earlier posts on this platform';
    if (!pool.length) {
      return { ok: false, why: 'No other posts logged on this platform yet, so there is nothing to compare a future post with.' };
    }
    const ers = pool.map(function (x) { return Number(x.er); }).filter(isFinite);
    const reaches = pool.map(function (x) { return Number(x.reach); }).filter(isFinite);
    const medER = met.median(ers), medReach = met.median(reaches);
    const lo = ers.length ? Math.min.apply(null, ers) : null;
    const hi = ers.length ? Math.max.apply(null, ers) : null;
    return { ok: true, basis: 'measured', n: pool.length, why: 'Median of ' + basisText + '.',
             er: medER, erLow: lo, erHigh: hi, reach: medReach == null ? null : Math.round(medReach),
             note: pool.length < 3 ? 'Only ' + pool.length + ' post(s) to compare, so treat this as a rough marker.' : null };
  }

  /**
   * The full past / present / future read on one post.
   * ctx: { posts, followersAt, baseline, now, readings, className }
   */
  function postReport(post, ctx) {
    const c = ctx || {}, p = post || {};
    const card = met.scorecard(p, { platform: p.platform, followersAt: c.followersAt, baseline: c.baseline, format: c.format });
    const life = lifecycle(p, { now: c.now, readings: c.readings });
    const peers = (c.posts || []).filter(function (x) { return x && x.platform === p.platform && x.id !== p.id; });

    /* How it ranked among everything logged on the platform at the time. */
    let rank = null;
    const erAll = peers.map(function (x) { return Number(x.er); }).filter(isFinite);
    if (erAll.length && isFinite(Number(p.er))) {
      const below = erAll.filter(function (v) { return v < Number(p.er); }).length;
      rank = { of: erAll.length + 1, position: erAll.length + 1 - below,
               percentile: Math.round((below / erAll.length) * 100),
               medianPeer: met.median(erAll) };
    }

    return {
      post: { id: p.id, title: p.title, platform: p.platform, posted_at: p.posted_at || null },
      past: { scorecard: card, rank: rank,
              summary: card.headline
                ? card.headline.label + ' of ' + card.headline.value + '% - ' + (card.headline.grade.text || 'no yardstick available')
                : 'No rate could be worked out from what was entered.' },
      present: life,
      future: {
        lifetime: life.lifetime, remaining: life.remaining, basis: life.basis,
        similar: similarPostForecast(p, c.posts, { className: c.className })
      }
    };
  }

  return { lifecycle: lifecycle, fitFromReadings: fitFromReadings, collectedShare: collectedShare,
           forecast: forecast, similarPostForecast: similarPostForecast, postReport: postReport,
           SPREAD: SPREAD, ANCHOR_WEIGHT: ANCHOR_WEIGHT, CLOSED_AT_HALF_LIVES: CLOSED_AT_HALF_LIVES };
});
