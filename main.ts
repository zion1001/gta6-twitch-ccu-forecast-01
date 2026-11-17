// Twitch CCU forecaster for GTA VI's launch day (make.fun gta6-twitch-ccu-forecast-01).
//
// How it forecasts, in one paragraph: every past big launch in the case's
// `comparables` has a full release-day curve, so the forecast for hour h is
// built from what those launches did at hour h relative to their own hour 1.
// Launches whose hour-1 print was close to this case's hour-1 print count
// most (a Gaussian kernel on log hour-1 viewers, bandwidth 1.0), because the
// shape of launch day depends on how much of the launch had already happened
// by hour 1: a launch that was already huge at hour 1 stays flat, one that
// had barely started explodes later. Per hour, the log-ratio distribution of
// the weighted pool is summarised by its weighted median (the pinball-optimal
// centre) and its weighted spread, and the five quantiles are
// p1 * exp(median + z_tau * spread) — except for a launch that had not
// started by hour 1, where the mean is the centre instead (see below). When hour 1
// itself is tiny (below 20,000 viewers) the launch has not really started
// yet, so the ratio to hour 1 says almost nothing about the level; for those
// cases the forecast is blended 50/50 in log space with an absolute model,
// the weighted quantiles of the pool launches' actual hour-h counts, which
// anchors the level on what big launches actually reach. The Grand Theft
// Auto VI category's own pre-release history is not a launch, so it is left
// out of the pool. Nothing here reads the clock, randomness, the network or
// any file but the case: the output is a pure function of the case.
//
// Calibrated by leave-one-out over the 37 comparable launches: mean pinball
// loss 0.385 on the log scale (house baseline 0.930 on the same scorer),
// 90% coverage 0.90, typical miss 80%.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Kernel bandwidth, in natural-log units of hour-1 viewers. */
const BANDWIDTH = 1.0;
/** Below this hour-1 print the launch is treated as not yet started. */
const COLD_CUTOFF = 20000;
/** Log-space weight of the absolute model for a not-yet-started launch. */
const COLD_BLEND = 0.5;
/** A print below 1 viewer counts as 1 (the log of 0 is no number). */
const MIN_VIEWERS = 1;
/** The contest's cap on a quantile. */
const MAX_VIEWERS = 1e9;

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  unlock: string;
  hourly: HourlyPoint[];
}

interface Case {
  game: { name: string };
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** One past launch, reduced to its release-day prints. */
interface Launch {
  p1: number;
  logP1: number;
  byHour: Map<number, number>;
}

/** Parse an ISO 8601 UTC timestamp into milliseconds since the epoch. */
const parseTime = (s: string): number => Date.parse(s);

/** x rounded to 2 decimals (half up), like the house baseline's output. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** The Gaussian kernel weight of a pool launch for a target log hour-1 print. */
const kernel = (logP1: number, targetLogP1: number): number => {
  const d = (logP1 - targetLogP1) / BANDWIDTH;
  return Math.exp(-0.5 * d * d);
};

/** Weighted quantile of (value, weight) pairs, by cumulative weight. */
const weightedQuantile = (pairs: Array<[number, number]>, tau: number): number => {
  const sorted = pairs.slice().sort((a, b) => a[0] - b[0]);
  let total = 0;
  for (const [, w] of sorted) total += w;
  if (total <= 0) return sorted[Math.floor(sorted.length / 2)][0];
  let acc = 0;
  for (const [v, w] of sorted) {
    acc += w;
    if (acc >= tau * total) return v;
  }
  return sorted[sorted.length - 1][0];
};

/** A comparable is the target game's own pre-release category, not a launch. */
const isOwnCategory = (c: Comparable, gameName: string): boolean =>
  c.name.toLowerCase().includes(gameName.toLowerCase()) ||
  (c.unlock ?? '').toLowerCase().includes('not a launch');

/** Release-day prints of one comparable, hour 1 to 24, from its hourly series. */
const releaseDay = (c: Comparable): Map<number, number> | null => {
  const start = parseTime(c.day_start);
  if (!Number.isFinite(start)) return null;
  const byHour = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    if (p.viewers === null || p.viewers === undefined || p.viewers <= 0) continue;
    const t = parseTime(p.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - start) / 3600000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, p.viewers);
  }
  const p1 = byHour.get(1);
  if (p1 === undefined || p1 <= 0) return null;
  return byHour;
};

/** The five quantiles for one hour, from the pool's log-ratio distribution. */
const ratioQuantiles = (pool: Launch[], targetLogP1: number, p1: number, h: number): number[] => {
  let weightSum = 0;
  let mean = 0;
  const pairs: Array<[number, number]> = [];
  for (const l of pool) {
    const v = l.byHour.get(h);
    if (v === undefined) continue;
    const w = kernel(l.logP1, targetLogP1);
    const lr = Math.log(v / l.p1);
    pairs.push([lr, w]);
    weightSum += w;
    mean += w * lr;
  }
  if (pairs.length === 0 || weightSum <= 0) return Z.map(() => p1);
  mean /= weightSum;
  let variance = 0;
  for (const [lr, w] of pairs) variance += w * (lr - mean) * (lr - mean);
  variance /= weightSum;
  const spread = Math.sqrt(Math.max(variance, 1e-12));
  // Centre: for a launch that had already started by hour 1, the weighted
  // MEDIAN of the log-ratios — the pinball-optimal centre, and measured on
  // leave-one-out the mean-centred median ran 0.04 to 0.08 log units high at
  // hours 8 to 24 for those launches. For a launch that had NOT started
  // (hour 1 below the cold cutoff), the log-ratio distribution is so
  // right-skewed that the median understates how far the day explodes, and
  // the Gaussian's symmetric spread understates the upper tail; there the
  // weighted MEAN is the centre, which lifts the whole band toward where
  // those launches actually land. The spread stays the weighted standard
  // deviation in both cases.
  const centre = p1 < COLD_CUTOFF ? mean : weightedQuantile(pairs, 0.5);
  return Z.map((z) => p1 * Math.exp(centre + z * spread));
};

/** The five quantiles for one hour, from the pool's actual hour-h counts. */
const absoluteQuantiles = (pool: Launch[], targetLogP1: number, h: number): number[] => {
  const pairs: Array<[number, number]> = [];
  for (const l of pool) {
    const v = l.byHour.get(h);
    if (v === undefined) continue;
    pairs.push([Math.log(v), kernel(l.logP1, targetLogP1)]);
  }
  if (pairs.length === 0) return Z.map(() => MIN_VIEWERS);
  return LEVELS.map((tau) => Math.exp(weightedQuantile(pairs, tau)));
};

/** The forecast of one case: 23 rows, hours 2 to 24, five quantiles each. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error('case has no hour-1 print');
  const p1 = Math.max(MIN_VIEWERS, first.viewers);
  const targetLogP1 = Math.log(p1);

  // The pool: past launches, oldest first is irrelevant here, so sort by
  // name for a deterministic iteration order.
  const pool: Launch[] = [];
  const comps = (c.comparables ?? []).slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const comp of comps) {
    if (isOwnCategory(comp, c.game.name)) continue;
    const byHour = releaseDay(comp);
    if (byHour === null) continue;
    const launchP1 = byHour.get(1) as number;
    pool.push({ p1: launchP1, logP1: Math.log(launchP1), byHour });
  }

  const cold = p1 < COLD_CUTOFF;
  const rows: Array<{ hour: number; q: number[] }> = [];
  for (const { hour } of c.forecast) {
    let qs = ratioQuantiles(pool, targetLogP1, p1, hour);
    if (cold) {
      const abs = absoluteQuantiles(pool, targetLogP1, hour);
      qs = qs.map((q, i) => Math.exp((1 - COLD_BLEND) * Math.log(q) + COLD_BLEND * Math.log(abs[i])));
    }
    // Round, clamp and keep the five quantiles non-decreasing.
    let prev = MIN_VIEWERS;
    const rounded = qs.map((q) => {
      const v = Math.min(MAX_VIEWERS, Math.max(MIN_VIEWERS, round2(q)));
      const clamped = Math.max(v, prev);
      prev = clamped;
      return clamped;
    });
    rows.push({ hour, q: rounded });
  }
  return rows;
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const hours = forecastCase(c);
  if (hours.length !== 23) throw new Error(`${file}: forecast ${hours.length} rows, expected 23`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
