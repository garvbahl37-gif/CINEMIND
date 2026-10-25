import { useEffect, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { fetchEvaluation } from '../lib/api';
import { cn } from '../lib/utils';

// Validated against the panel surface (#161616) with the dataviz validator:
// accent and cost pass every check; the grey is a neutral, above 3:1 contrast.
const ACCENT = '#7C6DFF', COST = '#F0454A', NEUTRAL = '#74747E';

/* ------------------------------------------------------------------ discovery */

const READING = [
    { kind: 'anchor', label: 'Like Interstellar' },
    { kind: 'mood', label: 'more emotional' },
];

const DISCOVERY_STEPS = [
    {
        title: 'Read the sentence',
        body: 'Runtime ("under two hours"), era ("from the 2000s"), language, genre and what to avoid become filters. A film named after "like" or "than" becomes an anchor. What is left is the mood.',
    },
    {
        title: 'Embed the mood',
        body: 'A 256-dimension sentence model (Model2Vec, distilled from a transformer) runs in NumPy on every request. Words it spells badly, like "heist", are read through the films viewers tagged with them.',
    },
    {
        title: 'Rank by meaning',
        body: 'Each film\'s description sits in the same space. Anchors add what their audience also loved; the mood leans the ranking towards it; quality breaks ties. If a filter leaves too few films, genre turns into a preference.',
    },
];

export const DiscoveryNote = ({ onDiscover }: { onDiscover?: (q: string) => void }) => (
    <div className="grid lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] gap-2">
        <div className="rounded-[18px] bg-ultra p-5 md:p-7 flex flex-col min-h-[320px]">
            <p className="text-[clamp(28px,3.4vw,44px)] font-medium leading-[1.04] tracking-[-0.03em] max-w-[18ch]">
                Movies that feel like Interstellar but are more emotional
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-2">
                <span className="text-[14px] text-white/70 mr-1">Read as</span>
                {READING.map(c => (
                    <span key={c.label} className={cn('pill h-8 px-3.5 normal-case tracking-normal text-[13px]',
                        c.kind === 'anchor' ? 'bg-white text-black' : 'bg-violet text-white')}>{c.label}</span>
                ))}
            </div>
            <p className="mt-auto pt-8 text-[15px] text-white/80">Her, The Theory of Everything, Up, Gravity.</p>
            {onDiscover && (
                <button type="button" className="pill pill-white h-10 mt-5 self-start"
                    onClick={() => onDiscover('Movies that feel like Interstellar but are more emotional')}>
                    Try it <ArrowUpRight className="w-4 h-4" />
                </button>
            )}
        </div>
        <ol className="grid gap-2">
            {DISCOVERY_STEPS.map((s, i) => (
                <li key={s.title} className="rounded-[18px] bg-panel-hi p-5 md:p-6 grid grid-cols-[auto_1fr] gap-x-5">
                    <span className="numeral text-[44px] text-white/40">{i + 1}</span>
                    <div>
                        <h3 className="text-[18px] font-medium">{s.title}</h3>
                        <p className="mt-1.5 text-[14px] leading-[1.6] text-white/65">{s.body}</p>
                    </div>
                </li>
            ))}
        </ol>
    </div>
);

/* ------------------------------------------------------------------- adaptive */

const LOOP = [
    { title: 'Your signals', body: 'Likes, passes, your watchlist, films you open and what you describe on Discover. Kept in your browser, sent with each request.' },
    { title: 'Your taste', body: 'The two-tower model turns your likes into a point in its space, twice: weighted over the long run, and over the last few days.' },
    { title: 'Candidates', body: 'Films near that point, the co-rating neighbours of what you liked, films that fit your descriptions, and well-rated films for range.' },
    { title: 'Reranking', body: 'Relevance from all of those, a quality prior, a penalty for films like the ones you passed on, and a diversity pass so one niche can\'t fill the row.' },
    { title: 'Your picks', body: 'Each pick says why it is there, citing only films you actually liked, and how confident the engine is.' },
    { title: 'The feedback log', body: 'What was shown, where, with what probability, and what you did next — the record counterfactual evaluation needs.' },
];

const IDEAS = [
    {
        title: 'Preference drift',
        body: 'Long-run taste decays over a year; recent taste over three days. The more you have done lately, the more say it gets, up to 40%. Like five romances after a year of horror and the shelf follows within the session.',
    },
    {
        title: 'Exploration',
        body: 'At the default 30% adventurousness, three of the 24 picks are drawn from well-rated films outside your usual genres or far from your likes, sampled rather than ranked. Each is flagged, and the probability it had of being shown is logged.',
    },
    {
        title: 'Counterfactual evaluation',
        body: 'Because exploration slots carry their probabilities, a new ranking can be scored on logs the old one produced, by reweighting — before anyone sees it. The evaluation below tests that on simulated logs.',
    },
];

export const AdaptiveNote = () => (
    <>
        <ol className="grid sm:grid-cols-2 xl:grid-cols-6 gap-2">
            {LOOP.map((s, i) => (
                <li key={s.title} className="rounded-[18px] bg-panel-hi p-5 flex flex-col min-h-[220px]">
                    <span className="numeral text-[44px] text-white/40">{i + 1}</span>
                    <h3 className="mt-auto pt-6 text-[17px] font-medium leading-tight">{s.title}</h3>
                    <p className="mt-2 text-[13.5px] leading-[1.55] text-white/65">{s.body}</p>
                </li>
            ))}
        </ol>
        <div className="mt-2 grid md:grid-cols-3 gap-2">
            {IDEAS.map(s => (
                <div key={s.title} className="rounded-[18px] border border-white/10 p-5 md:p-6">
                    <h3 className="text-[18px] font-medium">{s.title}</h3>
                    <p className="mt-2 text-[14px] leading-[1.6] text-white/70">{s.body}</p>
                </div>
            ))}
        </div>
    </>
);

/* ----------------------------------------------------------------- evaluation */

type CI = [number, number];
interface Metric {
    recall: number; recall_ci: CI; ndcg: number; ndcg_ci: CI; hit_rate_10: number;
    coverage: number; diversity: number; novelty: number; long_tail: number;
    vs_adaptive?: { recall: number; recall_ci: CI; ndcg: number; ndcg_ci: CI; significant: boolean };
}
interface Estimate { truth: number; ips: number; snips: number; dm: number; dr: number; dr_ci: CI; ess: number }
interface Evaluation {
    generated: string;
    split: { cutoff: string; train_ratings: number; test_ratings: number; eligible_users: number; evaluated_users: number };
    k: number;
    metrics: Record<string, Metric>;
    ope: { estimates: Record<string, Estimate>; kendall_tau_vs_truth: Record<string, number>; logs: number; logging_policy: string; target_policies: string };
    assumptions: string[];
    notes: string[];
}

const POLICIES: { id: string; label: string; note: string }[] = [
    { id: 'popularity', label: 'Most popular', note: 'The films most often liked' },
    { id: 'item-cf', label: 'Item-to-item', note: 'Co-rating neighbours, as the match stage uses' },
    { id: 'two-tower', label: 'Two-tower alone', note: 'Nearest films to your taste point' },
    { id: 'adaptive', label: 'Adaptive engine', note: 'Everything above, reranked' },
];

const PARTS: { id: string; label: string }[] = [
    { id: 'adaptive − diversity', label: 'Diversity pass' },
    { id: 'adaptive − negatives', label: 'Passes and low ratings' },
    { id: 'adaptive − drift', label: 'Drift' },
    { id: 'adaptive − exploration', label: 'Exploration' },
];

const pts = (v: number, sign = false) => `${sign && v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v * 100).toFixed(1)}`;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

/** A row's hover readout: the value leads, the label follows. */
const Tip = ({ lines }: { lines: string[] }) => (
    <span role="tooltip" className="pointer-events-none absolute right-0 -top-2 -translate-y-full z-10 rounded-[12px] bg-black border border-white/15 px-3 py-2 text-[12px] leading-snug shadow-[0_12px_30px_-8px_rgba(0,0,0,.9)] whitespace-nowrap">
        {lines.map((l, i) => <span key={i} className={cn('block', i ? 'text-white/60' : 'text-white font-medium')}>{l}</span>)}
    </span>
);

/** Recall@20 with its 95% interval: the adaptive engine in the accent, the baselines in grey. */
const RecallChart = ({ m }: { m: Record<string, Metric> }) => {
    const [hover, setHover] = useState<string | null>(null);
    const max = Math.ceil(Math.max(...POLICIES.map(p => m[p.id]?.recall_ci[1] ?? 0)) * 20) / 20 + 0.025;
    const x = (v: number) => `${(v / max) * 100}%`;
    const ticks = Array.from({ length: Math.floor(max / 0.05) + 1 }, (_, i) => i * 0.05);
    return (
        <figure>
            <figcaption className="text-[15px] font-medium">Recall@20 on the held-out ratings</figcaption>
            <p className="mt-1 text-[13px] text-white/55">Share of each viewer's later favourites found in a list of 20. Whiskers are 95% bootstrap intervals.</p>
            <div className="mt-5 grid grid-cols-[minmax(0,9.5rem)_1fr] gap-x-4">
                {POLICIES.map(p => {
                    const r = m[p.id];
                    if (!r) return null;
                    const on = p.id === 'adaptive';
                    return (
                        <div key={p.id} className="contents">
                            <div className="py-2.5">
                                <p className={cn('text-[14px] leading-tight', on ? 'text-white font-medium' : 'text-white/80')}>{p.label}</p>
                                <p className="text-[12px] text-white/45 leading-snug">{p.note}</p>
                            </div>
                            <div tabIndex={0} aria-label={`${p.label}: recall ${r.recall.toFixed(3)}, 95% interval ${r.recall_ci[0].toFixed(3)} to ${r.recall_ci[1].toFixed(3)}`}
                                onMouseEnter={() => setHover(p.id)} onMouseLeave={() => setHover(null)}
                                onFocus={() => setHover(p.id)} onBlur={() => setHover(null)}
                                className="relative self-stretch outline-offset-2">
                                {ticks.map(t => <span key={t} aria-hidden="true" className="absolute inset-y-0 w-px bg-white/[0.07]" style={{ left: x(t) }} />)}
                                <span aria-hidden="true" className={cn('absolute top-1/2 -translate-y-1/2 left-0 h-[22px] rounded-r-[4px] transition-opacity',
                                    hover && hover !== p.id && 'opacity-50')}
                                    style={{ width: x(r.recall), background: on ? ACCENT : NEUTRAL }} />
                                <span aria-hidden="true" className="absolute top-1/2 h-px bg-white/80" style={{ left: x(r.recall_ci[0]), width: `calc(${x(r.recall_ci[1])} - ${x(r.recall_ci[0])})` }} />
                                {r.recall_ci.map((c, i) => <span key={i} aria-hidden="true" className="absolute top-1/2 -translate-y-1/2 h-2.5 w-px bg-white/80" style={{ left: x(c) }} />)}
                                <span className={cn('absolute top-1/2 -translate-y-1/2 ml-2 text-[13px] tabular-nums', on ? 'text-white font-medium' : 'text-white/70')}
                                    style={{ left: x(r.recall_ci[1]) }}>{r.recall.toFixed(3)}</span>
                                {hover === p.id && <Tip lines={[`Recall@20 ${r.recall.toFixed(3)} (${r.recall_ci[0].toFixed(3)}–${r.recall_ci[1].toFixed(3)})`, `NDCG@20 ${r.ndcg.toFixed(3)}`, `${pct(r.coverage)} of the catalogue recommended`]} />}
                            </div>
                        </div>
                    );
                })}
                <span />
                <div aria-hidden="true" className="relative h-6 mt-1">
                    {ticks.map(t => <span key={t} className="absolute -translate-x-1/2 text-[11px] text-white/45 tabular-nums" style={{ left: x(t) }}>{t === 0 ? '0' : t.toFixed(2)}</span>)}
                </div>
            </div>
        </figure>
    );
};

/** What each part of the engine adds, in recall points, measured by switching it off on the same viewers. */
const PartsChart = ({ m }: { m: Record<string, Metric> }) => {
    const [hover, setHover] = useState<string | null>(null);
    const rows = PARTS.map(p => {
        const d = m[p.id]?.vs_adaptive;
        // Switching a part off moved recall by d; what the part adds is the opposite.
        return d && { ...p, v: -d.recall, lo: -d.recall_ci[1], hi: -d.recall_ci[0], sig: d.significant };
    }).filter((r): r is NonNullable<typeof r> => !!r);
    if (!rows.length) return null;
    const lo = Math.min(0, ...rows.map(r => r.lo)) - 0.002, hi = Math.max(0, ...rows.map(r => r.hi)) + 0.004;
    const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
    return (
        <figure>
            <figcaption className="text-[15px] font-medium">What each part adds</figcaption>
            <p className="mt-1 text-[13px] text-white/55">Recall points gained by keeping each part, from switching it off for the same 3,000 viewers. Paired 95% intervals; grey crosses zero.</p>
            <div className="mt-5 grid grid-cols-[minmax(0,9.5rem)_1fr] gap-x-4">
                {rows.map(r => {
                    const color = !r.sig ? NEUTRAL : r.v > 0 ? ACCENT : COST;
                    const left = r.v < 0 ? x(r.v) : x(0);
                    const width = `${(Math.abs(r.v) / (hi - lo)) * 100}%`;
                    const verdict = !r.sig ? 'no measurable effect' : r.v > 0 ? 'helps' : 'costs recall offline';
                    return (
                        <div key={r.id} className="contents">
                            <p className="py-3 text-[14px] text-white/85 leading-tight">{r.label}</p>
                            <div tabIndex={0} aria-label={`${r.label}: ${pts(r.v, true)} recall points, interval ${pts(r.lo, true)} to ${pts(r.hi, true)}, ${verdict}`}
                                onMouseEnter={() => setHover(r.id)} onMouseLeave={() => setHover(null)}
                                onFocus={() => setHover(r.id)} onBlur={() => setHover(null)}
                                className="relative self-stretch outline-offset-2">
                                <span aria-hidden="true" className="absolute inset-y-0 w-px bg-white/35" style={{ left: x(0) }} />
                                <span aria-hidden="true" className={cn('absolute top-1/2 -translate-y-1/2 h-[22px]', r.v < 0 ? 'rounded-l-[4px]' : 'rounded-r-[4px]')}
                                    style={{ left, width, background: color }} />
                                <span aria-hidden="true" className="absolute top-1/2 h-px bg-white/80" style={{ left: x(r.lo), width: `calc(${x(r.hi)} - ${x(r.lo)})` }} />
                                {[r.lo, r.hi].map((c, i) => <span key={i} aria-hidden="true" className="absolute top-1/2 -translate-y-1/2 h-2.5 w-px bg-white/80" style={{ left: x(c) }} />)}
                                <span className="absolute top-1/2 -translate-y-1/2 text-[13px] tabular-nums text-white/80 whitespace-nowrap"
                                    style={r.hi > 0 ? { left: x(r.hi), marginLeft: 8 } : { right: `calc(100% - ${x(r.lo)})`, marginRight: 8 }}>
                                    {pts(r.v, true)}
                                </span>
                                {hover === r.id && <Tip lines={[`${pts(r.v, true)} points (${pts(r.lo, true)} to ${pts(r.hi, true)})`, verdict]} />}
                            </div>
                        </div>
                    );
                })}
                <span />
                <div aria-hidden="true" className="relative h-6 mt-1 text-[11px] text-white/45">
                    <span className="absolute left-0">Costs recall</span>
                    <span className="absolute -translate-x-1/2" style={{ left: x(0) }}>0</span>
                    <span className="absolute right-0">Adds recall</span>
                </div>
            </div>
            {rows.some(r => r.id === 'adaptive − exploration' && r.sig && r.v < 0) && (
                <p className="mt-4 text-[13px] leading-[1.55] text-white/60 max-w-[62ch]">
                    Exploration is meant to cost a little here. A held-out test only rewards films people already went
                    on to rate, so it sees the slots exploration spends but not what it learns about you.
                </p>
            )}
        </figure>
    );
};

const ESTIMATORS: [keyof Omit<Estimate, 'truth' | 'dr_ci' | 'ess'>, string][] = [
    ['ips', 'inverse propensity scoring'], ['snips', 'its self-normalised form'],
    ['dr', 'the doubly robust estimator'], ['dm', 'the direct method'],
];

/** The OPE result in words, from the numbers rather than written ahead of them. */
function opeVerdict(est: Record<string, Estimate>, tau: Record<string, number>) {
    const exact = ESTIMATORS.filter(([k]) => tau[k] === 1).map(([, name]) => name);
    const off = ESTIMATORS.filter(([k]) => tau[k] != null && tau[k] < 1);
    const ids = POLICIES.map(p => p.id).filter(id => est[id]);
    const allLow = (k: 'dm') => ids.every(id => est[id][k] < est[id].truth);
    const sentences: string[] = [];
    if (exact.length) {
        const list = exact.length > 1 ? `${exact.slice(0, -1).join(', ')} and ${exact[exact.length - 1]}` : exact[0];
        sentences.push(`${list[0].toUpperCase()}${list.slice(1)} put the ${ids.length} policies in their true order (Kendall τ = 1).`);
    }
    for (const [k, name] of off) {
        const low = k === 'dm' && allLow('dm') ? ', reads every policy low' : '';
        sentences.push(`${name[0].toUpperCase()}${name.slice(1)}${k === 'dm' ? ', which trusts a reward model alone' : ''}${low} and gets part of the order wrong (τ = ${tau[k].toFixed(2)}).`);
    }
    const thin = POLICIES.filter(p => est[p.id] && est[p.id].ess < 100);
    if (thin.length) {
        sentences.push(`An estimate is only as good as the overlap: ${thin.map(p => p.label.toLowerCase()).join(' and ')} picks films the logging policy rarely showed, so it rests on ${thin.map(p => Math.round(est[p.id].ess)).join(' and ')} effective logs and a wide interval.`);
    }
    return sentences.join(' ');
}

const Stat = ({ value, label }: { value: string; label: string }) => (
    <div className="rounded-[18px] bg-panel-hi p-5 md:p-6 min-h-[150px] flex flex-col justify-between">
        <p className="numeral text-[52px] md:text-[64px]">{value}</p>
        <p className="mt-4 text-[14px] text-white/70 max-w-[30ch]">{label}</p>
    </div>
);

export const EvaluationNote = () => {
    const [d, setD] = useState<Evaluation | null>(null);
    const [missing, setMissing] = useState(false);

    useEffect(() => {
        const ctrl = new AbortController();
        fetchEvaluation(ctrl.signal).then(setD).catch(e => { if (e.name !== 'AbortError') setMissing(true); });
        return () => ctrl.abort();
    }, []);

    if (missing) return <p className="text-[15px] text-white/70">The evaluation hasn't been published with this build. Run pipelines/evaluate.py to produce it.</p>;
    if (!d) return <div className="h-[420px] rounded-[18px] bg-panel-hi animate-pulse" aria-label="Loading the evaluation" />;

    const m = d.metrics, a = m['adaptive'], cf = m['item-cf'];
    const gain = cf?.vs_adaptive;
    const est = d.ope.estimates;
    const tau = d.ope.kendall_tau_vs_truth;

    return (
        <div className="space-y-2">
            <div className="grid sm:grid-cols-3 gap-2">
                <Stat value={a.recall.toFixed(3)} label={`Recall@20 for the adaptive engine, on ${d.split.evaluated_users.toLocaleString()} viewers' ratings after ${d.split.cutoff.slice(0, 4)}`} />
                {gain && <Stat value={`${pts(-gain.recall, true)} pts`} label={`over item-to-item, 95% interval ${pts(-gain.recall_ci[1])} to ${pts(-gain.recall_ci[0])} points (paired)`} />}
                {cf && <Stat value={`${(a.coverage / cf.coverage).toFixed(1)}×`} label={`as much of the catalogue recommended (${pct(a.coverage)} against ${pct(cf.coverage)})`} />}
            </div>

            <div className="grid xl:grid-cols-2 gap-2">
                <div className="rounded-[18px] bg-panel-hi p-5 md:p-7"><RecallChart m={m} /></div>
                <div className="rounded-[18px] bg-panel-hi p-5 md:p-7"><PartsChart m={m} /></div>
            </div>

            <div className="rounded-[18px] bg-panel-hi p-5 md:p-7 overflow-x-auto">
                <h3 className="text-[15px] font-medium">Every measure</h3>
                <table className="mt-4 w-full min-w-[640px] text-[13.5px] tabular-nums">
                    <thead className="text-white/50 text-left">
                        <tr className="border-b border-white/10">
                            <th className="font-normal py-2 pr-4">Policy</th>
                            <th className="font-normal py-2 px-3 text-right">Recall@20</th>
                            <th className="font-normal py-2 px-3 text-right">NDCG@20</th>
                            <th className="font-normal py-2 px-3 text-right">Hit rate@10</th>
                            <th className="font-normal py-2 px-3 text-right">Catalogue coverage</th>
                            <th className="font-normal py-2 px-3 text-right">Novelty (bits)</th>
                            <th className="font-normal py-2 pl-3 text-right">Beyond the top 1,000</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[...POLICIES.map(p => [p.id, p.label]), ...PARTS.map(p => [p.id, `Without: ${p.label.toLowerCase()}`])].map(([id, label]) => {
                            const r = m[id];
                            if (!r) return null;
                            return (
                                <tr key={id} className={cn('border-b border-white/[0.06]', id === 'adaptive' && 'text-white font-medium')}>
                                    <td className="py-2 pr-4 text-left">{label}</td>
                                    <td className="py-2 px-3 text-right">{r.recall.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{r.ndcg.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{pct(r.hit_rate_10)}</td>
                                    <td className="py-2 px-3 text-right">{pct(r.coverage)}</td>
                                    <td className="py-2 px-3 text-right">{r.novelty.toFixed(2)}</td>
                                    <td className="py-2 pl-3 text-right">{pct(r.long_tail)}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            <div className="rounded-[18px] bg-panel-hi p-5 md:p-7 overflow-x-auto">
                <h3 className="text-[15px] font-medium">Scoring a policy without showing it</h3>
                <p className="mt-1 text-[13px] text-white/55 max-w-[80ch]">
                    {d.ope.logs.toLocaleString()} simulated impressions, shown by a popularity-leaning policy with known
                    probabilities. Each estimator reweights them to predict how often another policy's pick would be a
                    later favourite; the truth column is that rate computed exactly.
                </p>
                <table className="mt-4 w-full min-w-[560px] text-[13.5px] tabular-nums">
                    <thead className="text-white/50">
                        <tr className="border-b border-white/10">
                            <th className="font-normal py-2 pr-4 text-left">Policy</th>
                            <th className="font-normal py-2 px-3 text-right">Truth</th>
                            <th className="font-normal py-2 px-3 text-right">IPS</th>
                            <th className="font-normal py-2 px-3 text-right">SNIPS</th>
                            <th className="font-normal py-2 px-3 text-right">Direct method</th>
                            <th className="font-normal py-2 px-3 text-right">Doubly robust</th>
                            <th className="font-normal py-2 pl-3 text-right">Effective logs</th>
                        </tr>
                    </thead>
                    <tbody>
                        {POLICIES.map(p => {
                            const e = est[p.id];
                            if (!e) return null;
                            return (
                                <tr key={p.id} className="border-b border-white/[0.06]">
                                    <td className="py-2 pr-4">{p.label}</td>
                                    <td className="py-2 px-3 text-right text-white font-medium">{e.truth.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{e.ips.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{e.snips.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{e.dm.toFixed(3)}</td>
                                    <td className="py-2 px-3 text-right">{e.dr.toFixed(3)} <span className="text-white/45">({e.dr_ci[0].toFixed(3)}–{e.dr_ci[1].toFixed(3)})</span></td>
                                    <td className="py-2 pl-3 text-right">{Math.round(e.ess).toLocaleString()}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                <p className="mt-4 text-[13.5px] leading-[1.6] text-white/70 max-w-[80ch]">{opeVerdict(est, tau)}</p>
            </div>

            <div className="rounded-[18px] border border-white/10 p-5 md:p-7">
                <h3 className="text-[15px] font-medium">What these numbers can't tell you</h3>
                <ul className="mt-3 space-y-2.5 text-[13.5px] leading-[1.6] text-white/70 max-w-[86ch] list-disc pl-5 marker:text-white/30">
                    {[...d.assumptions, ...d.notes].map(t => <li key={t}>{t}</li>)}
                </ul>
            </div>
        </div>
    );
};
