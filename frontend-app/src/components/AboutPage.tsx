import { ReactNode, useEffect, useState } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { Movie } from '../types';
import { fetchSimilar } from '../lib/api';
import { cn } from '../lib/utils';
import ProjectFlowchart from './ProjectFlowchart';
import PosterCard from './PosterCard';
import { Ground } from './Collections';
import { Rays } from './Marks';

/** Sense and Sensibility — the film the ID-mapping bug was caught on. */
const CASE_ITEM = 16;

const Panel = ({ title, aside, children, className }: {
    title: ReactNode; aside?: ReactNode; children: ReactNode; className?: string;
}) => (
    <section className={cn('bg-panel rounded-[24px] p-5 md:p-8', className)}>
        <header className="flex flex-wrap items-end gap-x-4 gap-y-2 mb-6">
            <h2 className="display text-[30px] md:text-[44px]">{title}</h2>
            {aside && <p className="text-[15px] text-mute pb-1">{aside}</p>}
        </header>
        {children}
    </section>
);

const FIGURES: [string, string][] = [
    ['32M', 'ratings'],
    ['17,719', 'films indexed'],
    ['24', 'matches per film'],
    ['~1 ms', 'per lookup'],
];

const STEPS = [
    {
        title: 'Learn from 32 million ratings',
        body: 'Every film is scored against every other by how often the same viewers rated both highly. Prolific accounts are damped so none dominates, and blockbusters are damped so popularity alone never counts as similarity.',
        bg: 'linear-gradient(180deg,#3a22e6 0%,#2913a8 28%,#140a66 64%,#03010d 100%)', rays: true,
    },
    {
        title: 'Place each film in 64 dimensions',
        body: 'A two-tower network, trained with InfoNCE and in-batch negatives, maps 17,719 films into one embedding space. Distance there is a learned measure of taste, and it finds neighbours co-occurrence alone would miss.',
        bg: 'linear-gradient(180deg,#7a1fe0 0%,#45108f 36%,#14063a 72%,#05020e 100%)', rays: true,
    },
    {
        title: 'Gather candidates from both',
        body: 'The co-occurrence model and the embedding space each nominate their closest films. Taking the union keeps the precision of the first and the reach of the second, rather than trusting either alone.',
        bg: 'linear-gradient(180deg,#127075 0%,#0c4a5a 38%,#071d33 72%,#03050c 100%)', rays: false,
    },
    {
        title: 'Rerank by what the films are about',
        body: 'Raw co-occurrence drifts toward whatever else was big that year. Genre and tag overlap pull the ranking back toward the film itself, with a penalty for titles too thinly rated to trust.',
        bg: 'linear-gradient(180deg,#d40002 0%,#a30001 32%,#4a0000 70%,#120404 100%)', rays: false,
    },
];

/** The rerank's weighted sum, as configured in the build. */
const WEIGHTS = [
    { signal: 'Item-item co-occurrence', w: 1.35, what: 'People who rated this highly also rated that highly' },
    { signal: 'Two-tower similarity', w: 0.75, what: 'Closeness in the learned 64-dimension taste space' },
    { signal: 'Genre overlap', w: 0.6, what: 'Shared genres, rare ones counting for more than Drama' },
    { signal: 'Tag overlap', w: 0.5, what: 'Shared themes and subject matter' },
    { signal: 'Franchise bonus', w: 0.4, what: 'Keeps entries in a series together' },
    { signal: 'Thin-data penalty', w: -0.7, what: 'Holds back films with too few ratings to trust' },
];
const LO = -0.7, HI = 1.35, SPAN = HI - LO;
const at = (v: number) => ((v - LO) / SPAN) * 100;
// Validated against the panel surface: both pass the lightness, chroma, CVD and contrast checks.
const POS = '#7C6DFF', NEG = '#F0454A';

const CHECKS: [string, string][] = [
    ['0', 'films recommended to themselves'],
    ['0', 'out-of-range or duplicate IDs'],
    ['0', 'neighbours missing a title'],
    ['100%', 'of the 17,719 score rows sorted best first'],
    ['94.8%', 'of matches share a genre with their film'],
    ['8.5 yrs', 'apart, on average, a film and its matches'],
];

const STACK = [
    {
        where: 'On every request',
        tools: ['FastAPI', 'NumPy'],
        note: 'Artifacts load once per warm instance. A match lookup is an array slice — no PyTorch, no FAISS and no search at request time.',
    },
    {
        where: 'At build time only',
        tools: ['PyTorch', 'FAISS', 'pandas', 'SciPy'],
        note: 'About twenty minutes, offline. The build checks its own output across the whole catalogue and refuses to write artifacts that fail.',
    },
    {
        where: 'In your browser',
        tools: ['React 19', 'Vite', 'Tailwind CSS', 'Framer Motion'],
        note: 'One request draws the whole dashboard. Posters come straight from TMDB’s image CDN.',
    },
];

const WeightsChart = () => {
    const [hover, setHover] = useState<number | null>(null);
    return (
        <div className="relative">
            <ul className="space-y-1" onMouseLeave={() => setHover(null)}>
                {WEIGHTS.map((r, i) => {
                    const neg = r.w < 0;
                    const left = neg ? at(r.w) : at(0);
                    const width = Math.abs(r.w) / SPAN * 100;
                    return (
                        <li key={r.signal} tabIndex={0}
                            onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}
                            className={cn('grid grid-cols-[1fr_auto] md:grid-cols-[minmax(0,17rem)_1fr_4.5rem] items-center gap-x-5 gap-y-2 rounded-[14px] px-3 py-3 transition-opacity outline-offset-0',
                                hover != null && hover !== i ? 'opacity-40' : 'opacity-100',
                                hover === i && 'bg-white/[0.04]')}>
                            <div className="min-w-0">
                                <p className="text-[15px] font-medium">{r.signal}</p>
                                <p className="text-[13px] text-white/55 leading-snug">{r.what}</p>
                            </div>
                            <p className="md:order-last text-right text-[22px] numeral tabular-nums">
                                {neg ? '−' : '+'}{Math.abs(r.w).toFixed(2)}
                            </p>
                            <div aria-hidden="true" className="col-span-2 md:col-span-1 relative h-3.5">
                                <div className="absolute inset-y-[-6px] w-px bg-white/35" style={{ left: `${at(0)}%` }} />
                                <div className={cn('absolute inset-y-0', neg ? 'rounded-l-[4px]' : 'rounded-r-[4px]')}
                                    style={{ left: `${left}%`, width: `${width}%`, background: neg ? NEG : POS }} />
                            </div>
                        </li>
                    );
                })}
            </ul>
            <p className="mt-4 px-3 text-[13px] text-white/55 max-w-[70ch]">
                Each candidate's score is the weighted sum of these signals. Bars run from zero: everything but the
                penalty adds to the score; the penalty subtracts until a film has enough ratings to be trusted.
            </p>
        </div>
    );
};

const AboutPage = ({ onSelect }: { onSelect: (m: Movie) => void }) => {
    const [caseFilm, setCaseFilm] = useState<{ source: Movie; matches: Movie[] } | null>(null);

    useEffect(() => {
        const ctrl = new AbortController();
        fetchSimilar(CASE_ITEM, 4, ctrl.signal).then(setCaseFilm).catch(() => { });
        return () => ctrl.abort();
    }, []);

    return (
        <div className="space-y-1.5 md:space-y-2.5">
            <section className="relative overflow-hidden rounded-[24px] p-5 md:p-10 min-h-[440px] flex flex-col">
                <Ground look="ultra" />
                <h1 className="relative display text-[clamp(52px,10vw,148px)] max-w-[11ch]">Inside the system</h1>
                <p className="relative mt-6 max-w-[54ch] text-[17px] md:text-[19px] leading-[1.5] text-white/85">
                    How 32 million ratings become 24 matches for every film. All of it is worked out ahead of time,
                    so asking for a film's matches costs about a millisecond.
                </p>
                <dl className="relative mt-auto pt-10 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-6">
                    {FIGURES.map(([v, k]) => (
                        <div key={k}>
                            <dd className="numeral text-[52px] md:text-[76px]">{v}</dd>
                            <dt className="mt-2 text-[14px] text-white/75">{k}</dt>
                        </div>
                    ))}
                </dl>
            </section>

            <Panel title="The pipeline" aside="Running, one stage at a time">
                <ProjectFlowchart />
            </Panel>

            <Panel title="Four steps" aside="From ratings to a ranked list">
                <ol className="grid md:grid-cols-2 xl:grid-cols-4 gap-2">
                    {STEPS.map((s, i) => (
                        <li key={s.title} className="relative overflow-hidden rounded-[18px] p-5 min-h-[400px] flex flex-col">
                            <div aria-hidden="true" className="absolute inset-0" style={{ background: s.bg }} />
                            {s.rays && <Rays className="absolute inset-0 w-full h-full text-[#8f84ff]/50" origin={[0.7, 0.2]} />}
                            <span className="relative numeral text-[120px] md:text-[140px]">{i + 1}</span>
                            <h3 className="relative mt-auto text-[24px] font-medium leading-[1.05] tracking-[-0.02em] max-w-[14ch]">{s.title}</h3>
                            <p className="relative mt-3 text-[14px] leading-[1.55] text-white/80">{s.body}</p>
                        </li>
                    ))}
                </ol>
            </Panel>

            <Panel title="How the rerank weighs it" aside="Six signals, one weighted sum">
                <WeightsChart />
            </Panel>

            <Panel title="Checked before it ships" aside="The build verifies all 17,719 films before it writes anything">
                <dl className="grid sm:grid-cols-2 lg:grid-cols-3 gap-2">
                    {CHECKS.map(([v, k]) => (
                        <div key={k} className="rounded-[18px] bg-panel-hi p-5 md:p-6 min-h-[168px] flex flex-col justify-between">
                            <dd className="numeral text-[56px] md:text-[68px]">{v}</dd>
                            <dt className="mt-4 text-[14px] text-white/70 max-w-[26ch]">{k}</dt>
                        </div>
                    ))}
                </dl>
            </Panel>

            <Panel title="The bug it was rebuilt around">
                <div className="grid lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)] gap-2">
                    <div className="relative overflow-hidden rounded-[18px] p-5 md:p-7 min-h-[360px] flex flex-col"
                        style={{ background: 'linear-gradient(180deg,#2a0000 0%,#120000 60%,#000 100%)' }}>
                        <p className="text-[14px] text-white/60">Before, for Sense and Sensibility</p>
                        <ul className="mt-5 space-y-1">
                            {['Die Hard', 'Die Hard 2', 'The Cable Guy'].map(t => (
                                <li key={t} className="display text-[40px] md:text-[56px] text-white/45 line-through decoration-[#F0454A] decoration-[3px]">{t}</li>
                            ))}
                        </ul>
                        <p className="mt-auto pt-8 text-[15px] leading-[1.6] text-white/80 max-w-[52ch]">
                            Training encoded MovieLens IDs into contiguous indices but never saved the encoder, so the
                            server read row numbers as if they were IDs. The model was fine; the lookup table was gone.
                            It was rebuilt by reproducing the exact training sample, and the build now refuses to ship
                            if that mapping can't be verified.
                        </p>
                    </div>

                    <div className="rounded-[18px] bg-panel-hi p-5 md:p-7 flex flex-col">
                        <p className="text-[14px] text-white/60">Now, live from the API</p>
                        <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-2">
                            {caseFilm
                                ? caseFilm.matches.map((m, i) => (
                                    <PosterCard key={m.item_id} movie={m} onSelect={onSelect} figure="match" badge={i + 1} size="lg" />
                                ))
                                : Array.from({ length: 4 }, (_, i) => <div key={i} className="aspect-[2/3] rounded-[18px] bg-panel animate-pulse" />)}
                        </div>
                        <div className="mt-auto pt-6 flex flex-wrap items-center justify-between gap-3">
                            <p className="text-[15px] text-white/75 max-w-[40ch]">
                                Same film, same request: the neighbours it should have had all along.
                            </p>
                            {caseFilm && (
                                <button type="button" onClick={() => onSelect(caseFilm.source)} className="pill pill-ultra h-10">
                                    All 24 matches <ArrowUpRight className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    </div>
                </div>
            </Panel>

            <Panel title="What runs where" aside="Data from MovieLens (GroupLens) and TMDB">
                <div className="grid md:grid-cols-3 gap-2">
                    {STACK.map(s => (
                        <div key={s.where} className="rounded-[18px] bg-panel-hi p-5 md:p-6">
                            <h3 className="text-[18px] font-medium">{s.where}</h3>
                            <ul className="mt-4 flex flex-wrap gap-1.5">
                                {s.tools.map(t => (
                                    <li key={t} className="pill h-8 px-3.5 bg-white text-black normal-case text-[13px]">{t}</li>
                                ))}
                            </ul>
                            <p className="mt-5 text-[14px] leading-[1.6] text-white/65">{s.note}</p>
                        </div>
                    ))}
                </div>
            </Panel>
        </div>
    );
};

export default AboutPage;
