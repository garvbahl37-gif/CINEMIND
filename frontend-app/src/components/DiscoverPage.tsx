import { useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Minus } from 'lucide-react';
import { Movie } from '../types';
import { Discovery, fetchDiscover } from '../lib/api';
import { cn } from '../lib/utils';
import PosterCard from './PosterCard';

const EXAMPLES = [
    'Movies that feel like Interstellar but are more emotional',
    'Psychological thrillers with unreliable narrators and a shocking ending',
    'Something uplifting under two hours for tonight',
    'A heist movie that isn’t too violent',
    'Korean thrillers from the 2000s',
];

interface Props {
    /** The description to run, when one arrives from the search box. */
    query: string;
    onQuery: (q: string) => void;
    onSelect: (m: Movie) => void;
    /** A description was run: it counts as intent for your picks. */
    onSearched: (q: string) => void;
}

/** How each part of the reading is drawn: what you asked for is solid, what you ruled out is outlined. */
const chipClass = (kind: string) =>
    kind === 'anchor' ? 'bg-white text-black'
        : kind === 'mood' ? 'bg-violet text-white'
            : kind === 'avoid' ? 'border border-white/40 text-white'
                : 'bg-white/[0.13] text-white';

/** Runtime, era, language, genre — and "No Horror" — narrow the catalogue; a mood or an anchor only ranks it. */
const filtered = (d: Discovery) => d.understood.chips.some(c =>
    c.kind !== 'anchor' && c.kind !== 'mood' && (c.kind !== 'avoid' || c.label.startsWith('No ')));

function countLine(d: Discovery) {
    const n = d.results.length, total = d.total.toLocaleString();
    if (!filtered(d)) return `The ${n} closest, ranked from ${total} films`;
    return n < d.total ? `The best ${n} of ${total} films that fit` : `${total} ${d.total === 1 ? 'film fits' : 'films fit'}`;
}

/** A result: the card, then why it fits. Quoted viewer tags become small pills. */
const Fit = ({ m, onSelect }: { m: Movie; onSelect: (m: Movie) => void }) => {
    const tags = (m.why ?? []).filter(w => w.startsWith('“'));
    const lines = (m.why ?? []).filter(w => !w.startsWith('“'));
    return (
        <div>
            <PosterCard movie={m} onSelect={onSelect} size="lg" />
            <div className="mt-2.5 px-0.5 space-y-1.5">
                {lines.length > 0 && <p className="text-[12.5px] leading-[1.35] text-white/75 line-clamp-2">{lines.join(', ')}</p>}
                {tags.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                        {tags.map(t => (
                            <span key={t} className="pill-sm border border-white/25 text-white/80 max-w-full">
                                <span className="truncate">{t.slice(1, -1)}</span>
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

/**
 * Describe a film in your own words. The sentence is the page: set large on
 * the ultramarine, then read back as the parts it was understood as, then
 * answered with films that each say why they fit.
 */
const DiscoverPage = ({ query, onQuery, onSelect, onSearched }: Props) => {
    const [text, setText] = useState(query);
    const [data, setData] = useState<Discovery | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);
    const box = useRef<HTMLTextAreaElement>(null);

    // Grow the box with the sentence instead of scrolling inside it.
    useEffect(() => {
        const el = box.current;
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = `${el.scrollHeight}px`;
    }, [text]);

    useEffect(() => { setText(query); }, [query]);

    useEffect(() => {
        const q = query.trim();
        if (!q) { setData(null); return; }
        const ctrl = new AbortController();
        setLoading(true);
        setError(null);
        fetchDiscover(q, 36, ctrl.signal)
            .then(d => { setData(d); onSearched(q); })
            .catch(e => { if (e.name !== 'AbortError') setError(e instanceof Error ? e.message : String(e)); })
            .finally(() => { if (!ctrl.signal.aborted) setLoading(false); });
        return () => ctrl.abort();
        // onSearched is a fresh closure each render; the query (or a retry) decides a fetch.
    }, [query, attempt]);

    const run = (q: string) => {
        const t = q.trim();
        if (!t) { box.current?.focus(); return; }
        setText(t);
        onQuery(t);
    };

    const u = data?.understood;

    return (
        <>
            <section className="bg-ultra rounded-[24px] px-5 py-7 md:px-10 md:py-11">
                <form onSubmit={(e) => { e.preventDefault(); run(text); }}>
                    <label htmlFor="describe" className="block text-[15px] md:text-[17px] text-white/75 max-w-[52ch]">
                        Describe what you’re in the mood for: a feeling, a film it should be like, how long you’ve got.
                    </label>
                    <textarea
                        id="describe"
                        ref={box}
                        rows={1}
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); run(text); } }}
                        placeholder="Something uplifting under two hours"
                        maxLength={300}
                        className="mt-4 w-full resize-none overflow-hidden bg-transparent outline-none
                                   text-[clamp(32px,5.2vw,72px)] font-medium leading-[1.02] tracking-[-0.035em]
                                   text-white placeholder:text-white/35 caret-white
                                   focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-white/40 rounded-[6px]"
                    />
                    <div className="mt-6 flex flex-wrap items-center gap-2">
                        <button type="submit" className="pill pill-white h-11 px-5">
                            Find films <ArrowUpRight className="w-4 h-4" />
                        </button>
                        {EXAMPLES.filter(x => x !== query).slice(0, 4).map(x => (
                            <button key={x} type="button" onClick={() => run(x)}
                                className="pill h-9 px-3.5 normal-case tracking-normal text-[13px] font-normal bg-white/[0.13] hover:bg-white/[0.22] max-w-full">
                                <span className="truncate">{x}</span>
                            </button>
                        ))}
                    </div>
                </form>
            </section>

            {(data || loading || error) && (
                <section className="bg-panel rounded-[24px] p-5 md:p-7" aria-live="polite" aria-busy={loading}>
                    {error ? (
                        <>
                            <h2 className="display text-[30px] md:text-[44px]">That didn’t go through</h2>
                            <p className="mt-3 text-white/70 max-w-[56ch]">The discovery service didn’t answer ({error}). Try again in a moment.</p>
                            <button type="button" className="pill pill-white mt-5" onClick={() => setAttempt(a => a + 1)}>Try again</button>
                        </>
                    ) : (
                        <>
                            <div className="flex flex-wrap items-center gap-2">
                                <h2 className="text-[15px] text-white/60 mr-1">I read that as</h2>
                                {u?.chips.length ? u.chips.map((c, i) => (
                                    <span key={`${c.kind}-${i}`} className={cn('pill h-8 px-3.5 normal-case tracking-normal text-[13px]', chipClass(c.kind))}>
                                        {c.kind === 'avoid' && <Minus className="w-3.5 h-3.5" />}{c.label}
                                    </span>
                                )) : !loading && <span className="text-[14px] text-white/80">the best-rated films overall</span>}
                            </div>
                            {data?.notes.map(n => <p key={n} className="mt-3 text-[13px] text-white/60 max-w-[70ch]">{n}</p>)}

                            {data && data.results.length > 0 && (
                                <p className="mt-6 text-[13px] text-white/50 tabular-nums">{countLine(data)}</p>
                            )}

                            <div className={cn('mt-3 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6 gap-x-3 gap-y-6 transition-opacity',
                                loading && data && 'opacity-40')}>
                                {data
                                    ? data.results.map(m => <Fit key={m.item_id} m={m} onSelect={onSelect} />)
                                    : Array.from({ length: 12 }, (_, i) => (
                                        <div key={i}>
                                            <div className="aspect-[2/3] rounded-[18px] bg-panel-hi animate-pulse" />
                                            <div className="mt-2.5 h-3 w-3/4 rounded bg-panel-hi animate-pulse" />
                                        </div>
                                    ))}
                            </div>
                        </>
                    )}
                </section>
            )}
        </>
    );
};

export default DiscoverPage;
