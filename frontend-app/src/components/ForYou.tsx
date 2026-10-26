import { useEffect, useMemo, useState } from 'react';
import { Compass } from 'lucide-react';
import { Movie } from '../types';
import { ForYou as Picks, fetchForYou } from '../lib/api';
import { Taste } from '../lib/taste';
import { cn } from '../lib/utils';
import Shelf from './Shelf';
import PosterCard from './PosterCard';

const LEVEL = { High: 3, Medium: 2, Low: 1 } as const;

/** Three ticks for how sure the engine is: evidence from your likes, its score, agreement between sources. */
const Confidence = ({ label }: { label: NonNullable<Movie['confidence_label']> }) => (
    <span className="inline-flex items-center gap-1.5 text-[11px] text-white/50" title={`${label} confidence`}>
        <span aria-hidden="true" className="flex gap-[3px]">
            {[1, 2, 3].map(i => (
                <span key={i} className={cn('w-[5px] h-[5px] rounded-full', i <= LEVEL[label] ? 'bg-white/80' : 'bg-white/20')} />
            ))}
        </span>
        {label} confidence
    </span>
);

/** A recommendation: the card, then the one reason it is here. */
export const RecCard = ({ m, onSelect }: { m: Movie; onSelect: (m: Movie) => void }) => (
    <div className="shrink-0 snap-start w-[150px] sm:w-[172px] lg:w-[196px]">
        <PosterCard movie={m} onSelect={onSelect} size="lg"
            badge={m.exploring ? <Compass className="w-3.5 h-3.5" aria-label="Exploring" /> : undefined} />
        <div className="mt-2.5 px-0.5" title={m.evidence?.map(e => e.text).join('\n')}>
            <p className={cn('text-[12.5px] leading-[1.35] line-clamp-2', m.exploring ? 'text-violet-soft' : 'text-white/75')}>
                {m.reason}
            </p>
            {m.confidence_label && <div className="mt-1.5"><Confidence label={m.confidence_label} /></div>}
        </div>
    </div>
);

const Skeleton = () => (
    <>
        {Array.from({ length: 8 }, (_, i) => (
            <div key={i} className="shrink-0 w-[150px] sm:w-[172px] lg:w-[196px]">
                <div className="aspect-[2/3] rounded-[18px] bg-panel-hi animate-pulse" />
                <div className="mt-2.5 h-3 w-4/5 rounded bg-panel-hi animate-pulse" />
            </div>
        ))}
    </>
);

const count = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** What the profile says about you, in a line. */
function summary(p: Picks['profile']) {
    const parts = [count(p.liked, 'film')];
    if (p.skipped) parts.push(`${count(p.skipped, 'pass', 'passes')}`);
    if (p.searches) parts.push(count(p.searches, 'search', 'searches'));
    const base = `From ${parts.join(', ')}`;
    // Only when recent activity carries real weight and has moved the top genre.
    const shift = p.drift >= 0.2 && p.lately[0] && p.lately[0] !== p.long_term[0];
    return shift ? `${base}, leaning ${p.lately[0].toLowerCase()} lately` : base;
}

/**
 * The adaptive engine's picks for this browser, refetched whenever a like,
 * pass, watchlist change, search or the adventurousness setting changes —
 * but not on every film opened, so the shelf holds still behind the stage.
 */
const ForYou = ({ taste, onSelect }: { taste: Taste; onSelect: (m: Movie) => void }) => {
    const [data, setData] = useState<Picks | null>(null);
    const [failed, setFailed] = useState(false);

    const key = useMemo(() => JSON.stringify([
        taste.signals.filter(s => s.kind !== 'view').map(s => [s.item_id, s.kind]),
        taste.searches.map(s => s.q),
        taste.explore,
    ]), [taste.signals, taste.searches, taste.explore]);

    const { signals, searches, explore, logShown } = taste;
    useEffect(() => {
        const ctrl = new AbortController();
        const t = setTimeout(() => {
            fetchForYou({ signals, searches, explore }, ctrl.signal)
                .then(d => { setData(d); setFailed(false); logShown(d.results, d.policy.version ?? 'adaptive'); })
                .catch(e => { if (e.name !== 'AbortError') setFailed(true); });
        }, 300);
        return () => { clearTimeout(t); ctrl.abort(); };
        // The key captures every input that should refetch; views ride along when it does.
    }, [key]);

    const open = (m: Movie) => { taste.logOpened(m); onSelect(m); };
    const explored = data?.results.filter(m => m.exploring).length ?? 0;

    return (
        <>
            <Shelf id="for-you" title="For you"
                aside={data ? summary(data.profile) : undefined}
                controls={explored > 0 ? (
                    <span className="hidden md:inline-flex items-center gap-1.5 mr-2 text-[13px] text-white/60">
                        <Compass className="w-3.5 h-3.5" /> {count(explored, 'pick')} off your usual path
                    </span>
                ) : undefined}>
                {failed && !data ? (
                    <p className="text-[14px] text-white/70 py-6">Your picks didn't load. They'll try again when your taste changes.</p>
                ) : data ? (
                    data.results.map(m => <RecCard key={m.item_id} m={m} onSelect={open} />)
                ) : <Skeleton />}
            </Shelf>
            {/* Their own shelf, so a film nobody has rated never takes a place earned on ratings. */}
            {data && data.unrated.length > 0 && (
                <Shelf id="not-yet-rated" title="Nobody has rated these yet"
                    aside="Matched to your taste on what they are about">
                    {data.unrated.map(m => <RecCard key={m.item_id} m={m} onSelect={open} />)}
                </Shelf>
            )}
        </>
    );
};

export default ForYou;
