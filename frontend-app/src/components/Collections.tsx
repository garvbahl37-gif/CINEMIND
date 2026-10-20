import { useEffect, useState } from 'react';
import { ArrowDownRight } from 'lucide-react';
import { Movie } from '../types';
import { EMPTY, Filters, searchFilms } from '../lib/search';
import { img } from '../lib/api';
import { short } from '../lib/format';
import { cn } from '../lib/utils';
import { Rays } from './Marks';
import Shelf from './Shelf';

export type Look = 'ultra' | 'blood' | 'sunset' | 'teal' | 'violet' | 'paper' | 'smear';

export interface Collection {
    id: string;
    /** Title lines, each indented by a fraction of the tile's width. */
    lines: [string, number][];
    /** Free text the server parses, as if typed into search… */
    q?: string;
    /** …or explicit filters, which override anything the text implied. */
    filters?: Partial<Filters>;
    look: Look;
    /** Title size as a share of the tile's width, so the longest line fits. */
    size: number;
}

/**
 * Category searches, set as editorial covers. Every one is a real query —
 * the films listed on a tile are the top of what that search returns, and
 * opening it runs the same search with its filters selected.
 */
export const COLLECTIONS: Collection[] = [
    { id: 'korean-thrillers', lines: [['Korean', 0.3], ['thrillers', 0.3]], q: 'korean thrillers', look: 'ultra', size: 13.5 },
    { id: 'horror', lines: [['The best', 0], ['horrors', 0.34]], filters: { genres: ['Horror'], sort: 'rating' }, look: 'blood', size: 13 },
    { id: 'nolan', lines: [['Christopher', 0], ['Nolan', 0]], q: 'christopher nolan', look: 'smear', size: 14 },
    { id: '90s-sci-fi', lines: [['90s', 0], ['sci-fi', 0.3]], q: '90s sci-fi', look: 'teal', size: 17 },
    { id: 'noir', lines: [['Film', 0], ['noir', 0.3]], filters: { genres: ['Film-Noir'], sort: 'rating' }, look: 'paper', size: 17 },
    { id: 'ghibli', lines: [['Studio', 0], ['Ghibli', 0]], q: 'studio ghibli', look: 'smear', size: 17 },
    { id: 'hindi', lines: [['Top Hindi', 0], ['films', 0]], filters: { lang: 'hi', sort: 'rating' }, look: 'sunset', size: 17 },
    { id: 'french', lines: [['French', 0], ['cinema', 0.28]], filters: { lang: 'fr', sort: 'rating' }, look: 'violet', size: 17 },
];

export const title = (c: Collection) => c.lines.map(l => l[0]).join(' ');
export const filtersOf = (c: Collection): Filters | null => (c.filters ? { ...EMPTY, ...c.filters } : null);

const GRADIENT: Partial<Record<Look, string>> = {
    ultra: 'linear-gradient(180deg,#1a0c8a 0%,#4a34ff 9%,#2913a8 16%,#170b72 48%,#03010d 100%)',
    blood: 'linear-gradient(180deg,#d40002 0%,#b00001 30%,#5a0000 70%,#1c0505 100%)',
    sunset: 'linear-gradient(180deg,#24093f 0%,#5c0a45 26%,#c00917 58%,#e58b5e 80%,#f4e6d1 100%)',
    teal: 'linear-gradient(180deg,#0b0b0b 0%,#f0c79a 9%,#9fb79f 24%,#1b6a6a 48%,#0d2d4c 76%,#04060f 100%)',
    violet: 'linear-gradient(180deg,#7a1fe0 0%,#45108f 36%,#14063a 72%,#05020e 100%)',
};

/**
 * Slit-scan: the still on the left, and to the right of the seam the same
 * image stretched fifty times over, so each row of pixels runs out into a
 * streak of its own colour.
 */
export const Smear = ({ src, seam = 0.42, className }: { src?: string; seam?: number; className?: string }) => (
    <div aria-hidden="true" className={cn('absolute inset-0 bg-panel-hi', className)}>
        {src && <>
            <div className="absolute inset-0 bg-cover bg-center" style={{ backgroundImage: `url(${src})` }} />
            <div className="absolute inset-y-0 right-0"
                style={{
                    left: `${seam * 100}%`,
                    backgroundImage: `url(${src})`,
                    backgroundSize: '5000% 100%',
                    backgroundPosition: '46% 0',
                }} />
            <div className="absolute inset-y-0 right-0 mix-blend-multiply"
                style={{
                    left: `${seam * 100}%`,
                    backgroundImage: 'repeating-linear-gradient(180deg,rgba(0,0,0,.28) 0 2px,rgba(0,0,0,0) 2px 9px,rgba(0,0,0,.12) 9px 10px,rgba(0,0,0,0) 10px 17px)',
                }} />
        </>}
    </div>
);

/** The tile's ground, shared with the results banner so a collection opens into its own colours. */
export const Ground = ({ look, backdrop }: { look: Look; backdrop?: string }) => {
    if (look === 'smear') return <><Smear src={backdrop} /><div aria-hidden="true" className="absolute inset-0 bg-black/30" /></>;
    if (look === 'paper') return (
        <>
            <div aria-hidden="true" className="absolute inset-0 bg-[#F2F1EE]" />
            <div aria-hidden="true" className="absolute inset-x-0 bottom-0 h-[34%] overflow-hidden"><Smear src={backdrop} seam={0.5} /></div>
        </>
    );
    return (
        <>
            <div aria-hidden="true" className="absolute inset-0" style={{ background: GRADIENT[look] }} />
            {(look === 'ultra' || look === 'violet') && <Rays className="absolute inset-0 w-full h-full text-[#6f63ff]/60" />}
        </>
    );
};

const Tile = ({ c, films, total, onOpen }: {
    c: Collection; films: Movie[] | null; total?: number; onOpen: (c: Collection) => void;
}) => {
    const backdrop = img(films?.find(f => f.backdropPath)?.backdropPath, 'w780');
    const ink = c.look === 'paper' ? 'text-black' : 'text-white';
    // Sunset ends in cream, so its foot is set in black.
    const footInk = c.look === 'sunset' ? 'text-black' : ink;

    return (
        <button type="button" onClick={() => onOpen(c)}
            aria-label={`${title(c)}${total ? `, ${total.toLocaleString()} films` : ''}`}
            className="group relative shrink-0 snap-start overflow-hidden rounded-[14px] text-left
                       w-[264px] sm:w-[300px] lg:w-[320px] aspect-square flex flex-col p-4 [container-type:inline-size]">
            <Ground look={c.look} backdrop={backdrop} />

            <h3 className={cn('relative display', ink)} style={{ fontSize: `${c.size}cqw` }}>
                {c.lines.map(([text, indent]) => (
                    <span key={text} className="block" style={{ paddingLeft: `${indent * 100}%` }}>{text}</span>
                ))}
            </h3>

            <ol className={cn('relative mt-auto grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-[12px]',
                c.look === 'paper' ? 'text-black/80 mb-[37%]' : 'text-white/85', c.look === 'sunset' && 'mb-6')}
                style={{ marginLeft: c.look === 'ultra' ? '30%' : undefined }}>
                {(films ?? Array.from({ length: 4 }, () => null)).slice(0, 4).map((f, i) => f
                    ? <li key={f.item_id} className="contents">
                        <span className="truncate">{short(f.title)}</span>
                        <span className="tabular-nums">{f.year}</span>
                    </li>
                    : <li key={i} className="contents" aria-hidden="true">
                        <span className="h-3 my-0.5 rounded bg-white/15" /><span className="w-8 h-3 my-0.5 rounded bg-white/15" />
                    </li>)}
            </ol>

            <span className={cn('relative mt-3 flex items-end justify-between text-[12px]',
                c.look === 'paper' ? 'text-white' : footInk)}>
                <span className="tabular-nums">{total != null ? `${total.toLocaleString()} films` : ' '}</span>
                <ArrowDownRight className="w-6 h-6 transition-transform group-hover:translate-x-0.5 group-hover:translate-y-0.5" strokeWidth={1.25} />
            </span>
        </button>
    );
};

/** The collections panel: one search per tile, all fired together. */
const Collections = ({ onOpen }: { onOpen: (c: Collection, backdrop?: string) => void }) => {
    const [data, setData] = useState<Record<string, { films: Movie[]; total: number }>>({});

    useEffect(() => {
        const ctrl = new AbortController();
        COLLECTIONS.forEach(c => {
            searchFilms(c.q ?? '', filtersOf(c), { limit: 4, facets: false, signal: ctrl.signal })
                .then(r => setData(d => ({ ...d, [c.id]: { films: r.results, total: r.total } })))
                .catch(() => { /* the tile keeps its cover and stays clickable */ });
        });
        return () => ctrl.abort();
    }, []);

    return (
        <Shelf title="Collections" aside="Category searches, set as covers">
            {COLLECTIONS.map(c => (
                <Tile key={c.id} c={c} films={data[c.id]?.films ?? null} total={data[c.id]?.total}
                    onOpen={(col) => onOpen(col, img(data[c.id]?.films.find(f => f.backdropPath)?.backdropPath, 'w1280'))} />
            ))}
        </Shelf>
    );
};

export default Collections;
