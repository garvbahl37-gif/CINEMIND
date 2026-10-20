/** 142 → "2h 22m". */
export const runtime = (min?: number) => {
    if (!min) return undefined;
    const h = Math.floor(min / 60), m = min % 60;
    return h ? (m ? `${h}h ${m}m` : `${h}h`) : `${m}m`;
};

/** Match strength as a whole percentage: 0.946 → 95. */
export const pct = (score?: number) => (score == null ? undefined : Math.round(score * 100));

/** MovieLens average, to one decimal. */
export const stars = (r?: number) => (r == null ? undefined : r.toFixed(1));

/**
 * MovieLens appends alternate titles in brackets — "Old Boy", but also
 * "Memories of Murder (Salinui chueok)". Cards show the first one; the full
 * title stays in the detail view and the accessible label.
 */
export const short = (title: string) => {
    const t = title.replace(/\s*\(.*\)\s*$/, '') || title;
    // MovieLens files some titles article-last: "Postman, The" is "The Postman".
    const m = t.match(/^(.*), (The|A|An)$/);
    return m ? `${m[2]} ${m[1]}` : t;
};

/** ["Frank Darabont"] → "Frank Darabont"; three or more become "A, B and C". */
export const names = (list?: string[], max = 3) => {
    const l = (list ?? []).slice(0, max);
    return l.length < 2 ? l[0] : `${l.slice(0, -1).join(', ')} and ${l[l.length - 1]}`;
};
