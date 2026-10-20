/** The small superscript asterisk that follows a rating. */
export const Asterisk = ({ className }: { className?: string }) => (
    <svg viewBox="0 0 12 12" aria-hidden="true" className={className} fill="none"
        stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
        <path d="M6 1v10M1.7 3.5l8.6 5M1.7 8.5l8.6-5" />
    </svg>
);

/**
 * The CINEMIND mark: the match stage in miniature — one film standing tall,
 * a closer match on either side. The side cards are geometry shared with the
 * loader, which slides them out from behind the centre.
 */
export const MARK = {
    side: { w: 6, h: 14, y: 5, rx: 1.6 },
    centre: { x: 7.5, w: 9, h: 20, y: 2, rx: 2.2 },
    left: 1, right: 17,
} as const;

export const Mark = ({ className }: { className?: string }) => (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className} fill="currentColor">
        <rect x={MARK.left} y={MARK.side.y} width={MARK.side.w} height={MARK.side.h} rx={MARK.side.rx} opacity=".45" />
        <rect x={MARK.right} y={MARK.side.y} width={MARK.side.w} height={MARK.side.h} rx={MARK.side.rx} opacity=".45" />
        <rect x={MARK.centre.x} y={MARK.centre.y} width={MARK.centre.w} height={MARK.centre.h} rx={MARK.centre.rx} />
    </svg>
);

/**
 * Thin lines thrown from a vanishing point, laid over the gradient tiles.
 * Purely decorative, so it stays out of the accessibility tree.
 */
export const Rays = ({ className, origin = [0.62, 0.35] }: { className?: string; origin?: [number, number] }) => {
    const [ox, oy] = [origin[0] * 100, origin[1] * 100];
    const ends = [[-10, 112], [18, 112], [46, 112], [84, 112], [112, 104], [112, 64], [112, 22], [112, -8], [-10, 76], [-10, 30]];
    return (
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true" className={className}>
            {ends.map(([x, y], i) => (
                <line key={i} x1={ox} y1={oy} x2={x} y2={y}
                    stroke="currentColor" strokeWidth="0.25" vectorEffect="non-scaling-stroke" />
            ))}
        </svg>
    );
};
