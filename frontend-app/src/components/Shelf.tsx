import { ReactNode, useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '../lib/utils';

interface Props {
    title: ReactNode;
    /** Shown after the title in a lighter weight, e.g. a film count. */
    aside?: ReactNode;
    /** Extra controls placed before "Show all" — a toggle, typically. */
    controls?: ReactNode;
    onShowAll?: () => void;
    showAllLabel?: string;
    children: ReactNode;
    className?: string;
    /** Set false for panels whose body is not a horizontal row. */
    scroll?: boolean;
    id?: string;
}

/**
 * A dashboard panel: a heading set large in upper case, its controls on the
 * right, and a row that scrolls sideways under it.
 */
const Shelf = ({ title, aside, controls, onShowAll, showAllLabel = 'Show all', children, className, scroll = true, id }: Props) => {
    const row = useRef<HTMLDivElement>(null);
    const [edges, setEdges] = useState({ start: true, end: false });

    const measure = useCallback(() => {
        const el = row.current;
        if (!el) return;
        setEdges({
            start: el.scrollLeft <= 4,
            end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 4,
        });
    }, []);

    useEffect(() => {
        measure();
        const el = row.current;
        if (!el) return;
        const ro = new ResizeObserver(measure);
        ro.observe(el);
        return () => ro.disconnect();
    }, [measure, children]);

    const page = (dir: 1 | -1) => {
        const el = row.current;
        if (el) el.scrollBy({ left: dir * el.clientWidth * 0.85, behavior: 'smooth' });
    };

    return (
        <section id={id} className={cn('bg-panel rounded-[24px] py-5 md:py-7', className)}>
            <header className="flex flex-wrap items-end gap-x-6 gap-y-3 px-5 md:px-7 mb-5">
                <h2 className="display text-[30px] md:text-[44px] mr-auto">
                    {title}
                    {aside && <span className="ml-3 text-[15px] md:text-[17px] font-normal normal-case tracking-normal text-mute align-middle">{aside}</span>}
                </h2>
                <div className="flex items-center gap-2">
                    {controls}
                    {onShowAll && (
                        <button type="button" onClick={onShowAll} className="pill pill-ghost">
                            {showAllLabel} <ArrowUpRight className="w-3.5 h-3.5" />
                        </button>
                    )}
                    {scroll && (
                        <span className="hidden sm:flex gap-1.5 ml-1">
                            <button type="button" className="round-btn" aria-label="Scroll back"
                                disabled={edges.start} onClick={() => page(-1)}>
                                <ChevronLeft className="w-4 h-4" />
                            </button>
                            <button type="button" className="round-btn" aria-label="Scroll on"
                                disabled={edges.end} onClick={() => page(1)}>
                                <ChevronRight className="w-4 h-4" />
                            </button>
                        </span>
                    )}
                </div>
            </header>

            {scroll ? (
                <div ref={row} onScroll={measure}
                    className="scroller gap-3 px-5 md:px-7 scroll-px-5 md:scroll-px-7">
                    {children}
                </div>
            ) : children}
        </section>
    );
};

/** The reference's "My services" switch, here for "use my blend". */
export const Toggle = ({ label, on, onChange, disabled, hint }: {
    label: string; on: boolean; onChange: (v: boolean) => void; disabled?: boolean; hint?: string;
}) => (
    <label className={cn('flex items-center gap-2.5 mr-2 text-[13px] select-none',
        disabled ? 'text-white/35 cursor-not-allowed' : 'text-white/80 cursor-pointer')}
        title={disabled ? hint : undefined}>
        {label}
        <button type="button" role="switch" aria-checked={on} aria-label={label}
            disabled={disabled} onClick={() => onChange(!on)}
            className={cn('relative w-11 h-6 rounded-full border transition-colors',
                on ? 'bg-white border-white' : 'bg-transparent border-white/35')}>
            <span className={cn('absolute top-[3px] w-4 h-4 rounded-full transition-[left,background-color]',
                on ? 'left-[23px] bg-ultra' : 'left-[3px] bg-white/60')} />
        </button>
    </label>
);

export default Shelf;
