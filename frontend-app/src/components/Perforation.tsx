import { motion } from 'framer-motion';

/**
 * Similarity as film perforations rather than a progress bar: eight sprocket
 * holes, lit in proportion to the strength of the match.
 */
export default function Perforation({ score, label }: { score: number; label?: string }) {
  const lit = Math.max(1, Math.round(score * 8));
  return (
    <div className="flex items-center gap-2.5">
      <div className="perf" role="img"
           aria-label={`Match strength ${Math.round(score * 100)} out of 100`}>
        {Array.from({ length: 8 }, (_, i) => (
          <motion.i key={i} data-on={i < lit ? '1' : '0'}
            initial={{ scaleY: .4, opacity: 0 }}
            animate={{ scaleY: 1, opacity: 1 }}
            transition={{ delay: i * .035, duration: .3, ease: [0.16, 1, 0.3, 1] }} />
        ))}
      </div>
      {label !== undefined && (
        <span className="truncate text-[.7rem]" style={{ color: 'var(--halide-dim)' }}>{label}</span>
      )}
    </div>
  );
}
