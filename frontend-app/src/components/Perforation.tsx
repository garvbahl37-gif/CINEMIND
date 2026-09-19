/**
 * Similarity rendered as film perforations rather than a progress bar.
 * Eight sprocket holes, filled in proportion to how strong the match is.
 */
export default function Perforation({ score, label }: { score: number; label?: string }) {
  const lit = Math.max(1, Math.round(score * 8));
  return (
    <div className="flex items-center gap-3">
      <div className="perf" role="img"
           aria-label={`Match strength ${Math.round(score * 100)} out of 100`}>
        {Array.from({ length: 8 }, (_, i) => (
          <i key={i} data-on={i < lit ? '1' : '0'} />
        ))}
      </div>
      {label !== undefined && <span className="machine">{label}</span>}
    </div>
  );
}
