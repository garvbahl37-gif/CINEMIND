/**
 * Grain, vignette and slow ambient light. Purely atmospheric, never interactive:
 * everything here is pointer-events:none and sits outside the tab order.
 */
export default function Atmosphere() {
  return (
    <>
      <div className="ambient" aria-hidden="true">
        <span style={{ width: '46vw', height: '46vw', left: '-10vw', top: '-8vh',
                       background: 'radial-gradient(circle, rgba(232,53,74,.34), transparent 68%)' }} />
        <span style={{ width: '40vw', height: '40vw', right: '-8vw', top: '18vh',
                       background: 'radial-gradient(circle, rgba(127,227,212,.13), transparent 68%)',
                       animationDelay: '-9s' }} />
        <span style={{ width: '52vw', height: '52vw', left: '22vw', bottom: '-22vh',
                       background: 'radial-gradient(circle, rgba(142,21,38,.34), transparent 70%)',
                       animationDelay: '-16s' }} />
      </div>

      <svg className="grain" aria-hidden="true">
        <filter id="cm-grain">
          <feTurbulence type="fractalNoise" baseFrequency="0.85" numOctaves="3" stitchTiles="stitch" />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="100%" height="100%" filter="url(#cm-grain)" />
      </svg>

      <div className="vignette" aria-hidden="true" />
    </>
  );
}
