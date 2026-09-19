import { motion } from 'framer-motion';
/**
 * The pipeline really is a sequence, so numbering it is information rather than
 * decoration. Numbers are set as frame counts, in the machine's colour.
 */
const STAGES = [
  {
    n: '01', title: 'Learn from 32 million ratings',
    body: 'Every film is scored against every other by how often the same viewers rated both highly. ' +
          'Power users are damped so no single account dominates, and blockbusters are damped so ' +
          'popularity alone never counts as similarity.',
  },
  {
    n: '02', title: 'Place each film in 64 dimensions',
    body: 'A two-tower neural network, trained with InfoNCE and in-batch negatives, maps 17,719 films ' +
          'into a shared embedding space. Distance in that space is a learned measure of taste.',
  },
  {
    n: '03', title: 'Rerank against what the films are about',
    body: 'Raw co-occurrence drifts toward whatever else was popular that year. Genre and tag overlap ' +
          'pull the ranking back toward the film itself, rare genres count for more than common ones, ' +
          'and titles with thin rating data are held back.',
  },
  {
    n: '04', title: 'Precompute, then serve',
    body: 'Every ranking is computed ahead of time and stored, so a request is an array lookup rather ' +
          'than a search. That is why results return in single-digit milliseconds.',
  },
];

export default function HowItWorks() {
  return (
    <div className="pt-32 pb-24" style={{ paddingInline: 'var(--gut)' }}>
      <div className="mx-auto max-w-[760px]">
        <h1 style={{ fontSize: 'clamp(2.2rem, 5vw, var(--t-3xl))' }}>
          How Cinemind decides
        </h1>
        <p className="mt-6 max-w-[62ch] text-[1rem]"
           style={{ color: 'var(--halide-mid)', lineHeight: 1.75 }}>
          Most recommenders answer “what is popular near this?”. This one tries to answer
          “what is <em>like</em> this?” — which are different questions, and the difference
          is where the interesting films live.
        </p>

        <ol className="mt-16 space-y-12">
          {STAGES.map((s, i) => (
            <motion.li key={s.n} className="grid gap-5 sm:grid-cols-[62px_1fr]"
              initial={{ opacity: 0, y: 26 }} whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-80px' }}
              transition={{ duration: .6, delay: i * .08, ease: [0.16, 1, 0.3, 1] }}>
              <div className="machine pt-1" style={{ fontSize: '1rem' }}>{s.n}</div>
              <div>
                <h2 style={{ fontSize: 'var(--t-md)' }}>{s.title}</h2>
                <p className="mt-2.5 max-w-[58ch] text-[0.95rem]"
                   style={{ color: 'var(--halide-mid)', lineHeight: 1.72 }}>{s.body}</p>
              </div>
            </motion.li>
          ))}
        </ol>

        <div className="glass mt-20 p-8">
          <h2 style={{ fontSize: 'var(--t-md)' }}>What it is built on</h2>
          <dl className="mt-5 grid gap-x-10 gap-y-4 sm:grid-cols-2">
            {[
              ['Data', 'MovieLens 32M — 32,000,204 ratings'],
              ['Catalogue', '17,719 films with full metadata'],
              ['Model', 'Two-tower dual encoder, 64-d output'],
              ['Serving', 'FastAPI on Vercel Functions, NumPy only'],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="text-[0.75rem]" style={{ color: 'var(--halide-dim)' }}>{k}</dt>
                <dd className="mt-1 text-[0.9rem]">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </div>
  );
}
