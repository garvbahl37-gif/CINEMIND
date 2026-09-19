import ProjectFlowchart from './ProjectFlowchart';
import { motion } from 'framer-motion';
import {
    Layout, Server, Workflow, Brain, Layers, Database
} from 'lucide-react';

const TechCard = ({ icon: Icon, title, desc, delay }: { icon: any, title: string, desc: string, delay: number }) => (
    <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay, duration: 0.5 }}
        className="bg-white/5 backdrop-blur-md border border-white/10 p-6 rounded-2xl hover:bg-white/10 transition-colors group h-full"
    >
        <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center mb-4 text-primary group-hover:scale-110 transition-transform">
            <Icon size={24} />
        </div>
        <h3 className="text-xl font-bold text-white mb-2">{title}</h3>
        <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
    </motion.div>
);

const ProcessStep = ({ number, title, desc, delay }: { number: string, title: string, desc: string, delay: number }) => (
    <motion.div
        initial={{ opacity: 0, x: -20 }}
        whileInView={{ opacity: 1, x: 0 }}
        viewport={{ once: true }}
        transition={{ delay, duration: 0.5 }}
        className="flex gap-6 items-start relative"
    >
        <div className="hidden md:flex flex-col items-center">
            <div className="w-10 h-10 rounded-full bg-primary/20 border border-primary/50 text-primary font-bold flex items-center justify-center shrink-0 z-10">
                {number}
            </div>
            <div className="w-0.5 h-full bg-white/10 absolute top-10 left-5 -z-10" />
        </div>
        <div className="bg-white/5 border border-white/10 p-6 rounded-2xl flex-1 backdrop-blur-sm hover:border-primary/30 transition-colors">
            <h4 className="text-lg font-bold text-white mb-2">{title}</h4>
            <p className="text-gray-400 text-sm leading-relaxed">{desc}</p>
        </div>
    </motion.div>
);

const AboutPage = () => {
    return (
        <div className="min-h-screen text-white pt-24 px-4 md:px-12 relative overflow-hidden pb-20">

            <motion.div
                className="max-w-7xl mx-auto relative z-10"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.8 }}
            >
                {/* Header Section */}
                <div className="text-center mb-24 mt-10">
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.8 }}
                        className="inline-block"
                    >
                        <h1 className="text-5xl md:text-8xl font-display font-black tracking-tight text-transparent bg-clip-text bg-gradient-to-b from-white via-gray-200 to-gray-500 mb-6 drop-shadow-2xl">
                            Inside The System
                        </h1>
                    </motion.div>
                    <motion.p
                        className="text-xl text-gray-400 max-w-3xl mx-auto font-light leading-relaxed"
                        initial={{ y: 20, opacity: 0 }}
                        animate={{ y: 0, opacity: 1 }}
                        transition={{ delay: 0.3 }}
                    >
                        A deep dive into the architecture of a next-generation recommendation engine.
                        From raw data to hyper-personalized results using vector semantics.
                    </motion.p>
                </div>

                {/* Architecture & Workflow */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 mb-32 items-start">

                    {/* Left: The Pipeline */}
                    <div className="space-y-8">
                        <motion.div
                            initial={{ opacity: 0, x: -20 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                        >
                            <h2 className="text-3xl font-bold text-white mb-8 flex items-center gap-3">
                                <Workflow className="text-primary" />
                                The Recommendation Pipeline
                            </h2>
                        </motion.div>

                        <div className="space-y-8">
                            <ProcessStep
                                number="01"
                                title="Data Ingestion & Cleaning"
                                desc="The system ingests raw metadata from TMDB. Scripts clean this data, normalizing genres, extracting keywords, and formatting it for the embedding model, ensuring high-quality input signals."
                                delay={0.1}
                            />
                            <ProcessStep
                                number="02"
                                title="Vector Embedding Generation"
                                desc="We use sentence-transformers (all-MiniLM-L6-v2) to convert movie descriptions and tags into high-dimensional vector space (384 dimensions). This captures the semantic 'meaning' of a movie beyond simple keywords."
                                delay={0.2}
                            />
                            <ProcessStep
                                number="03"
                                title="FAISS Indexing"
                                desc="These vectors are indexed using FAISS (Facebook AI Similarity Search). We utilize an IVF (Inverted File) index structure to perform lightning-fast nearest neighbor searches across thousands of movies in milliseconds."
                                delay={0.3}
                            />
                            <ProcessStep
                                number="04"
                                title="Diversity Reranking"
                                desc="Raw similarity results can be repetitive. We apply a post-processing reranking algorithm (MMR - Maximal Marginal Relevance) to inject diversity, ensuring recommendations cover a mix of genres and styles while remaining relevant."
                                delay={0.4}
                            />
                        </div>
                    </div>

                    {/* Right: Animated System Architecture */}
                    <div className="space-y-8 sticky top-32">
                        <div className="bg-neutral-900/50 backdrop-blur-xl border border-white/10 rounded-3xl p-8 shadow-2xl relative overflow-hidden">
                            <div className="absolute top-0 right-0 p-32 bg-primary/10 rounded-full blur-[100px] pointer-events-none" />

                            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
                                <Brain className="text-primary" />
                                System Pipeline
                            </h3>

                            {/* Animated Diagram */}
                            <ProjectFlowchart />

                            <p className="text-gray-400 text-sm mt-4 italic text-center">
                                Real-time visualization of the recommendation data flow.
                            </p>
                        </div>
                    </div>
                </div>

                {/* Tech Stack Grid */}
                <div>
                    <motion.h2
                        className="text-3xl font-bold text-white text-center mb-12"
                        initial={{ opacity: 0, y: 20 }}
                        whileInView={{ opacity: 1, y: 0 }}
                        viewport={{ once: true }}
                    >
                        Core Technologies
                    </motion.h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <TechCard
                            icon={Layers}
                            title="FAISS"
                            desc="Facebook's library for efficient similarity search and clustering of dense vectors."
                            delay={0.1}
                        />
                        <TechCard
                            icon={Brain}
                            title="Sentence Transformers"
                            desc="HuggingFace models (all-MiniLM) to generate state-of-the-art semantic embeddings."
                            delay={0.2}
                        />
                        <TechCard
                            icon={Server}
                            title="Python & NumPy"
                            desc="The backbone of our data engineering and numerical processing pipeline."
                            delay={0.3}
                        />
                        <TechCard
                            icon={Layout}
                            title="React Ecosystem"
                            desc="Vite, TypeScript, TailwindCSS, and Framer Motion for a premium, type-safe UX."
                            delay={0.4}
                        />
                        <TechCard
                            icon={Database}
                            title="MovieLens 32M"
                            desc="Powered by the massive MovieLens 32M dataset: 32 million ratings and 1M tags to fuel accurate predictions."
                            delay={0.5}
                        />
                    </div>
                </div>

            </motion.div>
        </div>
    );
};

export default AboutPage;
