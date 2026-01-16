import { motion } from 'framer-motion';
import { Database, FileText, Cpu, Layers, Zap } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';

const FlowNode = ({ icon: Icon, label, details, delay, isActive, isCompleted }: any) => (
    <motion.div
        className={`relative p-4 rounded-xl border ${isActive ? 'border-primary bg-primary/10 shadow-[0_0_15px_rgba(220,38,38,0.3)]' : isCompleted ? 'border-primary/50 bg-primary/5' : 'border-white/10 bg-neutral-900/80'} backdrop-blur-md flex flex-col items-center gap-3 w-32 md:w-40 z-10 transition-colors duration-500`}
        initial={{ opacity: 0, scale: 0.8 }}
        animate={{ opacity: 1, scale: isActive ? 1.05 : 1 }}
        transition={{ delay, duration: 0.5 }}
        whileHover={{ scale: 1.05, borderColor: 'rgba(220,38,38,0.8)' }}
    >
        <div className={`p-3 rounded-lg ${isActive ? 'bg-primary text-white' : isCompleted ? 'bg-primary/50 text-white/50' : 'bg-white/5 text-gray-400'} transition-colors duration-500`}>
            <Icon size={24} />
        </div>
        <div className="text-center">
            <h4 className={`text-sm font-bold ${isActive ? 'text-white' : 'text-gray-200'}`}>{label}</h4>
            <p className="text-[10px] text-gray-300 mt-1 leading-tight font-medium">{details}</p>
        </div>

        {/* Active Ping Effect */}
        {isActive && (
            <motion.div
                className="absolute inset-0 rounded-xl border border-primary"
                initial={{ scale: 1, opacity: 1 }}
                animate={{ scale: 1.2, opacity: 0 }}
                transition={{ duration: 1, repeat: Infinity }}
            />
        )}
    </motion.div>
);

const ConnectionLine = ({ activeStage, stageIndex }: { activeStage: number, stageIndex: number }) => {
    const isActive = activeStage > stageIndex;
    const isTraversing = activeStage === stageIndex;

    return (
        <div className="h-full w-1 md:w-16 md:h-1 bg-white/5 relative overflow-hidden self-center">
            <motion.div
                className="absolute inset-0 bg-gradient-to-r from-primary/50 to-primary"
                initial={{ x: '-100%' }}
                animate={{ x: isActive ? '0%' : isTraversing ? ['-100%', '100%'] : '-100%' }}
                transition={isTraversing ? { duration: 1.5, ease: "linear", repeat: Infinity } : { duration: 0.5 }}
            />
        </div>
    );
};

// Mobile Vertical Connection
const VerticalConnection = ({ activeStage, stageIndex }: { activeStage: number, stageIndex: number }) => {
    const isActive = activeStage > stageIndex;
    const isTraversing = activeStage === stageIndex;
    return (
        <div className="w-1 h-8 md:hidden bg-white/5 relative overflow-hidden self-center my-1">
            <motion.div
                className="absolute inset-0 bg-gradient-to-b from-primary/50 to-primary"
                initial={{ y: '-100%' }}
                animate={{ y: isActive ? '0%' : isTraversing ? ['-100%', '100%'] : '-100%' }}
                transition={isTraversing ? { duration: 1.5, ease: "linear", repeat: Infinity } : { duration: 0.5 }}
            />
        </div>
    )
}

const ProjectFlowchart = () => {
    const [activeStage, setActiveStage] = useState(0);
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const interval = setInterval(() => {
            setActiveStage(prev => (prev >= 4 ? 0 : prev + 1));
        }, 2000);
        return () => clearInterval(interval);
    }, []);

    // Auto-scroll effect
    useEffect(() => {
        if (scrollContainerRef.current) {
            const container = scrollContainerRef.current;
            // The content is inside the first child (the flex container)
            // But we are scrolling the container itself which has overflow-x
            const wrapper = container.querySelector('div') as HTMLElement; // The inner flex wrapper
            if (!wrapper) return;

            // We need to target the specific stage element within the wrapper.
            // Since there are 5 stages, we can likely find it by index.
            const stageElements = wrapper.querySelectorAll('.snap-center');
            const targetElement = stageElements[activeStage] as HTMLElement;

            if (targetElement) {
                // Determine center position
                const containerWidth = container.clientWidth;
                const elementWidth = targetElement.offsetWidth;
                const elementLeft = targetElement.offsetLeft;

                // Calculate the scroll position needed to center the element
                // elementLeft is relative to the wrapper, but since wrapper is the first child
                // and container scrolls, we might need to adjust if wrapper has logic.
                // Actually, if 'container' is the one with overflow-x-auto, then:

                const scrollPos = elementLeft - (containerWidth / 2) + (elementWidth / 2);

                container.scrollTo({
                    left: scrollPos,
                    behavior: 'smooth'
                });
            }
        }
    }, [activeStage]);

    const stages = [
        { icon: Database, label: "TMDB Data", details: "Raw Movies & Metadata" },
        { icon: FileText, label: "Preprocessing", details: "Cleaning & Normalization" },
        { icon: Cpu, label: "Embedding", details: "Sentence-BERT (384d)" },
        { icon: Layers, label: "FAISS Index", details: "Vector Similarity Search" },
        { icon: Zap, label: "Inference", details: "Real-time Recommendations" }
    ];

    return (
        <div className="w-full bg-black/40 border border-white/10 rounded-3xl p-8 backdrop-blur-sm relative overflow-hidden group">
            {/* Background Grid */}
            <div className="absolute inset-0 opacity-10"
                style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '40px 40px' }}
            />

            <div
                ref={scrollContainerRef}
                className="relative z-10 flex flex-col md:flex-row items-center justify-start md:justify-between gap-4 md:gap-2 overflow-x-auto md:overflow-visible pb-4 md:pb-0 scrollbar-hide snap-x"
            >
                <div className="flex md:contents w-max md:w-full gap-4 md:gap-0 px-4 md:px-0">
                    {stages.map((stage, index) => (
                        <div key={index} className="contents md:flex md:items-center snap-center">
                            <FlowNode
                                {...stage}
                                delay={index * 0.1}
                                isActive={activeStage === index}
                                isCompleted={activeStage > index}
                            />
                            {index < stages.length - 1 && (
                                <>
                                    <div className="hidden md:block">
                                        <ConnectionLine activeStage={activeStage} stageIndex={index} />
                                    </div>
                                    <VerticalConnection activeStage={activeStage} stageIndex={index} />
                                </>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            <div className="mt-8 text-center">
                <p className="text-xs text-gray-500 font-mono">
                    <span className="text-primary mr-2">●</span>
                    Pipeline Status: {activeStage < 5 ? 'PROCESSING...' : 'COMPLETE'}
                    <span className="ml-4 opacity-50">Latency: 45ms</span>
                </p>
            </div>
        </div>
    );
};

export default ProjectFlowchart;
