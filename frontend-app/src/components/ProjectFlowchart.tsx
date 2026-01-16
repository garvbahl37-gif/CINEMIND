import { motion } from 'framer-motion';
import { Database, FileText, Cpu, Layers, Zap, ArrowRight } from 'lucide-react';
import { useState, useEffect, useRef } from 'react';

const FlowNode = ({ icon: Icon, label, details, delay, isActive, isCompleted, id }: any) => (
    <motion.div
        id={id}
        className={`relative p-6 rounded-2xl border ${isActive ? 'border-primary bg-primary/10 shadow-[0_0_20px_rgba(220,38,38,0.2)]' : isCompleted ? 'border-primary/30 bg-primary/5' : 'border-white/10 bg-neutral-900/40'} backdrop-blur-xl flex flex-col items-center gap-4 min-w-[160px] md:w-48 z-10 transition-all duration-500`}
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: isActive ? 1.05 : 1 }}
        transition={{ delay, duration: 0.5 }}
    >
        <div className={`p-4 rounded-xl ${isActive ? 'bg-primary text-white' : isCompleted ? 'bg-primary/50 text-white/50' : 'bg-white/5 text-gray-400'} transition-colors duration-500 shadow-lg`}>
            <Icon size={28} />
        </div>
        <div className="text-center">
            <h4 className={`text-base font-bold ${isActive ? 'text-white' : 'text-gray-300'} mb-1`}>{label}</h4>
            <p className="text-[11px] text-gray-400 leading-tight font-medium max-w-[120px] mx-auto">{details}</p>
        </div>

        {/* Active Ping Effect */}
        {isActive && (
            <motion.div
                className="absolute inset-0 rounded-2xl border border-primary"
                initial={{ scale: 1, opacity: 1 }}
                animate={{ scale: 1.15, opacity: 0 }}
                transition={{ duration: 1.5, repeat: Infinity }}
            />
        )}
    </motion.div>
);       

const ConnectionLine = ({ activeStage, stageIndex }: { activeStage: number, stageIndex: number }) => {
    const isActive = activeStage > stageIndex;
    const isTraversing = activeStage === stageIndex;

    return (
        <div className="hidden md:flex flex-1 h-1 bg-white/5 relative overflow-hidden items-center justify-center mx-2">
            <motion.div
                className="absolute inset-0 bg-gradient-to-r from-primary/50 to-primary"
                initial={{ x: '-100%' }}
                animate={{ x: isActive ? '0%' : isTraversing ? ['-100%', '100%'] : '-100%' }}
                transition={isTraversing ? { duration: 1.5, ease: "linear", repeat: Infinity } : { duration: 0.5 }}
            />
            {/* Arrow Head */}
            <div className={`absolute right-0 text-white/10 ${isActive || isTraversing ? 'text-primary' : ''}`}>
                <ArrowRight size={16} />
            </div>
        </div>
    );
};

const ProjectFlowchart = () => {
    const [activeStage, setActiveStage] = useState(0);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const stagesRef = useRef<(HTMLDivElement | null)[]>([]);

    useEffect(() => {
        const interval = setInterval(() => {
            setActiveStage(prev => (prev >= 4 ? 0 : prev + 1));
        }, 2500); // Slightly slower for better readability
        return () => clearInterval(interval);
    }, []);

    // Robust Auto-scroll using scrollIntoView
    useEffect(() => {
        const activeElement = stagesRef.current[activeStage];
        if (activeElement) {
            activeElement.scrollIntoView({
                behavior: 'smooth',
                block: 'nearest',
                inline: 'center'
            });
        }
    }, [activeStage]);

    const stages = [
        { icon: Database, label: "32M Rating Data", details: "Raw Movies & Metadata" },
        { icon: FileText, label: "Preprocessing", details: "Cleaning & Normalization" },
        { icon: Cpu, label: "Embedding", details: "Sentence-BERT (384d)" },
        { icon: Layers, label: "FAISS Index", details: "Vector Similarity Search" },
        { icon: Zap, label: "Inference", details: "Real-time Recommendations" }
    ];

    return (
        <div className="w-full bg-black/40 border border-white/10 rounded-3xl p-6 md:p-10 backdrop-blur-sm relative overflow-hidden group">
            {/* Background Grid */}
            <div className="absolute inset-0 opacity-10"
                style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '40px 40px' }}
            />

            <div className="relative z-10">
                {/* Header */}
                <div className="flex items-center justify-between mb-8 px-2">
                    <h3 className="text-lg font-semibold text-white/80">System Architecture</h3>
                    <div className="flex items-center gap-2 text-xs text-primary font-mono bg-primary/10 px-3 py-1 rounded-full border border-primary/20">
                        <span className="relative flex h-2 w-2">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
                        </span>
                        Live Pipeline
                    </div>
                </div>

                {/* Pipeline Container */}
                <div
                    ref={scrollContainerRef}
                    className="flex items-center overflow-x-auto pb-8 md:pb-0 hide-scrollbar snap-x snap-mandatory"
                    style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
                >
                    <div className="flex items-center min-w-full px-4 md:px-0 gap-4 md:gap-0">
                        {stages.map((stage, index) => (
                            <div key={index} className="flex items-center snap-center shrink-0">
                                {/* Node */}
                                <div ref={el => stagesRef.current[index] = el}>
                                    <FlowNode
                                        {...stage}
                                        id={`stage-${index}`}
                                        delay={index * 0.1}
                                        isActive={activeStage === index}
                                        isCompleted={activeStage > index}
                                    />
                                </div>

                                {/* Connector (Desktop) */}
                                {index < stages.length - 1 && (
                                    <div className="w-12 md:w-24 hidden md:flex items-center justify-center">
                                        <ConnectionLine activeStage={activeStage} stageIndex={index} />
                                    </div>
                                )}

                                {/* Connector (Mobile) */}
                                {index < stages.length - 1 && (
                                    <div className="md:hidden flex items-center justify-center px-2 text-white/20">
                                        <ArrowRight size={20} />
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <div className="mt-4 flex justify-between items-center text-xs text-gray-500 font-mono border-t border-white/5 pt-4">
                <span>Latency: 45ms</span>
                <span>Status: {activeStage === stages.length - 1 ? 'COMPLETE' : 'PROCESSING...'}</span>
            </div>
        </div>
    );
};

export default ProjectFlowchart;
