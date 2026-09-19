import { motion } from 'framer-motion';
import { cn } from '../lib/utils';

interface GlassSkeletonProps {
    className?: string;
}

const GlassSkeleton = ({ className }: GlassSkeletonProps) => {
    return (
        <div className={cn("relative overflow-hidden bg-white/5 backdrop-blur-sm border border-white/5", className)}>
            <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent skew-x-12"
                initial={{ x: '-150%' }}
                animate={{ x: '150%' }}
                transition={{
                    repeat: Infinity,
                    duration: 1.2,
                    ease: "easeInOut",
                    repeatDelay: 0.1
                }}
            />

            {/* Subtle pulsing glow */}
            <motion.div
                className="absolute inset-0 bg-primary/10 opacity-0"
                animate={{ opacity: [0, 0.2, 0] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
            />
        </div>
    );
};

export default GlassSkeleton;
