import { motion } from 'framer-motion';

const GlassLoader = () => {
    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black overflow-hidden">
            {/* Simple Background Gradient */}
            <div className="absolute inset-0 z-0 bg-gradient-to-br from-neutral-900 to-black" />

            <div className="relative z-10 flex flex-col items-center justify-center gap-12">

                {/* Logo Reveal */}
                <motion.div
                    initial={{ opacity: 0, scale: 0.9, y: 20 }}
                    animate={{ opacity: 1, scale: 1, y: 0 }}
                    transition={{ duration: 1.2, ease: "easeOut" }}
                    className="flex flex-col items-center"
                >
                    <div className="relative mb-8">
                        <h1 className="text-6xl md:text-9xl font-display font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-b from-white via-gray-200 to-gray-500 drop-shadow-2xl select-none relative z-10 flex items-center gap-4">
                            <span className="bg-gradient-to-b from-white via-gray-300 to-gray-500 bg-clip-text text-transparent filter drop-shadow-[0_2px_2px_rgba(0,0,0,0.8)]">CINE</span>
                            <span className="bg-gradient-to-b from-primary via-red-500 to-red-900 bg-clip-text text-transparent filter drop-shadow-[0_0_20px_rgba(220,38,38,0.6)]">MIND</span>
                        </h1>
                    </div>

                    {/* Futuristic Loading Bar */}
                    <div className="w-[300px] h-1 bg-white/10 rounded-full overflow-hidden relative">
                        <motion.div
                            className="absolute inset-0 bg-gradient-to-r from-transparent via-primary to-transparent w-1/2"
                            animate={{ x: ['-100%', '200%'] }}
                            transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
                        />
                        <div className="absolute inset-0 bg-primary/20 blur-[2px]" />
                    </div>
                </motion.div>

                {/* Status Text */}
                <div className="flex flex-col items-center gap-2">
                    <p className="text-[10px] font-mono font-medium text-primary/80 tracking-[0.5em] uppercase animate-pulse">
                        INITIALIZING SYSTEM
                    </p>
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.5, duration: 1 }}
                        className="text-[9px] text-gray-500 font-mono tracking-widest"
                    >
                        ESTABLISHING SECURE CONNECTION...
                    </motion.div>
                </div>
            </div>
        </div>
    );
};

export default GlassLoader;
