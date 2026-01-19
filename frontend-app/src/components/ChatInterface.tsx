import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, X, Send, Loader2, Video, Sparkles, BrainCircuit } from 'lucide-react';
import { API_BASE, TMDB_IMAGE_BASE } from '../config';
import { cn } from '../lib/utils';
import MovieCard from './MovieCard';
import { Movie } from '../types';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    recommendations?: Movie[];
}

const AutoSlideCarousel = ({ movies }: { movies: Movie[] }) => {
    const [index, setIndex] = useState(0);

    useEffect(() => {
        if (movies.length <= 2) return;
        const interval = setInterval(() => {
            setIndex((prev) => (prev + 1) % Math.ceil(movies.length / 2));
        }, 4000);
        return () => clearInterval(interval);
    }, [movies.length]);

    const visibleMovies = [
        movies[(index * 2) % movies.length],
        movies[(index * 2 + 1) % movies.length]
    ].filter(Boolean);

    return (
        <div className="w-full relative h-[220px] overflow-hidden rounded-xl mt-2 select-none">
            <AnimatePresence mode="wait">
                <motion.div
                    key={index}
                    initial={{ opacity: 0, x: 50 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -50 }}
                    transition={{ duration: 0.5, ease: "easeInOut" }}
                    className="absolute inset-0 flex gap-3 px-2 justify-center items-center"
                >
                    {visibleMovies.map((movie) => (
                        <div key={`${index}-${movie.tmdbId}`} className="w-[120px] flex-shrink-0">
                            <MovieCard
                                movie={movie}
                                onSelect={() => { }}
                                compact={true}
                                className="shadow-lg"
                            />
                        </div>
                    ))}
                </motion.div>
            </AnimatePresence>

            {/* Progress Indicators */}
            <div className="absolute bottom-1 left-0 right-0 flex justify-center gap-1.5 z-10">
                {Array.from({ length: Math.ceil(movies.length / 2) }).map((_, i) => (
                    <div
                        key={i}
                        className={cn(
                            "w-1.5 h-1.5 rounded-full transition-colors duration-300",
                            i === index ? "bg-primary" : "bg-white/20"
                        )}
                    />
                ))}
            </div>
        </div>
    );
};

export const ChatInterface: React.FC = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId] = useState(() => Math.random().toString(36).substring(7));
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, isOpen]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMsg = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsLoading(true);

        try {
            const res = await fetch(`${API_BASE}/chat/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg, session_id: sessionId })
            });

            if (!res.ok) throw new Error('Failed to send message');

            const data = await res.json();

            // Map API response to Movie type
            const recommendations: Movie[] = (data.recommendations || []).map((rec: any) => ({
                tmdbId: rec.tmdbId,
                title: rec.title,
                poster: rec.poster_path ? `${TMDB_IMAGE_BASE}/w500${rec.poster_path}` : '',
                overview: rec.overview,
                vote_average: rec.vote_average,
                releaseDate: rec.year ? `${rec.year}-01-01` : undefined,
                genres: rec.genres
            }));

            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.response,
                recommendations: recommendations.length > 0 ? recommendations : undefined
            }]);
        } catch (error) {
            console.error('Chat Error:', error);
            let errorMessage = "I'm having trouble connecting to the CineMind core.";

            if (error instanceof Error) {
                errorMessage += ` (${error.message})`;
            }

            setMessages(prev => [...prev, { role: 'assistant', content: errorMessage }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="fixed bottom-8 right-8 z-[100] flex flex-col items-end pointer-events-none">
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.9, y: 20, transformOrigin: "bottom right" }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.9, y: 20 }}
                        transition={{ type: "spring", bounce: 0.2, duration: 0.5 }}
                        className="pointer-events-auto mb-6 w-[360px] md:w-[420px] h-[600px] bg-black/80 backdrop-blur-2xl border border-white/10 rounded-[2rem] shadow-[0_20px_50px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden ring-1 ring-white/5"
                    >
                        {/* Header */}
                        <div className="p-5 border-b border-white/10 flex justify-between items-center bg-gradient-to-r from-neutral-900/80 to-black/80">
                            <div className="flex items-center gap-3">
                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-primary to-rose-900 flex items-center justify-center shadow-[0_0_15px_rgba(229,9,20,0.4)] ring-1 ring-white/10">
                                    <BrainCircuit className="w-6 h-6 text-white" />
                                </div>
                                <div>
                                    <h3 className="font-['Orbitron'] font-bold text-white tracking-wider flex items-center gap-2">
                                        CINEMIND <span className="text-primary text-xs bg-primary/10 px-1.5 py-0.5 rounded border border-primary/20">AI</span>
                                    </h3>
                                    <p className="text-xs text-white/40 font-medium tracking-wide">ASSISTANT ACTIVE</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setIsOpen(false)}
                                className="p-2 hover:bg-white/5 rounded-full transition-colors group"
                            >
                                <X className="w-5 h-5 text-white/50 group-hover:text-white transition-colors" />
                            </button>
                        </div>

                        {/* Messages */}
                        <div className="flex-1 overflow-y-auto p-5 space-y-5 scrollbar-thin scrollbar-thumb-white/10 scrollbar-track-transparent">
                            {messages.length === 0 && (
                                <div className="h-full flex flex-col items-center justify-center text-center opacity-40 p-6">
                                    <Video className="w-12 h-12 mb-4 text-white/20" />
                                    <h4 className="font-['Orbitron'] text-sm font-bold text-white mb-2">READY TO ASSIST</h4>
                                    <p className="text-xs leading-relaxed max-w-[200px]">
                                        Ask me to recommend a movie based on your mood, genre preferences, or favorite actors.
                                    </p>
                                </div>
                            )}
                            {messages.map((msg, i) => (
                                <div
                                    key={i}
                                    className={`flex flex-col gap-3 ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
                                >
                                    {/* Text Bubble */}
                                    <div className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} w-full`}>
                                        <div
                                            className={cn(
                                                "max-w-[85%] p-4 text-sm leading-relaxed shadow-lg",
                                                msg.role === 'user'
                                                    ? "bg-primary text-white rounded-2xl rounded-tr-none font-medium"
                                                    : "bg-[#1a1a1a] text-gray-200 rounded-2xl rounded-tl-none border border-white/5"
                                            )}
                                        >
                                            {msg.content}
                                        </div>
                                    </div>

                                    {/* Auto-Slide Recommendations */}
                                    {msg.recommendations && msg.recommendations.length > 0 && (
                                        <AutoSlideCarousel movies={msg.recommendations} />
                                    )}
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start">
                                    <div className="bg-[#1a1a1a] border border-white/5 rounded-2xl rounded-tl-none p-4 flex items-center gap-3">
                                        <div className="flex gap-1">
                                            <motion.div
                                                animate={{ scale: [1, 1.2, 1] }}
                                                transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                                                className="w-1.5 h-1.5 bg-primary rounded-full"
                                            />
                                            <motion.div
                                                animate={{ scale: [1, 1.2, 1] }}
                                                transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                                                className="w-1.5 h-1.5 bg-primary rounded-full"
                                            />
                                            <motion.div
                                                animate={{ scale: [1, 1.2, 1] }}
                                                transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                                                className="w-1.5 h-1.5 bg-primary rounded-full"
                                            />
                                        </div>
                                        <span className="text-xs text-white/40 font-['Orbitron'] tracking-widest">PROCESSING</span>
                                    </div>
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        {/* Input */}
                        <form onSubmit={handleSubmit} className="p-4 bg-black/60 border-t border-white/10 backdrop-blur-md">
                            <div className="relative flex items-center group">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    placeholder="Type your request..."
                                    className="w-full bg-[#111] border border-white/10 rounded-xl pl-5 pr-12 py-4 text-sm text-white focus:outline-none focus:border-primary/50 focus:ring-1 focus:ring-primary/50 transition-all placeholder:text-white/20"
                                />
                                <button
                                    type="submit"
                                    disabled={isLoading || !input.trim()}
                                    className="absolute right-2 p-2 bg-primary hover:bg-red-700 rounded-lg text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-[0_0_10px_rgba(229,9,20,0.3)]"
                                >
                                    {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                                </button>
                            </div>
                        </form>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Premium Launcher Button */}
            <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setIsOpen(!isOpen)}
                className="pointer-events-auto w-14 h-14 md:w-16 md:h-16 rounded-2xl bg-gradient-to-br from-primary via-red-600 to-rose-900 shadow-[0_0_30px_rgba(229,9,20,0.5)] flex items-center justify-center border border-white/20 z-50 hover:shadow-[0_0_50px_rgba(229,9,20,0.8)] transition-all duration-500 group relative overflow-hidden"
            >
                {/* Shine Effect */}
                <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                <AnimatePresence mode="wait">
                    {isOpen ? (
                        <motion.div
                            key="close"
                            initial={{ rotate: -90, opacity: 0 }}
                            animate={{ rotate: 0, opacity: 1 }}
                            exit={{ rotate: 90, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                        >
                            <X className="w-6 h-6 md:w-8 md:h-8 text-white drop-shadow-md" />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="chat"
                            initial={{ scale: 0.5, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.5, opacity: 0 }}
                            transition={{ type: "spring", stiffness: 260, damping: 20 }}
                        >
                            <BrainCircuit className="w-7 h-7 md:w-9 md:h-9 text-white fill-white/10 drop-shadow-[0_0_10px_rgba(255,255,255,0.5)]" />
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.button>
        </div>
    );
};
