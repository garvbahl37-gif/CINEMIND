import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ArrowUp, Loader2, MessageCircle } from 'lucide-react';
import { API_BASE } from '../config';
import { toMovie } from '../lib/api';
import { cn } from '../lib/utils';
import PosterCard from './PosterCard';
import { Movie } from '../types';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    recommendations?: Movie[];
}

const PROMPTS = ['90s horror', 'Korean thrillers', 'Christopher Nolan', 'Studio Ghibli'];

/**
 * The film assistant. Intent parsing is deterministic on the server, so it
 * answers in milliseconds; the films it names open on the match stage.
 */
export const ChatInterface = ({ onSelect }: { onSelect: (m: Movie) => void }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionId] = useState(() => Math.random().toString(36).substring(7));
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isOpen]);

    const send = async (text: string) => {
        const userMsg = text.trim();
        if (!userMsg || isLoading) return;
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
        setIsLoading(true);

        try {
            const res = await fetch(`${API_BASE}/api/chat/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: userMsg, session_id: sessionId })
            });
            if (!res.ok) throw new Error(`the API returned ${res.status}`);
            const data = await res.json();
            const recommendations: Movie[] = (data.recommendations || []).map(toMovie);
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: data.response,
                recommendations: recommendations.length > 0 ? recommendations : undefined
            }]);
        } catch (error) {
            const why = error instanceof Error ? error.message : 'no response';
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: `That request didn't go through (${why}). Check your connection and send it again.`,
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleSubmit = (e: React.FormEvent) => { e.preventDefault(); send(input); };

    return (
        <div className="fixed bottom-5 right-5 md:bottom-8 md:right-8 z-[100] flex flex-col items-end pointer-events-none">
            <AnimatePresence>
                {isOpen && (
                    <motion.div
                        role="dialog"
                        aria-label="Film assistant"
                        initial={{ opacity: 0, y: 16 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 16 }}
                        transition={{ duration: 0.2 }}
                        className="pointer-events-auto mb-3 w-[min(400px,calc(100vw-2.5rem))] h-[min(600px,calc(100vh-7rem))]
                                   bg-black border-[6px] border-ultra rounded-[28px] flex flex-col overflow-hidden
                                   shadow-[0_30px_70px_-20px_rgba(0,0,0,.9)]"
                    >
                        <div className="px-5 pt-4 pb-3 flex items-start justify-between gap-3">
                            <div>
                                <h3 className="display text-[30px]">Ask</h3>
                                <p className="mt-1 text-[13px] text-white/55">Name a genre, a decade, a language or a director.</p>
                            </div>
                            <button onClick={() => setIsOpen(false)} className="round-btn" aria-label="Close the assistant">
                                <X className="w-4 h-4" />
                            </button>
                        </div>

                        <div className="flex-1 overflow-y-auto px-5 py-3 space-y-4">
                            {messages.length === 0 && (
                                <div className="flex flex-wrap gap-1.5">
                                    {PROMPTS.map(p => (
                                        <button key={p} type="button" onClick={() => send(p)}
                                            className="pill h-9 px-3.5 border border-white/25 normal-case text-[13px] hover:border-white/60">
                                            {p}
                                        </button>
                                    ))}
                                </div>
                            )}
                            {messages.map((msg, i) => (
                                <div key={i} className={cn('flex flex-col gap-2.5', msg.role === 'user' ? 'items-end' : 'items-start')}>
                                    <div className={cn(
                                        'max-w-[88%] px-4 py-3 text-[14px] leading-relaxed rounded-[20px]',
                                        msg.role === 'user' ? 'bg-ultra text-white rounded-br-[6px]' : 'bg-panel-hi text-white/90 rounded-bl-[6px]',
                                    )}>
                                        {msg.content}
                                    </div>
                                    {msg.recommendations && (
                                        <div className="scroller gap-2 w-full -mx-1 px-1">
                                            {msg.recommendations.map((m, j) => (
                                                <PosterCard key={m.item_id ?? j} movie={m} size="sm" figure="none"
                                                    onSelect={(film) => { setIsOpen(false); onSelect(film); }} />
                                            ))}
                                        </div>
                                    )}
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex items-center gap-2 text-[13px] text-white/55">
                                    <Loader2 className="w-4 h-4 animate-spin" /> Looking through the catalogue
                                </div>
                            )}
                            <div ref={messagesEndRef} />
                        </div>

                        <form onSubmit={handleSubmit} className="p-3">
                            <div className="relative flex items-center">
                                <input
                                    type="text"
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    aria-label="Ask for a film"
                                    placeholder="A genre, a decade, a director…"
                                    className="w-full h-12 bg-white/[0.08] rounded-full pl-5 pr-14 text-[14px] text-white
                                               placeholder:text-white/40 outline-none focus-visible:outline-2 focus-visible:outline-white"
                                />
                                <button
                                    type="submit"
                                    aria-label="Send"
                                    disabled={isLoading || !input.trim()}
                                    className="absolute right-1.5 grid place-items-center w-9 h-9 rounded-full bg-ultra hover:bg-ultra-hi
                                               text-white disabled:opacity-40 transition-colors"
                                >
                                    <ArrowUp className="w-4 h-4" />
                                </button>
                            </div>
                        </form>
                    </motion.div>
                )}
            </AnimatePresence>

            {/*
              The launcher. Round and flat, to sit with the other round buttons
              rather than announce itself.
            */}
            <button
                type="button"
                onClick={() => setIsOpen(!isOpen)}
                aria-expanded={isOpen}
                aria-label={isOpen ? 'Close the film assistant' : 'Ask the film assistant'}
                className="pointer-events-auto w-14 h-14 rounded-full bg-ultra hover:bg-ultra-hi grid place-items-center
                           border-2 border-black shadow-[0_10px_30px_-8px_rgba(0,0,0,.85)] transition-colors"
            >
                {isOpen ? <X className="w-5 h-5" strokeWidth={2.25} /> : <MessageCircle className="w-[22px] h-[22px]" strokeWidth={2} />}
            </button>
        </div>
    );
};
