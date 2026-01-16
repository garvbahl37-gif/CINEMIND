import { useState, useEffect } from 'react';
import { Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../lib/utils';

interface NavbarProps {
    onNavigate: (page: string) => void;
    currentPage: string;
}

const Navbar = ({ onNavigate, currentPage }: NavbarProps) => {
    const [isScrolled, setIsScrolled] = useState(false);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    useEffect(() => {
        const handleScroll = () => {
            setIsScrolled(window.scrollY > 50);
        };
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const navLinks = [
        { id: 'home', label: 'Home' },
        { id: 'top50', label: 'Top 50' },
        { id: 'tvshows', label: 'TV Shows' },
        { id: 'about', label: 'About' }
    ];

    const handleNavClick = (pageId: string) => {
        onNavigate(pageId);
        setMobileMenuOpen(false);
    };

    return (
        <motion.nav
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5 }}
            className={cn(
                "fixed top-0 w-full z-50 transition-all duration-500",
                isScrolled
                    ? "bg-black/80 backdrop-blur-xl py-3 shadow-[0_4px_30px_rgba(0,0,0,0.5)]"
                    : "bg-gradient-to-b from-black/90 to-transparent py-6"
            )}
        >
            <div className="max-w-[1800px] mx-auto px-4 md:px-12 flex items-center justify-between h-full relative">

                {/* Left: Spacer to balance Right Spacer */}
                <div className="hidden lg:block w-1/3" />

                {/* Center: Navigation Links */}
                <div className="hidden lg:flex flex-1 justify-center items-center gap-2">
                    {navLinks.map((item) => (
                        <button
                            key={item.id}
                            onClick={() => handleNavClick(item.id)}
                            className={cn(
                                "relative px-5 py-2 text-sm font-sans font-bold uppercase tracking-[0.2em] transition-all duration-300 rounded-full",
                                currentPage === item.id
                                    ? "text-white"
                                    : "text-neutral-500 hover:text-white hover:bg-white/5"
                            )}
                        >
                            <span className="relative z-10">{item.label}</span>
                            {currentPage === item.id && (
                                <motion.div
                                    layoutId="nav-pill"
                                    className="absolute inset-0 bg-white/10 border border-white/10 rounded-full shadow-[0_0_15px_-3px_rgba(255,255,255,0.1)]"
                                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                                />
                            )}
                        </button>
                    ))}
                </div>

                {/* Right: Spacer for Balance */}
                <div className="hidden lg:block w-1/3" />

                {/* Mobile Header (Visible only on mobile) */}
                <div className="lg:hidden text-lg font-bold text-white">
                    CINE<span className="text-primary">MIND</span>
                </div>

                {/* Mobile Menu Toggle */}
                <button
                    className="lg:hidden text-white p-2 hover:bg-white/10 rounded-full transition-colors z-50"
                    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                >
                    {mobileMenuOpen ? <X /> : <Menu />}
                </button>
            </div>

            {/* Mobile Menu Overlay */}
            <AnimatePresence>
                {mobileMenuOpen && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="lg:hidden bg-black/95 backdrop-blur-xl border-b border-white/10 overflow-hidden"
                    >
                        <div className="p-6 flex flex-col gap-6">
                            {navLinks.map((item) => (
                                <button
                                    key={item.id}
                                    onClick={() => handleNavClick(item.id)}
                                    className={cn(
                                        "text-lg font-medium hover:text-white hover:translate-x-2 transition-all duration-300 flex items-center gap-3 w-full text-left",
                                        currentPage === item.id ? "text-white" : "text-gray-300"
                                    )}
                                >
                                    <span className={cn(
                                        "w-1 h-1 rounded-full bg-primary transition-opacity",
                                        currentPage === item.id ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                                    )} />
                                    {item.label}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.nav>
    );
};

export default Navbar;
