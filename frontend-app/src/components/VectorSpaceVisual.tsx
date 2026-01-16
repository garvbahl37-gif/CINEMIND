import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    color: string;
    connections: number[];
}

const VectorSpaceVisual = () => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let particles: Particle[] = [];
        const particleCount = 60;
        const connectionDistance = 100;
        const mouseDistance = 150;

        const colors = ['#60A5FA', '#A855F7', '#ec4899', '#10B981']; // Tailwind blue, purple, pink, emerald

        // Initialize particles
        const init = () => {
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;

            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    size: Math.random() * 2 + 1,
                    color: colors[Math.floor(Math.random() * colors.length)],
                    connections: []
                });
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Update and draw particles
            particles.forEach((p, i) => {
                // Move
                p.x += p.vx;
                p.y += p.vy;

                // Bounce off walls
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

                // Mouse interaction
                const dx = mousePos.x - p.x;
                const dy = mousePos.y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < mouseDistance) {
                    const forceDirectionX = dx / distance;
                    const forceDirectionY = dy / distance;
                    const force = (mouseDistance - distance) / mouseDistance;
                    // Gentle attraction
                    p.vx += forceDirectionX * force * 0.05;
                    p.vy += forceDirectionY * force * 0.05;
                }

                // Draw particle
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = p.color;
                ctx.fill();
            });

            // Draw connections
            ctx.lineWidth = 0.5;
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < connectionDistance) {
                        ctx.beginPath();
                        ctx.strokeStyle = `rgba(255, 255, 255, ${1 - distance / connectionDistance})`;
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.stroke();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(animate);
        };

        init();
        animate();

        const handleResize = () => {
            init();
        };

        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
            cancelAnimationFrame(animationFrameId);
        };
    }, [mousePos]); // Re-init on resize, mousePos used in ref inside loop

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!canvasRef.current) return;
        const rect = canvasRef.current.getBoundingClientRect();
        setMousePos({
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        });
    };

    return (
        <div className="relative w-full h-96 bg-black/40 rounded-3xl border border-white/10 overflow-hidden backdrop-blur-sm group" ref={containerRef} onMouseMove={handleMouseMove}>
            <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 via-transparent to-purple-500/10 opacity-50 pointer-events-none" />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />

            <div className="absolute bottom-6 left-6 pointer-events-none">
                <h4 className="text-white font-bold text-lg mb-1 drop-shadow-md">Interactive Vector Space</h4>
                <p className="text-gray-400 text-xs max-w-xs">
                    Simulating high-dimensional semantic relationships.
                    Using FAISS, we map movies to these nodes based on plot similarity.
                    <br /><span className="text-primary mt-2 block">Hover to disrupt the field.</span>
                </p>
            </div>
        </div>
    );
};

export default VectorSpaceVisual;
