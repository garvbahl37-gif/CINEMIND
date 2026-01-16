import React from 'react';
import { motion, HTMLMotionProps } from 'framer-motion';
import { cn } from '../../lib/utils';
import { Loader2, LucideIcon } from 'lucide-react';

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
    variant?: 'primary' | 'secondary' | 'glass' | 'ghost' | 'outline' | 'cinema';
    size?: 'sm' | 'md' | 'lg' | 'icon';
    icon?: LucideIcon;
    isLoading?: boolean;
    children?: React.ReactNode;
}

const variants = {
    primary: 'bg-primary text-primary-foreground hover:bg-primary/90 shadow-[0_0_20px_-5px_var(--primary)]',
    secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    glass: 'bg-white/10 backdrop-blur-md border border-white/10 text-white hover:bg-white/20',
    ghost: 'bg-transparent text-gray-300 hover:text-white hover:bg-white/5',
    outline: 'border border-gray-600 text-gray-300 hover:border-white hover:text-white',
    cinema: 'bg-gradient-to-r from-[hsl(var(--primary))] to-[#b91c1c] text-white shadow-lg shadow-[hsl(var(--primary))]/40 hover:shadow-[hsl(var(--primary))]/60 border border-white/10'
};

const sizes = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-6 py-2.5 text-base',
    lg: 'px-8 py-3.5 text-lg font-semibold tracking-wide',
    icon: 'p-2',
};

const Button: React.FC<ButtonProps> = ({
    children,
    variant = 'primary',
    size = 'md',
    className,
    icon: Icon,
    isLoading,
    disabled,
    ...props
}) => {
    return (
        <motion.button
            whileHover={{ scale: 1.02, y: -1 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
                'relative inline-flex items-center justify-center rounded-lg transition-all duration-300',
                'disabled:opacity-50 disabled:cursor-not-allowed',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                variants[variant],
                sizes[size],
                className
            )}
            disabled={isLoading || disabled}
            {...props}
        >
            {isLoading && (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            )}
            {!isLoading && Icon && (
                <Icon className={cn("w-5 h-5", children && "mr-2")} />
            )}
            {children}
        </motion.button>
    );
};

export default Button;
