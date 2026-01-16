import React from 'react';
import { cn } from '../../lib/utils';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
    variant?: 'default' | 'primary' | 'success' | 'warning' | 'outline';
}

const variants = {
    default: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    primary: 'bg-primary/20 text-primary-foreground border border-primary/50 shadow-[0_0_10px_-4px_var(--primary)]',
    success: 'bg-green-500/10 text-green-400 border border-green-500/20',
    warning: 'bg-yellow-500/10 text-yellow-400 border border-yellow-500/20',
    outline: 'text-foreground border border-border',
};

const Badge: React.FC<BadgeProps> = ({ children, variant = 'default', className, ...props }) => {
    return (
        <span className={cn(
            'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
            variants[variant],
            className
        )} {...props}>
            {children}
        </span>
    );
};

export default Badge;
