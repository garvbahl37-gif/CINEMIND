import React from 'react';
import { Search } from 'lucide-react';
import { cn } from '../../lib/utils';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
    icon?: React.ReactNode;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(({
    className,
    type,
    icon,
    ...props
}, ref) => {
    return (
        <div className="relative group w-full">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <Search className="h-4 w-4 text-muted-foreground group-focus-within:text-primary transition-colors duration-300" />
            </div>
            <input
                type={type}
                className={cn(
                    "flex h-10 w-full rounded-md border border-white/10 bg-black/50 px-3 py-2 pl-10 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/50 focus-visible:border-transparent disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-300 backdrop-blur-md hover:bg-black/70",
                    className
                )}
                ref={ref}
                {...props}
            />
        </div>
    );
});
Input.displayName = "Input";

export default Input;
