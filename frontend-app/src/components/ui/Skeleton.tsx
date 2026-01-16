import React from 'react';
import { cn } from '../../lib/utils';

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> { }

const Skeleton = ({ className, ...props }: SkeletonProps) => {
    return (
        <div
            className={cn("animate-pulse rounded-md bg-white/5", className)}
            {...props}
        />
    );
};

export default Skeleton;
