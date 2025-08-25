import React, { Suspense, memo } from 'react';
import { Box, Skeleton, CircularProgress } from '@mui/material';

interface LazyLoadWrapperProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  height?: number | string;
  width?: number | string;
}

const DefaultSkeleton: React.FC<{ height?: number | string; width?: number | string }> = ({ 
  height = 200, 
  width = '100%' 
}) => (
  <Box sx={{ width, height, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
    <CircularProgress />
  </Box>
);

const LazyLoadWrapper: React.FC<LazyLoadWrapperProps> = memo(({ 
  children, 
  fallback, 
  height,
  width 
}) => {
  return (
    <Suspense 
      fallback={fallback || <DefaultSkeleton height={height} width={width} />}
    >
      {children}
    </Suspense>
  );
});

LazyLoadWrapper.displayName = 'LazyLoadWrapper';

export default LazyLoadWrapper;