import React from 'react';
import { Box, CircularProgress, Typography, Fade } from '@mui/material';

interface LoadingSpinnerProps {
  message?: string;
  fullScreen?: boolean;
  size?: number;
}

/**
 * LoadingSpinner - A reusable loading component for Suspense fallbacks
 *
 * Used as the fallback component when lazy-loaded routes are being fetched.
 * Provides a consistent loading experience across the application.
 */
const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  message = 'Loading...',
  fullScreen = false,
  size = 40,
}) => {
  const containerStyles = fullScreen
    ? {
        position: 'fixed' as const,
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(255, 255, 255, 0.9)',
        zIndex: 9999,
      }
    : {
        minHeight: '200px',
      };

  return (
    <Fade in timeout={300}>
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 2,
          ...containerStyles,
        }}
        role="status"
        aria-live="polite"
        aria-busy="true"
      >
        <CircularProgress size={size} thickness={4} />
        {message && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 1 }}
          >
            {message}
          </Typography>
        )}
      </Box>
    </Fade>
  );
};

export default LoadingSpinner;
