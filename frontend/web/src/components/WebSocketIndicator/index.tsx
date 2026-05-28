import React from 'react';
import { Box, Tooltip, Typography } from '@mui/material';
import { Circle } from '@mui/icons-material';
import { useAppSelector } from '../../hooks/redux';
import { env } from '../../utils/env';

const WebSocketIndicator: React.FC = () => {
  const { webSocketConnected } = useAppSelector((state) => state.app);

  // Real-time updates are opt-in (VITE_ENABLE_WEBSOCKETS). When disabled -- e.g. local
  // dev with no socket server -- hide the indicator rather than show an alarming
  // permanent red "Offline" badge.
  if (!env.ENABLE_WEBSOCKETS) {
    return null;
  }

  return (
    <Tooltip
      title={
        <Box>
          <Typography variant="caption">
            Real-time data: {webSocketConnected ? 'Connected' : 'Disconnected'}
          </Typography>
        </Box>
      }
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <Circle
          sx={{
            fontSize: 10,
            color: webSocketConnected ? 'success.main' : 'error.main',
            animation: webSocketConnected ? 'pulse 2s infinite' : 'none',
            '@keyframes pulse': {
              '0%': {
                opacity: 1,
              },
              '50%': {
                opacity: 0.5,
              },
              '100%': {
                opacity: 1,
              },
            },
          }}
        />
        <Typography variant="caption" color="text.secondary">
          {webSocketConnected ? 'Live' : 'Offline'}
        </Typography>
      </Box>
    </Tooltip>
  );
};

export default WebSocketIndicator;