import React from 'react';
import { Box, Tooltip, Typography } from '@mui/material';
import { Circle } from '@mui/icons-material';
import { useAppSelector } from '../../hooks/redux';

const WebSocketIndicator: React.FC = () => {
  const { webSocketConnected } = useAppSelector((state) => state.app);

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