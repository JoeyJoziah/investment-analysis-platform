import React, { memo, useCallback } from 'react';
import {
  Box,
  TableCell,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Visibility as ViewIcon,
  NotificationsActive as AlertIcon,
} from '@mui/icons-material';

interface HoldingsActionsProps {
  ticker: string;
  onViewDetails: (ticker: string) => void;
  onSetAlert?: (ticker: string) => void;
}

/**
 * HoldingsActions - Inline action buttons rendered within a table row.
 *
 * Provides "View details" and "Set alert" icon buttons for each holding.
 * Clicks are stopped from propagating to the parent row handler.
 */
const HoldingsActions: React.FC<HoldingsActionsProps> = ({
  ticker,
  onViewDetails,
  onSetAlert,
}) => {
  const handleViewClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onViewDetails(ticker);
    },
    [ticker, onViewDetails]
  );

  const handleAlertClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (onSetAlert) {
        onSetAlert(ticker);
      }
    },
    [ticker, onSetAlert]
  );

  return (
    <TableCell align="center">
      <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
        <Tooltip title="View details">
          <IconButton
            size="small"
            onClick={handleViewClick}
            aria-label={`View details for ${ticker}`}
          >
            <ViewIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Set alert">
          <IconButton
            size="small"
            onClick={handleAlertClick}
            aria-label={`Set alert for ${ticker}`}
          >
            <AlertIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </TableCell>
  );
};

export default memo(HoldingsActions);
