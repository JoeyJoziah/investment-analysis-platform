/**
 * Actions section of the EnhancedRecommendationCard
 * Includes action buttons, options menu, trade dialog, and notification snackbar
 */

import React, { useState, useCallback } from 'react';
import {
  CardActions,
  Box,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  InputAdornment,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Visibility,
  CompareArrows,
  Analytics,
  History,
  Share,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { announceToScreenReader } from '../../../utils/accessibility';
import type { Recommendation, NotificationState } from './types';
import { getActionMuiColor, formatValue } from './utils';

export interface RecommendationActionsProps {
  recommendation: Recommendation;
  onAction?: (ticker: string, action: string) => void;
  anchorEl: null | HTMLElement;
  onMenuClose: () => void;
}

const RecommendationActions: React.FC<RecommendationActionsProps> = ({
  recommendation,
  onAction,
  anchorEl,
  onMenuClose,
}) => {
  const navigate = useNavigate();
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);
  const [tradeAmount, setTradeAmount] = useState('');
  const [notification, setNotification] = useState<NotificationState>({
    open: false,
    message: '',
    severity: 'info',
  });

  const handleViewDetails = useCallback(() => {
    navigate(`/analysis/${recommendation.ticker}`);
    announceToScreenReader(`Navigating to ${recommendation.ticker} details`);
  }, [navigate, recommendation.ticker]);

  const handleTrade = useCallback(() => {
    if (!tradeAmount || isNaN(Number(tradeAmount))) {
      setNotification({
        open: true,
        message: 'Please enter a valid amount',
        severity: 'error',
      });
      return;
    }

    onAction?.(recommendation.ticker, `trade_${recommendation.action.toLowerCase()}_${tradeAmount}`);
    setTradeDialogOpen(false);
    setTradeAmount('');

    setNotification({
      open: true,
      message: `${recommendation.action} order placed for ${tradeAmount} shares of ${recommendation.ticker}`,
      severity: 'success',
    });

    announceToScreenReader(`Trade executed: ${recommendation.action} ${tradeAmount} shares of ${recommendation.ticker}`);
  }, [tradeAmount, recommendation, onAction]);

  const muiColor = getActionMuiColor(recommendation.action);

  return (
    <>
      {/* Actions */}
      <CardActions sx={{ p: 2, pt: 0 }}>
        <Button
          size="small"
          variant="contained"
          startIcon={<Visibility />}
          onClick={handleViewDetails}
          fullWidth
          aria-label={`View detailed analysis for ${recommendation.ticker}`}
        >
          View Details
        </Button>
        <Button
          size="small"
          variant="outlined"
          color={muiColor}
          onClick={() => setTradeDialogOpen(true)}
          aria-label={`Execute ${recommendation.action} trade for ${recommendation.ticker}`}
        >
          {recommendation.action}
        </Button>
      </CardActions>

      {/* More Options Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={onMenuClose}
      >
        <MenuItem onClick={() => { navigate(`/compare?tickers=${recommendation.ticker}`); onMenuClose(); }}>
          <ListItemIcon><CompareArrows /></ListItemIcon>
          <ListItemText>Compare</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { navigate(`/analysis/${recommendation.ticker}?tab=history`); onMenuClose(); }}>
          <ListItemIcon><History /></ListItemIcon>
          <ListItemText>View History</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { onMenuClose(); }}>
          <ListItemIcon><Share /></ListItemIcon>
          <ListItemText>Share</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { navigate(`/analysis/${recommendation.ticker}?tab=analytics`); onMenuClose(); }}>
          <ListItemIcon><Analytics /></ListItemIcon>
          <ListItemText>Advanced Analytics</ListItemText>
        </MenuItem>
      </Menu>

      {/* Trade Dialog */}
      <Dialog
        open={tradeDialogOpen}
        onClose={() => setTradeDialogOpen(false)}
        aria-labelledby="trade-dialog-title"
      >
        <DialogTitle id="trade-dialog-title">
          {recommendation.action} {recommendation.ticker}
        </DialogTitle>
        <DialogContent>
          <Box py={2}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Current Price: {formatValue(recommendation.current_price, 'currency')}
            </Typography>
            <TextField
              autoFocus
              margin="dense"
              label="Number of Shares"
              type="number"
              fullWidth
              variant="outlined"
              value={tradeAmount}
              onChange={(e) => setTradeAmount(e.target.value)}
              InputProps={{
                startAdornment: <InputAdornment position="start">#</InputAdornment>,
              }}
              aria-label="Enter number of shares to trade"
            />
            {tradeAmount && !isNaN(Number(tradeAmount)) && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Total Value: {formatValue(Number(tradeAmount) * recommendation.current_price, 'currency')}
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setTradeDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleTrade}
            variant="contained"
            color={muiColor}
          >
            Confirm {recommendation.action}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Notification Snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
      >
        <Alert
          onClose={() => setNotification({ ...notification, open: false })}
          severity={notification.severity}
          variant="filled"
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default RecommendationActions;
