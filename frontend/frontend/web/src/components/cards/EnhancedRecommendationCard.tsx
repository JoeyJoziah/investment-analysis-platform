/**
 * Enhanced Recommendation Card with improved accessibility and usability
 * Features: Screen reader support, keyboard navigation, loading states, error handling
 */

import React, { useState, useCallback, memo } from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Box,
  Typography,
  Chip,
  Button,
  LinearProgress,
  Avatar,
  IconButton,
  Tooltip,
  Rating,
  useTheme,
  alpha,
  Skeleton,
  Collapse,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  InputAdornment,
  Fade,
  Zoom,
  Badge,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Star,
  InfoOutlined,
  ShoppingCart,
  Visibility,
  Schedule,
  AttachMoney,
  ShowChart,
  Assessment,
  ExpandMore,
  ExpandLess,
  BookmarkBorder,
  Bookmark,
  Share,
  MoreVert,
  NotificationsNone,
  NotificationsActive,
  CompareArrows,
  Analytics,
  History,
  Security,
  Warning,
  CheckCircle,
  ErrorOutline,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { designTokens } from '../../theme/tokens';
import { 
  announceToScreenReader, 
  useKeyboardNavigation,
  useReducedMotion,
  ScreenReaderOnly,
} from '../../utils/accessibility';

interface EnhancedRecommendationCardProps {
  recommendation: {
    ticker: string;
    company_name?: string;
    action: 'BUY' | 'SELL' | 'HOLD';
    confidence: number;
    target_price?: number;
    current_price: number;
    potential_return?: number;
    risk_level?: 'LOW' | 'MEDIUM' | 'HIGH';
    reasoning?: string;
    technical_score?: number;
    fundamental_score?: number;
    sentiment_score?: number;
    ml_prediction?: number;
    time_horizon?: string;
    sector?: string;
    market_cap?: number;
    volume?: number;
    pe_ratio?: number;
    dividend_yield?: number;
    price_history?: Array<{ date: string; price: number }>;
    analyst_ratings?: { buy: number; hold: number; sell: number };
    esg_score?: number;
  };
  compact?: boolean;
  onAction?: (ticker: string, action: string) => void;
  loading?: boolean;
  error?: string;
  selected?: boolean;
  onSelect?: (ticker: string) => void;
}

const EnhancedRecommendationCard: React.FC<EnhancedRecommendationCardProps> = memo(({
  recommendation,
  compact = false,
  onAction,
  loading = false,
  error,
  selected = false,
  onSelect,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const reducedMotion = useReducedMotion();
  
  // State
  const [expanded, setExpanded] = useState(false);
  const [bookmarked, setBookmarked] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [tradeDialogOpen, setTradeDialogOpen] = useState(false);
  const [tradeAmount, setTradeAmount] = useState('');
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, message: '', severity: 'info' });
  
  // Keyboard navigation
  useKeyboardNavigation({
    onEnter: () => {
      if (document.activeElement === document.getElementById(`card-${recommendation.ticker}`)) {
        handleViewDetails();
      }
    },
    onEscape: () => {
      setTradeDialogOpen(false);
      setAnchorEl(null);
    },
  });
  
  // Handlers
  const handleViewDetails = useCallback(() => {
    navigate(`/analysis/${recommendation.ticker}`);
    announceToScreenReader(`Navigating to ${recommendation.ticker} details`);
  }, [navigate, recommendation.ticker]);
  
  const handleToggleBookmark = useCallback(() => {
    setBookmarked(!bookmarked);
    announceToScreenReader(
      bookmarked 
        ? `${recommendation.ticker} removed from watchlist` 
        : `${recommendation.ticker} added to watchlist`
    );
    onAction?.(recommendation.ticker, bookmarked ? 'remove_watchlist' : 'add_watchlist');
  }, [bookmarked, recommendation.ticker, onAction]);
  
  const handleToggleNotifications = useCallback(() => {
    setNotificationsEnabled(!notificationsEnabled);
    announceToScreenReader(
      notificationsEnabled 
        ? `Price alerts disabled for ${recommendation.ticker}` 
        : `Price alerts enabled for ${recommendation.ticker}`
    );
  }, [notificationsEnabled, recommendation.ticker]);
  
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
  
  // Utility functions
  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY':
        return theme.palette.success.main;
      case 'SELL':
        return theme.palette.error.main;
      case 'HOLD':
        return theme.palette.warning.main;
      default:
        return theme.palette.text.secondary;
    }
  };
  
  const getRiskIcon = (risk?: string) => {
    switch (risk) {
      case 'LOW':
        return <CheckCircle sx={{ fontSize: 16, color: theme.palette.success.main }} />;
      case 'MEDIUM':
        return <Warning sx={{ fontSize: 16, color: theme.palette.warning.main }} />;
      case 'HIGH':
        return <ErrorOutline sx={{ fontSize: 16, color: theme.palette.error.main }} />;
      default:
        return null;
    }
  };
  
  const formatValue = (value?: number, type: 'currency' | 'percent' | 'number' = 'number') => {
    if (value === undefined || value === null) return 'N/A';
    
    switch (type) {
      case 'currency':
        return new Intl.NumberFormat('en-US', {
          style: 'currency',
          currency: 'USD',
          minimumFractionDigits: 2,
        }).format(value);
      case 'percent':
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
      default:
        return value.toLocaleString();
    }
  };
  
  // Loading state
  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box display="flex" justifyContent="space-between" mb={2}>
            <Skeleton variant="circular" width={48} height={48} />
            <Skeleton width="30%" height={32} />
          </Box>
          <Skeleton width="60%" height={24} sx={{ mb: 1 }} />
          <Skeleton width="40%" height={20} sx={{ mb: 2 }} />
          <Skeleton variant="rectangular" height={60} />
        </CardContent>
      </Card>
    );
  }
  
  // Error state
  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            <Typography variant="body2">{error}</Typography>
          </Alert>
        </CardContent>
      </Card>
    );
  }
  
  // Compact view
  if (compact) {
    return (
      <motion.div
        initial={!reducedMotion ? { opacity: 0, scale: 0.95 } : {}}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.2 }}
      >
        <Card 
          id={`card-${recommendation.ticker}`}
          sx={{ 
            cursor: 'pointer',
            transition: 'all 0.2s ease-in-out',
            border: selected ? `2px solid ${theme.palette.primary.main}` : 'none',
            '&:hover': {
              transform: !reducedMotion ? 'translateY(-2px)' : 'none',
              boxShadow: theme.shadows[4],
            },
            '&:focus-within': {
              outline: `2px solid ${theme.palette.primary.main}`,
              outlineOffset: 2,
            }
          }}
          onClick={handleViewDetails}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              handleViewDetails();
            }
          }}
          role="article"
          aria-label={`${recommendation.ticker} recommendation: ${recommendation.action} with ${recommendation.confidence}% confidence`}
          tabIndex={0}
        >
          <CardContent sx={{ p: 2 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Box display="flex" alignItems="center" gap={1}>
                <Badge
                  badgeContent={bookmarked ? <Star sx={{ fontSize: 12 }} /> : null}
                  color="primary"
                >
                  <Avatar sx={{ 
                    bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                    color: getActionColor(recommendation.action),
                    width: 32,
                    height: 32,
                    fontSize: '0.875rem'
                  }}>
                    {recommendation.ticker.substring(0, 2)}
                  </Avatar>
                </Badge>
                <Box>
                  <Typography variant="subtitle2" fontWeight="bold">
                    {recommendation.ticker}
                  </Typography>
                  <Typography 
                    variant="caption" 
                    color="textSecondary" 
                    noWrap 
                    sx={{ maxWidth: 100, display: 'block' }}
                  >
                    {recommendation.company_name}
                  </Typography>
                </Box>
              </Box>
              <Box textAlign="right">
                <Chip
                  label={recommendation.action}
                  size="small"
                  sx={{
                    bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                    color: getActionColor(recommendation.action),
                    fontWeight: 'bold'
                  }}
                />
                <Typography variant="caption" display="block" color="textSecondary">
                  {recommendation.confidence}% confidence
                </Typography>
              </Box>
            </Box>
            <Box display="flex" justifyContent="space-between" mt={1}>
              <Typography variant="body2" aria-label={`Current price: ${formatValue(recommendation.current_price, 'currency')}`}>
                {formatValue(recommendation.current_price, 'currency')}
              </Typography>
              {recommendation.potential_return !== undefined && (
                <Box display="flex" alignItems="center" gap={0.5}>
                  {recommendation.potential_return >= 0 ? (
                    <TrendingUp sx={{ fontSize: 16, color: theme.palette.success.main }} />
                  ) : (
                    <TrendingDown sx={{ fontSize: 16, color: theme.palette.error.main }} />
                  )}
                  <Typography
                    variant="body2"
                    sx={{
                      color: recommendation.potential_return >= 0 
                        ? theme.palette.success.main 
                        : theme.palette.error.main
                    }}
                    aria-label={`Potential return: ${formatValue(recommendation.potential_return, 'percent')}`}
                  >
                    {formatValue(recommendation.potential_return, 'percent')}
                  </Typography>
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      </motion.div>
    );
  }
  
  // Full view
  return (
    <motion.div
      initial={!reducedMotion ? { opacity: 0, y: 20 } : {}}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        id={`card-${recommendation.ticker}`}
        sx={{ 
          height: '100%', 
          display: 'flex', 
          flexDirection: 'column',
          transition: 'all 0.2s ease-in-out',
          border: selected ? `2px solid ${theme.palette.primary.main}` : 'none',
          '&:hover': {
            boxShadow: theme.shadows[4],
          },
          '&:focus-within': {
            outline: `2px solid ${theme.palette.primary.main}`,
            outlineOffset: 2,
          }
        }}
        role="article"
        aria-label={`Detailed recommendation for ${recommendation.ticker}`}
      >
        <CardContent sx={{ flexGrow: 1 }}>
          {/* Header */}
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <Badge
                badgeContent={
                  notificationsEnabled ? 
                    <NotificationsActive sx={{ fontSize: 16 }} /> : 
                    null
                }
                color="primary"
              >
                <Avatar sx={{ 
                  bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                  color: getActionColor(recommendation.action),
                  width: 48,
                  height: 48
                }}>
                  {recommendation.ticker.substring(0, 2)}
                </Avatar>
              </Badge>
              <Box>
                <Typography variant="h6" component="h3">
                  {recommendation.ticker}
                </Typography>
                <Typography 
                  variant="body2" 
                  color="textSecondary" 
                  noWrap 
                  sx={{ maxWidth: 200 }}
                >
                  {recommendation.company_name}
                </Typography>
                {recommendation.sector && (
                  <Chip 
                    label={recommendation.sector} 
                    size="small" 
                    variant="outlined" 
                    sx={{ mt: 0.5 }}
                  />
                )}
              </Box>
            </Box>
            <Box display="flex" gap={1}>
              <IconButton 
                size="small" 
                onClick={handleToggleBookmark}
                aria-label={bookmarked ? 'Remove from watchlist' : 'Add to watchlist'}
              >
                {bookmarked ? <Bookmark color="primary" /> : <BookmarkBorder />}
              </IconButton>
              <IconButton 
                size="small"
                onClick={handleToggleNotifications}
                aria-label={notificationsEnabled ? 'Disable price alerts' : 'Enable price alerts'}
              >
                {notificationsEnabled ? <NotificationsActive color="primary" /> : <NotificationsNone />}
              </IconButton>
              <IconButton 
                size="small"
                onClick={(e) => setAnchorEl(e.currentTarget)}
                aria-label="More options"
              >
                <MoreVert />
              </IconButton>
            </Box>
          </Box>
          
          {/* Action and Confidence */}
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Chip
              label={recommendation.action}
              sx={{
                bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                color: getActionColor(recommendation.action),
                fontWeight: 'bold',
              }}
            />
            <Box>
              <Typography variant="caption" color="textSecondary" display="block">
                Confidence
              </Typography>
              <LinearProgress
                variant="determinate"
                value={recommendation.confidence}
                sx={{
                  height: 8,
                  borderRadius: designTokens.borderRadius.xs,
                  bgcolor: alpha(theme.palette.primary.main, 0.1),
                  '& .MuiLinearProgress-bar': {
                    bgcolor: getActionColor(recommendation.action),
                    borderRadius: designTokens.borderRadius.xs,
                  }
                }}
                aria-label={`Confidence: ${recommendation.confidence}%`}
              />
              <Typography variant="caption" color="textSecondary">
                {recommendation.confidence}%
              </Typography>
            </Box>
          </Box>
          
          {/* Price Information */}
          <Box display="grid" gridTemplateColumns="repeat(3, 1fr)" gap={2} mb={2}>
            <Box>
              <Typography variant="caption" color="textSecondary">
                Current Price
              </Typography>
              <Typography variant="h6">
                {formatValue(recommendation.current_price, 'currency')}
              </Typography>
            </Box>
            {recommendation.target_price && (
              <Box>
                <Typography variant="caption" color="textSecondary">
                  Target Price
                </Typography>
                <Typography variant="h6" color="primary">
                  {formatValue(recommendation.target_price, 'currency')}
                </Typography>
              </Box>
            )}
            {recommendation.potential_return !== undefined && (
              <Box>
                <Typography variant="caption" color="textSecondary">
                  Potential Return
                </Typography>
                <Box display="flex" alignItems="center">
                  {recommendation.potential_return >= 0 ? (
                    <TrendingUp sx={{ color: theme.palette.success.main, mr: 0.5 }} />
                  ) : (
                    <TrendingDown sx={{ color: theme.palette.error.main, mr: 0.5 }} />
                  )}
                  <Typography
                    variant="h6"
                    sx={{
                      color: recommendation.potential_return >= 0 
                        ? theme.palette.success.main 
                        : theme.palette.error.main
                    }}
                  >
                    {formatValue(recommendation.potential_return, 'percent')}
                  </Typography>
                </Box>
              </Box>
            )}
          </Box>
          
          {/* Analysis Scores */}
          {(recommendation.technical_score !== undefined || 
            recommendation.fundamental_score !== undefined || 
            recommendation.sentiment_score !== undefined) && (
            <Box mb={2}>
              <Typography variant="caption" color="textSecondary" gutterBottom>
                Analysis Scores
              </Typography>
              <Box display="grid" gridTemplateColumns="repeat(3, 1fr)" gap={1}>
                {recommendation.technical_score !== undefined && (
                  <Box>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <ShowChart sx={{ fontSize: 16 }} />
                      <Typography variant="caption">Technical</Typography>
                    </Box>
                    <Rating
                      value={recommendation.technical_score / 20}
                      readOnly
                      precision={0.5}
                      size="small"
                    />
                  </Box>
                )}
                {recommendation.fundamental_score !== undefined && (
                  <Box>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <Assessment sx={{ fontSize: 16 }} />
                      <Typography variant="caption">Fundamental</Typography>
                    </Box>
                    <Rating
                      value={recommendation.fundamental_score / 20}
                      readOnly
                      precision={0.5}
                      size="small"
                    />
                  </Box>
                )}
                {recommendation.sentiment_score !== undefined && (
                  <Box>
                    <Box display="flex" alignItems="center" gap={0.5}>
                      <Star sx={{ fontSize: 16 }} />
                      <Typography variant="caption">Sentiment</Typography>
                    </Box>
                    <Rating
                      value={recommendation.sentiment_score / 20}
                      readOnly
                      precision={0.5}
                      size="small"
                    />
                  </Box>
                )}
              </Box>
            </Box>
          )}
          
          {/* Risk and Time Horizon */}
          <Box display="flex" gap={1} mb={2} flexWrap="wrap">
            {recommendation.risk_level && (
              <Chip
                icon={getRiskIcon(recommendation.risk_level)}
                label={`Risk: ${recommendation.risk_level}`}
                size="small"
                variant="outlined"
              />
            )}
            {recommendation.time_horizon && (
              <Chip
                icon={<Schedule />}
                label={recommendation.time_horizon}
                size="small"
                variant="outlined"
              />
            )}
            {recommendation.esg_score && (
              <Chip
                icon={<Security />}
                label={`ESG: ${recommendation.esg_score}/100`}
                size="small"
                variant="outlined"
                color={
                  recommendation.esg_score >= 70 ? 'success' :
                  recommendation.esg_score >= 40 ? 'warning' : 'error'
                }
              />
            )}
          </Box>
          
          {/* Expandable Reasoning */}
          {recommendation.reasoning && (
            <Box>
              <Button
                size="small"
                onClick={() => setExpanded(!expanded)}
                endIcon={expanded ? <ExpandLess /> : <ExpandMore />}
                aria-expanded={expanded}
                aria-controls={`reasoning-${recommendation.ticker}`}
              >
                {expanded ? 'Hide' : 'Show'} Analysis
              </Button>
              <Collapse in={expanded} id={`reasoning-${recommendation.ticker}`}>
                <Box mt={1} p={2} bgcolor={alpha(theme.palette.primary.main, 0.05)} borderRadius={1}>
                  <Typography variant="body2">
                    {recommendation.reasoning}
                  </Typography>
                </Box>
              </Collapse>
            </Box>
          )}
        </CardContent>
        
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
            color={getActionColor(recommendation.action) as any}
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
          onClose={() => setAnchorEl(null)}
        >
          <MenuItem onClick={() => { navigate(`/compare?tickers=${recommendation.ticker}`); setAnchorEl(null); }}>
            <ListItemIcon><CompareArrows /></ListItemIcon>
            <ListItemText>Compare</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => { navigate(`/analysis/${recommendation.ticker}?tab=history`); setAnchorEl(null); }}>
            <ListItemIcon><History /></ListItemIcon>
            <ListItemText>View History</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => { /* Share logic */ setAnchorEl(null); }}>
            <ListItemIcon><Share /></ListItemIcon>
            <ListItemText>Share</ListItemText>
          </MenuItem>
          <MenuItem onClick={() => { navigate(`/analysis/${recommendation.ticker}?tab=analytics`); setAnchorEl(null); }}>
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
              color={getActionColor(recommendation.action) as any}
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
      </Card>
    </motion.div>
  );
});

EnhancedRecommendationCard.displayName = 'EnhancedRecommendationCard';

export default EnhancedRecommendationCard;