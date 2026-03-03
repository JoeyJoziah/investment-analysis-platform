/**
 * Enhanced Recommendation Card with improved accessibility and usability
 * Features: Screen reader support, keyboard navigation, loading states, error handling
 *
 * Thin orchestrator that composes:
 *  - RecommendationHeader (avatar, ticker info, bookmark/notification/menu icons)
 *  - RecommendationMetrics (action chip, confidence, prices, scores, risk, reasoning)
 *  - RecommendationActions (view/trade buttons, menu, trade dialog, snackbar)
 */

import React, { useState, useCallback, memo } from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Chip,
  Avatar,
  Badge,
  Skeleton,
  Alert,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Star,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  announceToScreenReader,
  useKeyboardNavigation,
  useReducedMotion,
} from '../../utils/accessibility';
import RecommendationHeader from './recommendation/RecommendationHeader';
import RecommendationMetrics from './recommendation/RecommendationMetrics';
import RecommendationActions from './recommendation/RecommendationActions';
import type { Recommendation } from './recommendation/types';
import { getActionColor, formatValue } from './recommendation/utils';

interface EnhancedRecommendationCardProps {
  recommendation: Recommendation;
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
  onSelect: _onSelect,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const reducedMotion = useReducedMotion();

  // State
  const [bookmarked, setBookmarked] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  // Keyboard navigation
  useKeyboardNavigation({
    onEnter: () => {
      if (document.activeElement === document.getElementById(`card-${recommendation.ticker}`)) {
        handleViewDetails();
      }
    },
    onEscape: () => {
      setAnchorEl(null);
    },
  });

  // Handlers
  const handleViewDetails = useCallback(() => {
    navigate(`/analysis/${recommendation.ticker}`);
    announceToScreenReader(`Navigating to ${recommendation.ticker} details`);
  }, [navigate, recommendation.ticker]);

  const handleToggleBookmark = useCallback(() => {
    setBookmarked((prev) => {
      const next = !prev;
      announceToScreenReader(
        next
          ? `${recommendation.ticker} added to watchlist`
          : `${recommendation.ticker} removed from watchlist`
      );
      onAction?.(recommendation.ticker, next ? 'add_watchlist' : 'remove_watchlist');
      return next;
    });
  }, [recommendation.ticker, onAction]);

  const handleToggleNotifications = useCallback(() => {
    setNotificationsEnabled((prev) => {
      const next = !prev;
      announceToScreenReader(
        next
          ? `Price alerts enabled for ${recommendation.ticker}`
          : `Price alerts disabled for ${recommendation.ticker}`
      );
      return next;
    });
  }, [recommendation.ticker]);

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
    const actionColor = getActionColor(recommendation.action, theme);
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
                  badgeContent={bookmarked ? <Star sx={{ fontSize: 12 }} /> : undefined}
                  color="primary"
                >
                  <Avatar sx={{
                    bgcolor: alpha(actionColor, 0.1),
                    color: actionColor,
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
                    bgcolor: alpha(actionColor, 0.1),
                    color: actionColor,
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
          <RecommendationHeader
            recommendation={recommendation}
            bookmarked={bookmarked}
            notificationsEnabled={notificationsEnabled}
            onToggleBookmark={handleToggleBookmark}
            onToggleNotifications={handleToggleNotifications}
            onMenuOpen={(e) => setAnchorEl(e.currentTarget)}
          />
          <RecommendationMetrics recommendation={recommendation} />
        </CardContent>

        <RecommendationActions
          recommendation={recommendation}
          onAction={onAction}
          anchorEl={anchorEl}
          onMenuClose={() => setAnchorEl(null)}
        />
      </Card>
    </motion.div>
  );
});

EnhancedRecommendationCard.displayName = 'EnhancedRecommendationCard';

export default EnhancedRecommendationCard;
