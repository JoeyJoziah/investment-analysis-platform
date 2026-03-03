/**
 * Header section of the EnhancedRecommendationCard
 * Displays ticker info, avatar, sector chip, and action icons (bookmark, notifications, menu)
 */

import React from 'react';
import {
  Box,
  Typography,
  Chip,
  Avatar,
  IconButton,
  Badge,
  alpha,
  useTheme,
} from '@mui/material';
import {
  BookmarkBorder,
  Bookmark,
  NotificationsNone,
  NotificationsActive,
  MoreVert,
} from '@mui/icons-material';
import type { Recommendation } from './types';
import { getActionColor } from './utils';

export interface RecommendationHeaderProps {
  recommendation: Recommendation;
  bookmarked: boolean;
  notificationsEnabled: boolean;
  onToggleBookmark: () => void;
  onToggleNotifications: () => void;
  onMenuOpen: (event: React.MouseEvent<HTMLElement>) => void;
}

const RecommendationHeader: React.FC<RecommendationHeaderProps> = ({
  recommendation,
  bookmarked,
  notificationsEnabled,
  onToggleBookmark,
  onToggleNotifications,
  onMenuOpen,
}) => {
  const theme = useTheme();
  const actionColor = getActionColor(recommendation.action, theme);

  return (
    <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
      <Box display="flex" alignItems="center" gap={2}>
        <Badge
          badgeContent={
            notificationsEnabled ?
              <NotificationsActive sx={{ fontSize: 16 }} /> :
              undefined
          }
          color="primary"
        >
          <Avatar sx={{
            bgcolor: alpha(actionColor, 0.1),
            color: actionColor,
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
          onClick={onToggleBookmark}
          aria-label={bookmarked ? 'Remove from watchlist' : 'Add to watchlist'}
        >
          {bookmarked ? <Bookmark color="primary" /> : <BookmarkBorder />}
        </IconButton>
        <IconButton
          size="small"
          onClick={onToggleNotifications}
          aria-label={notificationsEnabled ? 'Disable price alerts' : 'Enable price alerts'}
        >
          {notificationsEnabled ? <NotificationsActive color="primary" /> : <NotificationsNone />}
        </IconButton>
        <IconButton
          size="small"
          onClick={onMenuOpen}
          aria-label="More options"
        >
          <MoreVert />
        </IconButton>
      </Box>
    </Box>
  );
};

export default RecommendationHeader;
