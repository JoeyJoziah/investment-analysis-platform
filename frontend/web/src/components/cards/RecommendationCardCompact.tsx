import React, { memo } from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Chip,
  Avatar,
  LinearProgress,
  useTheme,
  alpha,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  StarOutline as StarIcon,
} from '@mui/icons-material';

interface RecommendationCardCompactProps {
  ticker: string;
  companyName: string;
  action: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  currentPrice: number;
  targetPrice: number;
  potentialReturn: number;
  riskLevel?: 'LOW' | 'MEDIUM' | 'HIGH';
  onClick?: () => void;
  rank?: number;
}

/**
 * RecommendationCardCompact - Compact recommendation display
 *
 * Features:
 * - Action badge with color coding
 * - Confidence progress bar
 * - Price targets
 * - Potential return
 *
 * WCAG 2.1 AA Compliant:
 * - Icons paired with colors
 * - Proper contrast ratios
 * - Keyboard accessible
 */
const RecommendationCardCompact: React.FC<RecommendationCardCompactProps> = ({
  ticker,
  companyName,
  action,
  confidence,
  currentPrice,
  targetPrice,
  potentialReturn,
  riskLevel,
  onClick,
  rank,
}) => {
  const theme = useTheme();

  // Get action color
  const getActionColor = (action: string): string => {
    switch (action) {
      case 'STRONG_BUY':
        return '#1B5E20'; // Dark green
      case 'BUY':
        return theme.palette.success.main;
      case 'HOLD':
        return theme.palette.warning.main;
      case 'SELL':
        return theme.palette.error.main;
      case 'STRONG_SELL':
        return '#B71C1C'; // Dark red
      default:
        return theme.palette.grey[500];
    }
  };

  // Get action icon
  const getActionIcon = (action: string) => {
    const iconProps = { sx: { fontSize: 16 }, 'aria-hidden': true as const };
    switch (action) {
      case 'STRONG_BUY':
      case 'BUY':
        return <TrendingUpIcon {...iconProps} />;
      case 'HOLD':
        return <TrendingFlatIcon {...iconProps} />;
      case 'SELL':
      case 'STRONG_SELL':
        return <TrendingDownIcon {...iconProps} />;
      default:
        return null;
    }
  };

  // Format action label
  const formatActionLabel = (action: string): string => {
    return action.replace('_', ' ');
  };

  // Get risk color
  const getRiskColor = (risk?: string): string => {
    switch (risk) {
      case 'LOW':
        return theme.palette.success.main;
      case 'MEDIUM':
        return theme.palette.warning.main;
      case 'HIGH':
        return theme.palette.error.main;
      default:
        return theme.palette.grey[500];
    }
  };

  const actionColor = getActionColor(action);
  const isPositiveReturn = potentialReturn >= 0;

  return (
    <Card
      sx={{
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease-out',
        border: `1px solid ${alpha(actionColor, 0.2)}`,
        '&:hover': onClick
          ? {
              transform: 'translateY(-2px)',
              boxShadow: theme.shadows[4],
              borderColor: alpha(actionColor, 0.4),
            }
          : {},
        '&:focus-visible': {
          outline: `2px solid ${theme.palette.primary.main}`,
          outlineOffset: 2,
        },
      }}
      onClick={onClick}
      tabIndex={onClick ? 0 : -1}
      role={onClick ? 'button' : undefined}
      onKeyDown={(e) => {
        if (onClick && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          onClick();
        }
      }}
      aria-label={`${ticker} ${formatActionLabel(action)} recommendation, ${confidence}% confidence, potential return ${isPositiveReturn ? 'positive' : 'negative'} ${Math.abs(potentialReturn).toFixed(1)}%`}
    >
      <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
        {/* Top row: Avatar, Ticker/Company, Action Badge */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            mb: 1.5,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            {/* Rank or Avatar */}
            <Avatar
              sx={{
                width: 36,
                height: 36,
                backgroundColor: alpha(actionColor, 0.1),
                color: actionColor,
                fontSize: '0.875rem',
                fontWeight: 700,
              }}
            >
              {rank ? `#${rank}` : ticker.substring(0, 2)}
            </Avatar>

            {/* Ticker and Company */}
            <Box>
              <Typography
                variant="subtitle2"
                sx={{
                  fontWeight: 700,
                  fontFamily: '"SF Mono", Monaco, monospace',
                  lineHeight: 1.2,
                }}
              >
                {ticker}
              </Typography>
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{
                  display: 'block',
                  maxWidth: 120,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {companyName}
              </Typography>
            </Box>
          </Box>

          {/* Action Badge */}
          <Chip
            icon={getActionIcon(action)}
            label={formatActionLabel(action)}
            size="small"
            sx={{
              backgroundColor: alpha(actionColor, 0.1),
              color: actionColor,
              fontWeight: 700,
              fontSize: '0.7rem',
              height: 24,
              '& .MuiChip-icon': {
                color: actionColor,
              },
            }}
          />
        </Box>

        {/* Confidence bar */}
        <Box sx={{ mb: 1.5 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 0.5,
            }}
          >
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: '0.7rem' }}
            >
              AI Confidence
            </Typography>
            <Typography
              variant="caption"
              sx={{
                fontWeight: 600,
                fontFamily: '"SF Mono", Monaco, monospace',
                color:
                  confidence >= 80
                    ? theme.palette.success.main
                    : confidence >= 60
                      ? theme.palette.warning.main
                      : theme.palette.error.main,
              }}
            >
              {confidence}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={confidence}
            sx={{
              height: 4,
              borderRadius: 2,
              backgroundColor: alpha(theme.palette.grey[500], 0.2),
              '& .MuiLinearProgress-bar': {
                backgroundColor:
                  confidence >= 80
                    ? theme.palette.success.main
                    : confidence >= 60
                      ? theme.palette.warning.main
                      : theme.palette.error.main,
                borderRadius: 2,
              },
            }}
            aria-label={`Confidence level: ${confidence}%`}
          />
        </Box>

        {/* Bottom row: Price targets and return */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          {/* Price info */}
          <Box>
            <Typography
              variant="body2"
              sx={{
                fontFamily: '"SF Mono", Monaco, monospace',
                fontSize: '0.8125rem',
              }}
            >
              ${currentPrice.toFixed(2)}
              <Typography
                component="span"
                sx={{
                  color: 'text.secondary',
                  mx: 0.5,
                }}
              >
                to
              </Typography>
              <Typography
                component="span"
                sx={{
                  color: theme.palette.primary.main,
                  fontWeight: 600,
                }}
              >
                ${targetPrice.toFixed(2)}
              </Typography>
            </Typography>
          </Box>

          {/* Potential Return */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
            }}
          >
            {isPositiveReturn ? (
              <TrendingUpIcon
                sx={{
                  fontSize: 18,
                  color: theme.palette.success.main,
                }}
                aria-hidden="true"
              />
            ) : (
              <TrendingDownIcon
                sx={{
                  fontSize: 18,
                  color: theme.palette.error.main,
                }}
                aria-hidden="true"
              />
            )}
            <Typography
              variant="body2"
              sx={{
                fontWeight: 700,
                fontFamily: '"SF Mono", Monaco, monospace',
                color: isPositiveReturn
                  ? theme.palette.success.main
                  : theme.palette.error.main,
              }}
            >
              {isPositiveReturn ? '+' : ''}
              {potentialReturn.toFixed(1)}%
            </Typography>
          </Box>
        </Box>

        {/* Risk level indicator */}
        {riskLevel && (
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              mt: 1,
            }}
          >
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: '0.65rem' }}
            >
              Risk:
            </Typography>
            <Chip
              label={riskLevel}
              size="small"
              sx={{
                height: 16,
                fontSize: '0.6rem',
                backgroundColor: alpha(getRiskColor(riskLevel), 0.1),
                color: getRiskColor(riskLevel),
                fontWeight: 600,
              }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default memo(RecommendationCardCompact);
