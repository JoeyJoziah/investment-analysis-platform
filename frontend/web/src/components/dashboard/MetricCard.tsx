import React, { memo } from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  Skeleton,
  useTheme,
  alpha,
} from '@mui/material';
import {
  AccountBalance as PortfolioIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  ShowChart as PnLIcon,
  Psychology as SentimentIcon,
  Timeline as ReturnIcon,
} from '@mui/icons-material';

interface MetricCardProps {
  title: string;
  value: string;
  change?: number;
  changeValue?: number;
  sentiment?: number;
  icon: 'portfolio' | 'return' | 'pnl' | 'sentiment';
  color: string;
  isLoading?: boolean;
  onClick?: () => void;
}

/**
 * MetricCard - Key performance indicator card
 *
 * Displays portfolio metrics with:
 * - Large value display
 * - Change indicator with icon
 * - Color-coded gain/loss
 * - Loading skeleton state
 *
 * WCAG 2.1 AA Compliant:
 * - Uses icons + color for gain/loss indication
 * - Proper contrast ratios
 * - Screen reader announcements
 */
const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  change,
  changeValue,
  sentiment,
  icon,
  color,
  isLoading = false,
  onClick,
}) => {
  const theme = useTheme();

  // Get icon component based on type
  const getIcon = () => {
    const iconProps = {
      sx: {
        fontSize: 28,
        color: color,
      },
    };

    switch (icon) {
      case 'portfolio':
        return <PortfolioIcon {...iconProps} />;
      case 'return':
        return <ReturnIcon {...iconProps} />;
      case 'pnl':
        return <PnLIcon {...iconProps} />;
      case 'sentiment':
        return <SentimentIcon {...iconProps} />;
      default:
        return <PortfolioIcon {...iconProps} />;
    }
  };

  // Determine if change is positive
  const isPositive = (change ?? 0) >= 0;

  // Format change value for display
  const formatChangeValue = (val: number): string => {
    const absVal = Math.abs(val);
    if (absVal >= 1000000) {
      return `$${(absVal / 1000000).toFixed(2)}M`;
    }
    if (absVal >= 1000) {
      return `$${(absVal / 1000).toFixed(1)}K`;
    }
    return `$${absVal.toFixed(0)}`;
  };

  // Loading state
  if (isLoading) {
    return (
      <Card
        sx={{
          height: '100%',
          minHeight: { xs: 100, sm: 120 },
        }}
      >
        <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
            }}
          >
            <Box sx={{ flex: 1 }}>
              <Skeleton
                variant="text"
                width={100}
                height={20}
                sx={{ mb: 1 }}
              />
              <Skeleton
                variant="text"
                width={140}
                height={36}
                sx={{ mb: 1 }}
              />
              <Skeleton variant="text" width={80} height={20} />
            </Box>
            <Skeleton
              variant="circular"
              width={48}
              height={48}
              sx={{ borderRadius: 2 }}
            />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card
      sx={{
        height: '100%',
        minHeight: { xs: 100, sm: 120 },
        cursor: onClick ? 'pointer' : 'default',
        transition: 'all 0.2s ease-out',
        '&:hover': onClick
          ? {
              transform: 'translateY(-2px)',
              boxShadow: theme.shadows[4],
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
      aria-label={
        onClick
          ? `${title}: ${value}${change !== undefined ? `, ${isPositive ? 'up' : 'down'} ${Math.abs(change).toFixed(2)} percent` : ''}`
          : undefined
      }
    >
      <CardContent sx={{ p: { xs: 2, sm: 3 } }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
          }}
        >
          <Box sx={{ flex: 1, minWidth: 0 }}>
            {/* Label */}
            <Typography
              variant="caption"
              component="span"
              sx={{
                color: 'text.secondary',
                fontWeight: 500,
                textTransform: 'uppercase',
                letterSpacing: '0.05em',
                display: 'block',
                mb: 0.5,
              }}
            >
              {title}
            </Typography>

            {/* Value */}
            <Typography
              variant="h4"
              component="div"
              sx={{
                fontWeight: 700,
                fontSize: { xs: '1.25rem', sm: '1.5rem', md: '1.75rem' },
                lineHeight: 1.2,
                fontFamily: '"SF Mono", Monaco, monospace',
                mb: 0.5,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {value}
            </Typography>

            {/* Change indicator */}
            {change !== undefined && (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5,
                }}
                aria-live="polite"
              >
                {isPositive ? (
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
                  component="span"
                  sx={{
                    color: isPositive
                      ? theme.palette.success.main
                      : theme.palette.error.main,
                    fontWeight: 600,
                    fontFamily: '"SF Mono", Monaco, monospace',
                  }}
                  aria-label={`${isPositive ? 'Gain' : 'Loss'} of ${Math.abs(change).toFixed(2)} percent`}
                >
                  {isPositive ? '+' : ''}
                  {change.toFixed(2)}%
                </Typography>
                {changeValue !== undefined && (
                  <Typography
                    variant="caption"
                    component="span"
                    sx={{
                      color: 'text.secondary',
                      ml: 0.5,
                    }}
                  >
                    ({isPositive ? '+' : '-'}
                    {formatChangeValue(changeValue)})
                  </Typography>
                )}
              </Box>
            )}

            {/* Sentiment indicator (for AI Sentiment card) */}
            {sentiment !== undefined && (
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                  mt: 0.5,
                }}
              >
                <Box
                  sx={{
                    flex: 1,
                    height: 6,
                    borderRadius: 3,
                    backgroundColor: alpha(theme.palette.grey[300], 0.5),
                    overflow: 'hidden',
                  }}
                  role="progressbar"
                  aria-valuenow={sentiment}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label={`AI sentiment score: ${sentiment} out of 100`}
                >
                  <Box
                    sx={{
                      height: '100%',
                      width: `${sentiment}%`,
                      borderRadius: 3,
                      backgroundColor:
                        sentiment >= 70
                          ? theme.palette.success.main
                          : sentiment >= 40
                            ? theme.palette.warning.main
                            : theme.palette.error.main,
                      transition: 'width 0.3s ease-out',
                    }}
                  />
                </Box>
                <Typography
                  variant="caption"
                  sx={{
                    color: 'text.secondary',
                    fontWeight: 600,
                    minWidth: 32,
                  }}
                >
                  {sentiment}/100
                </Typography>
              </Box>
            )}
          </Box>

          {/* Icon */}
          <Box
            sx={{
              backgroundColor: alpha(color, 0.1),
              borderRadius: 2,
              p: 1.5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              ml: 2,
            }}
            aria-hidden="true"
          >
            {getIcon()}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

export default memo(MetricCard);
