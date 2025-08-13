import React from 'react';
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
  alpha
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
  Assessment
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface RecommendationCardProps {
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
  };
  compact?: boolean;
  onAction?: (ticker: string, action: string) => void;
}

const RecommendationCard: React.FC<RecommendationCardProps> = ({
  recommendation,
  compact = false,
  onAction
}) => {
  const theme = useTheme();
  const navigate = useNavigate();

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

  const getRiskColor = (risk?: string) => {
    switch (risk) {
      case 'LOW':
        return theme.palette.success.main;
      case 'MEDIUM':
        return theme.palette.warning.main;
      case 'HIGH':
        return theme.palette.error.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const formatMarketCap = (value?: number) => {
    if (!value) return 'N/A';
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toFixed(2)}`;
  };

  const formatVolume = (value?: number) => {
    if (!value) return 'N/A';
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toString();
  };

  const handleViewDetails = () => {
    navigate(`/stocks/${recommendation.ticker}`);
  };

  const handleAddToPortfolio = () => {
    if (onAction) {
      onAction(recommendation.ticker, 'add_to_portfolio');
    }
  };

  const handleAddToWatchlist = () => {
    if (onAction) {
      onAction(recommendation.ticker, 'add_to_watchlist');
    }
  };

  if (compact) {
    return (
      <Card sx={{ 
        cursor: 'pointer',
        transition: 'all 0.3s',
        '&:hover': {
          transform: 'translateY(-2px)',
          boxShadow: theme.shadows[4]
        }
      }}
      onClick={handleViewDetails}
      >
        <CardContent sx={{ p: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={1}>
              <Avatar sx={{ 
                bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                color: getActionColor(recommendation.action),
                width: 32,
                height: 32,
                fontSize: '0.875rem'
              }}>
                {recommendation.ticker.substring(0, 2)}
              </Avatar>
              <Box>
                <Typography variant="subtitle2" fontWeight="bold">
                  {recommendation.ticker}
                </Typography>
                <Typography variant="caption" color="textSecondary" noWrap sx={{ maxWidth: 100 }}>
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
            <Typography variant="body2">
              ${recommendation.current_price.toFixed(2)}
            </Typography>
            {recommendation.potential_return && (
              <Typography
                variant="body2"
                sx={{
                  color: recommendation.potential_return >= 0 
                    ? theme.palette.success.main 
                    : theme.palette.error.main
                }}
              >
                {recommendation.potential_return >= 0 ? '+' : ''}
                {recommendation.potential_return.toFixed(2)}%
              </Typography>
            )}
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          <Box display="flex" alignItems="center" gap={2}>
            <Avatar sx={{ 
              bgcolor: alpha(getActionColor(recommendation.action), 0.1),
              color: getActionColor(recommendation.action),
              width: 48,
              height: 48
            }}>
              {recommendation.ticker.substring(0, 2)}
            </Avatar>
            <Box>
              <Typography variant="h6" component="div">
                {recommendation.ticker}
              </Typography>
              <Typography variant="body2" color="textSecondary" noWrap sx={{ maxWidth: 200 }}>
                {recommendation.company_name}
              </Typography>
              {recommendation.sector && (
                <Chip label={recommendation.sector} size="small" variant="outlined" sx={{ mt: 0.5 }} />
              )}
            </Box>
          </Box>
          <Box textAlign="right">
            <Chip
              label={recommendation.action}
              sx={{
                bgcolor: alpha(getActionColor(recommendation.action), 0.1),
                color: getActionColor(recommendation.action),
                fontWeight: 'bold',
                mb: 1
              }}
            />
            <LinearProgress
              variant="determinate"
              value={recommendation.confidence}
              sx={{
                height: 6,
                borderRadius: 3,
                bgcolor: alpha(theme.palette.primary.main, 0.1),
                '& .MuiLinearProgress-bar': {
                  bgcolor: getActionColor(recommendation.action)
                }
              }}
            />
            <Typography variant="caption" color="textSecondary">
              {recommendation.confidence}% confidence
            </Typography>
          </Box>
        </Box>

        <Box display="flex" justifyContent="space-between" mb={2}>
          <Box>
            <Typography variant="body2" color="textSecondary">
              Current Price
            </Typography>
            <Typography variant="h5">
              ${recommendation.current_price.toFixed(2)}
            </Typography>
          </Box>
          {recommendation.target_price && (
            <Box>
              <Typography variant="body2" color="textSecondary">
                Target Price
              </Typography>
              <Typography variant="h5" color="primary">
                ${recommendation.target_price.toFixed(2)}
              </Typography>
            </Box>
          )}
          {recommendation.potential_return !== undefined && (
            <Box>
              <Typography variant="body2" color="textSecondary">
                Potential Return
              </Typography>
              <Box display="flex" alignItems="center">
                {recommendation.potential_return >= 0 ? (
                  <TrendingUp sx={{ color: theme.palette.success.main, mr: 0.5 }} />
                ) : (
                  <TrendingDown sx={{ color: theme.palette.error.main, mr: 0.5 }} />
                )}
                <Typography
                  variant="h5"
                  sx={{
                    color: recommendation.potential_return >= 0 
                      ? theme.palette.success.main 
                      : theme.palette.error.main
                  }}
                >
                  {recommendation.potential_return >= 0 ? '+' : ''}
                  {recommendation.potential_return.toFixed(2)}%
                </Typography>
              </Box>
            </Box>
          )}
        </Box>

        {(recommendation.technical_score !== undefined || 
          recommendation.fundamental_score !== undefined || 
          recommendation.sentiment_score !== undefined) && (
          <Box mb={2}>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Analysis Scores
            </Typography>
            <Box display="flex" gap={2}>
              {recommendation.technical_score !== undefined && (
                <Box flex={1}>
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
                <Box flex={1}>
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
                <Box flex={1}>
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

        <Box display="flex" gap={2} mb={2} flexWrap="wrap">
          {recommendation.risk_level && (
            <Chip
              label={`Risk: ${recommendation.risk_level}`}
              size="small"
              sx={{
                bgcolor: alpha(getRiskColor(recommendation.risk_level), 0.1),
                color: getRiskColor(recommendation.risk_level)
              }}
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
          {recommendation.market_cap && (
            <Chip
              label={`MCap: ${formatMarketCap(recommendation.market_cap)}`}
              size="small"
              variant="outlined"
            />
          )}
          {recommendation.volume && (
            <Chip
              label={`Vol: ${formatVolume(recommendation.volume)}`}
              size="small"
              variant="outlined"
            />
          )}
        </Box>

        {recommendation.reasoning && (
          <Box>
            <Typography variant="body2" color="textSecondary" gutterBottom>
              Key Insights
            </Typography>
            <Typography variant="body2" sx={{ 
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 3,
              WebkitBoxOrient: 'vertical'
            }}>
              {recommendation.reasoning}
            </Typography>
          </Box>
        )}

        <Box display="flex" gap={1} mt={2}>
          {recommendation.pe_ratio && (
            <Tooltip title="Price-to-Earnings Ratio">
              <Chip
                label={`P/E: ${recommendation.pe_ratio.toFixed(2)}`}
                size="small"
                variant="outlined"
              />
            </Tooltip>
          )}
          {recommendation.dividend_yield && (
            <Tooltip title="Dividend Yield">
              <Chip
                icon={<AttachMoney />}
                label={`${recommendation.dividend_yield.toFixed(2)}%`}
                size="small"
                variant="outlined"
              />
            </Tooltip>
          )}
        </Box>
      </CardContent>

      <CardActions sx={{ p: 2, pt: 0 }}>
        <Button
          size="small"
          variant="contained"
          startIcon={<Visibility />}
          onClick={handleViewDetails}
          fullWidth
        >
          View Details
        </Button>
        <IconButton
          size="small"
          onClick={handleAddToPortfolio}
          sx={{ 
            bgcolor: alpha(theme.palette.primary.main, 0.1),
            '&:hover': {
              bgcolor: alpha(theme.palette.primary.main, 0.2)
            }
          }}
        >
          <Tooltip title="Add to Portfolio">
            <ShoppingCart />
          </Tooltip>
        </IconButton>
        <IconButton
          size="small"
          onClick={handleAddToWatchlist}
          sx={{ 
            bgcolor: alpha(theme.palette.secondary.main, 0.1),
            '&:hover': {
              bgcolor: alpha(theme.palette.secondary.main, 0.2)
            }
          }}
        >
          <Tooltip title="Add to Watchlist">
            <Star />
          </Tooltip>
        </IconButton>
      </CardActions>
    </Card>
  );
};

export default RecommendationCard;