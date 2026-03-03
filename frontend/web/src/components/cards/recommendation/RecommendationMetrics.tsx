/**
 * Metrics section of the EnhancedRecommendationCard
 * Displays action chip, confidence bar, price info, analysis scores, risk/time chips, and reasoning
 */

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Chip,
  LinearProgress,
  Rating,
  Button,
  Collapse,
  alpha,
  useTheme,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Assessment,
  Star,
  Schedule,
  Security,
  ExpandMore,
  ExpandLess,
  CheckCircle,
  Warning,
  ErrorOutline,
} from '@mui/icons-material';
import { designTokens } from '../../../theme/tokens';
import type { Recommendation } from './types';
import { getActionColor, formatValue } from './utils';

export interface RecommendationMetricsProps {
  recommendation: Recommendation;
}

const RecommendationMetrics: React.FC<RecommendationMetricsProps> = ({
  recommendation,
}) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);
  const actionColor = getActionColor(recommendation.action, theme);

  const getRiskIcon = (risk?: string) => {
    switch (risk) {
      case 'LOW':
        return <CheckCircle sx={{ fontSize: 16, color: theme.palette.success.main }} />;
      case 'MEDIUM':
        return <Warning sx={{ fontSize: 16, color: theme.palette.warning.main }} />;
      case 'HIGH':
        return <ErrorOutline sx={{ fontSize: 16, color: theme.palette.error.main }} />;
      default:
        return undefined;
    }
  };

  return (
    <>
      {/* Action and Confidence */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Chip
          label={recommendation.action}
          sx={{
            bgcolor: alpha(actionColor, 0.1),
            color: actionColor,
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
                bgcolor: actionColor,
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
    </>
  );
};

export default RecommendationMetrics;
