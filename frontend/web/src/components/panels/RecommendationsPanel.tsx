import React, { memo, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Skeleton,
  useTheme,
  alpha,
} from '@mui/material';
import { Assessment as AssessmentIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Recommendation } from '../../types';
import RecommendationCardCompact from '../cards/RecommendationCardCompact';

interface RecommendationsPanelProps {
  recommendations: Recommendation[];
  isLoading?: boolean;
  maxItems?: number;
}

/**
 * RecommendationsPanel - AI-generated stock recommendations
 *
 * Displays top recommendations with:
 * - Compact card format
 * - Confidence scores
 * - Action badges (BUY/SELL/HOLD)
 * - Potential returns
 *
 * WCAG 2.1 AA Compliant
 */
const RecommendationsPanel: React.FC<RecommendationsPanelProps> = ({
  recommendations = [],
  isLoading = false,
  maxItems = 3,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();

  const handleViewAll = useCallback(() => {
    navigate('/recommendations');
  }, [navigate]);

  const handleCardClick = useCallback(
    (ticker: string) => {
      navigate(`/stocks/${ticker}`);
    },
    [navigate]
  );

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Skeleton variant="text" width={180} height={28} />
          <Skeleton variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {[...Array(maxItems)].map((_, index) => (
            <Skeleton
              key={index}
              variant="rectangular"
              height={90}
              sx={{ borderRadius: 1 }}
            />
          ))}
        </Box>
      </Box>
    );
  }

  // Empty state
  if (recommendations.length === 0) {
    return (
      <Box>
        <Typography variant="h6" component="h2" gutterBottom>
          AI Recommendations
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            py: 6,
            color: 'text.secondary',
          }}
        >
          <AssessmentIcon
            sx={{ fontSize: 48, mb: 2, color: alpha(theme.palette.primary.main, 0.5) }}
          />
          <Typography variant="body1" gutterBottom>
            No recommendations available
          </Typography>
          <Typography variant="caption" textAlign="center">
            Recommendations are generated daily based on AI analysis
          </Typography>
        </Box>
      </Box>
    );
  }

  const displayedRecommendations = recommendations.slice(0, maxItems);

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography variant="h6" component="h2">
          AI Recommendations
        </Typography>
        <Button
          size="small"
          endIcon={<AssessmentIcon />}
          onClick={handleViewAll}
          aria-label="View all recommendations"
        >
          View All
        </Button>
      </Box>

      {/* Recommendations list */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
        role="list"
        aria-label="Top AI recommendations"
      >
        {displayedRecommendations.map((rec, index) => (
          <Box key={rec.id || rec.ticker} role="listitem">
            <RecommendationCardCompact
              ticker={rec.ticker}
              companyName={rec.companyName}
              action={rec.recommendation as 'BUY' | 'SELL' | 'HOLD'}
              confidence={rec.confidence}
              currentPrice={rec.price}
              targetPrice={rec.targetPrice}
              potentialReturn={rec.expectedReturn}
              riskLevel={rec.risk}
              onClick={() => handleCardClick(rec.ticker)}
              rank={index + 1}
            />
          </Box>
        ))}
      </Box>

      {/* Total count indicator */}
      {recommendations.length > maxItems && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: 'block',
            textAlign: 'center',
            mt: 2,
          }}
        >
          +{recommendations.length - maxItems} more recommendations available
        </Typography>
      )}
    </Box>
  );
};

export default memo(RecommendationsPanel);
