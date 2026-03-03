import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Button,
  Chip,
  IconButton,
  LinearProgress,
  Paper,
} from '@mui/material';
import {
  Info as InfoIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
  Assessment,
  Timeline,
  Security,
  Speed,
  Lightbulb,
} from '@mui/icons-material';

export interface Recommendation {
  id: string;
  ticker: string;
  companyName: string;
  sector: string;
  price: number;
  targetPrice: number;
  recommendation: 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
  confidence: number;
  signals: {
    technical: number;
    fundamental: number;
    sentiment: number;
    ml_prediction: number;
  };
  reasons: string[];
  risk: 'LOW' | 'MEDIUM' | 'HIGH';
  timeHorizon: 'SHORT' | 'MEDIUM' | 'LONG';
  expectedReturn: number;
  lastUpdated: string;
}

// --- Color helpers ---

export const getRecommendationColor = (recommendation: string): string => {
  switch (recommendation) {
    case 'STRONG_BUY':
    case 'BUY':
      return 'success';
    case 'HOLD':
      return 'warning';
    case 'SELL':
    case 'STRONG_SELL':
      return 'error';
    default:
      return 'default';
  }
};

export const getRiskColor = (risk: string): string => {
  switch (risk) {
    case 'LOW':
      return 'success.main';
    case 'MEDIUM':
      return 'warning.main';
    case 'HIGH':
      return 'error.main';
    default:
      return 'text.secondary';
  }
};

// --- Grid Card ---

interface RecommendationGridCardProps {
  rec: Recommendation;
  isInWatchlist: boolean;
  onWatchlistToggle: (ticker: string) => void;
  onViewAnalysis: (ticker: string) => void;
}

const RecommendationGridCard: React.FC<RecommendationGridCardProps> = ({
  rec,
  isInWatchlist,
  onWatchlistToggle,
  onViewAnalysis,
}) => (
  <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
    <CardContent sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Typography variant="h6" fontWeight="bold">
            {rec.ticker}
          </Typography>
          <Typography variant="body2" color="text.secondary" noWrap>
            {rec.companyName}
          </Typography>
        </Box>
        <IconButton
          size="small"
          onClick={() => onWatchlistToggle(rec.ticker)}
        >
          {isInWatchlist ? (
            <BookmarkIcon color="primary" />
          ) : (
            <BookmarkBorderIcon />
          )}
        </IconButton>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Chip
          label={rec.recommendation.replace('_', ' ')}
          color={getRecommendationColor(rec.recommendation) as 'success' | 'warning' | 'error' | 'default'}
          size="small"
          sx={{ mr: 1 }}
        />
        <Chip
          label={rec.sector}
          variant="outlined"
          size="small"
        />
      </Box>

      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Current Price
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            ${rec.price.toFixed(2)}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Target Price
          </Typography>
          <Typography variant="body2" fontWeight="bold" color="primary.main">
            ${rec.targetPrice.toFixed(2)}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Expected Return
          </Typography>
          <Typography
            variant="body2"
            fontWeight="bold"
            color={rec.expectedReturn > 0 ? 'success.main' : 'error.main'}
          >
            {rec.expectedReturn > 0 ? '+' : ''}{rec.expectedReturn.toFixed(1)}%
          </Typography>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Confidence Score
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LinearProgress
            variant="determinate"
            value={rec.confidence}
            sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" fontWeight="bold">
            {rec.confidence}%
          </Typography>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Analysis Signals
        </Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Timeline sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption">
              Tech: {rec.signals.technical}%
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Assessment sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption">
              Fund: {rec.signals.fundamental}%
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Speed sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption">
              Sent: {rec.signals.sentiment}%
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <Security sx={{ fontSize: 16, color: 'text.secondary' }} />
            <Typography variant="caption">
              ML: {rec.signals.ml_prediction}%
            </Typography>
          </Box>
        </Box>
      </Box>

      <Box sx={{ mb: 2 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
          <Typography variant="body2" color="text.secondary">
            Risk Level
          </Typography>
          <Typography variant="body2" fontWeight="bold" color={getRiskColor(rec.risk)}>
            {rec.risk}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Typography variant="body2" color="text.secondary">
            Time Horizon
          </Typography>
          <Typography variant="body2" fontWeight="bold">
            {rec.timeHorizon} TERM
          </Typography>
        </Box>
      </Box>

      {rec.reasons.length > 0 && (
        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Key Reasons
          </Typography>
          <Box component="ul" sx={{ pl: 2, m: 0 }}>
            {rec.reasons.slice(0, 2).map((reason, index) => (
              <Typography
                component="li"
                variant="caption"
                key={index}
                sx={{ mb: 0.5 }}
              >
                {reason}
              </Typography>
            ))}
          </Box>
        </Box>
      )}
    </CardContent>
    <CardActions>
      <Button
        size="small"
        fullWidth
        onClick={() => onViewAnalysis(rec.ticker)}
      >
        View Analysis
      </Button>
    </CardActions>
  </Card>
);

// --- List Row ---

interface RecommendationListRowProps {
  rec: Recommendation;
  isInWatchlist: boolean;
  onWatchlistToggle: (ticker: string) => void;
  onViewAnalysis: (ticker: string) => void;
}

const RecommendationListRow: React.FC<RecommendationListRowProps> = ({
  rec,
  isInWatchlist,
  onWatchlistToggle,
  onViewAnalysis,
}) => (
  <tr style={{ borderBottom: '1px solid rgba(0,0,0,0.12)' }}>
    <td style={{ padding: '16px' }}>
      <Typography variant="subtitle2" fontWeight="bold">
        {rec.ticker}
      </Typography>
    </td>
    <td style={{ padding: '16px' }}>
      <Typography variant="body2">{rec.companyName}</Typography>
      <Typography variant="caption" color="text.secondary">
        {rec.sector}
      </Typography>
    </td>
    <td style={{ padding: '16px' }}>
      <Chip
        label={rec.recommendation.replace('_', ' ')}
        color={getRecommendationColor(rec.recommendation) as 'success' | 'warning' | 'error' | 'default'}
        size="small"
      />
    </td>
    <td style={{ padding: '16px', textAlign: 'right' }}>
      ${rec.price.toFixed(2)}
    </td>
    <td style={{ padding: '16px', textAlign: 'right' }}>
      ${rec.targetPrice.toFixed(2)}
    </td>
    <td style={{ padding: '16px', textAlign: 'right' }}>
      <Typography
        variant="body2"
        color={rec.expectedReturn > 0 ? 'success.main' : 'error.main'}
      >
        {rec.expectedReturn > 0 ? '+' : ''}{rec.expectedReturn.toFixed(1)}%
      </Typography>
    </td>
    <td style={{ padding: '16px', textAlign: 'center' }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <LinearProgress
          variant="determinate"
          value={rec.confidence}
          sx={{ flexGrow: 1, height: 6 }}
        />
        <Typography variant="caption">{rec.confidence}%</Typography>
      </Box>
    </td>
    <td style={{ padding: '16px', textAlign: 'center' }}>
      <Typography variant="body2" color={getRiskColor(rec.risk)}>
        {rec.risk}
      </Typography>
    </td>
    <td style={{ padding: '16px', textAlign: 'center' }}>
      <IconButton
        size="small"
        onClick={() => onWatchlistToggle(rec.ticker)}
      >
        {isInWatchlist ? (
          <BookmarkIcon fontSize="small" color="primary" />
        ) : (
          <BookmarkBorderIcon fontSize="small" />
        )}
      </IconButton>
      <IconButton
        size="small"
        onClick={() => onViewAnalysis(rec.ticker)}
      >
        <InfoIcon fontSize="small" />
      </IconButton>
    </td>
  </tr>
);

// --- Main Recommendations List ---

export interface RecommendationsListProps {
  recommendations: Recommendation[];
  totalCount: number;
  viewMode: 'grid' | 'list';
  isInWatchlist: (ticker: string) => boolean;
  onWatchlistToggle: (ticker: string) => void;
  onViewAnalysis: (ticker: string) => void;
  onClearFilters: () => void;
}

const RecommendationsList: React.FC<RecommendationsListProps> = ({
  recommendations,
  totalCount,
  viewMode,
  isInWatchlist,
  onWatchlistToggle,
  onViewAnalysis,
  onClearFilters,
}) => {
  if (recommendations.length === 0) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          py: 8,
          px: 2,
        }}
      >
        <Box sx={{ mb: 1.5, opacity: 0.5 }}>
          <Lightbulb sx={{ fontSize: 64, color: 'text.secondary' }} />
        </Box>
        <Typography variant="h6" color="text.secondary" textAlign="center">
          {totalCount === 0
            ? 'No recommendations available yet'
            : 'No recommendations match your filters'}
        </Typography>
        <Typography variant="body2" color="text.disabled" textAlign="center" sx={{ mt: 1, maxWidth: 420 }}>
          {totalCount === 0
            ? 'AI-powered recommendations will appear here once market data is available. Try running an analysis from the Analysis page to generate recommendations.'
            : 'Try adjusting your filters or search query to see more results.'}
        </Typography>
        {totalCount > 0 && (
          <Button
            variant="outlined"
            sx={{ mt: 2 }}
            onClick={onClearFilters}
          >
            Clear All Filters
          </Button>
        )}
      </Box>
    );
  }

  if (viewMode === 'grid') {
    return (
      <Grid container spacing={3}>
        {recommendations.map((rec) => (
          <Grid item xs={12} sm={6} md={4} lg={3} key={rec.id}>
            <RecommendationGridCard
              rec={rec}
              isInWatchlist={isInWatchlist(rec.ticker)}
              onWatchlistToggle={onWatchlistToggle}
              onViewAnalysis={onViewAnalysis}
            />
          </Grid>
        ))}
      </Grid>
    );
  }

  return (
    <Paper>
      <Box sx={{ overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ backgroundColor: 'rgba(0,0,0,0.04)' }}>
              <th style={{ padding: '16px', textAlign: 'left' }}>Ticker</th>
              <th style={{ padding: '16px', textAlign: 'left' }}>Company</th>
              <th style={{ padding: '16px', textAlign: 'left' }}>Recommendation</th>
              <th style={{ padding: '16px', textAlign: 'right' }}>Price</th>
              <th style={{ padding: '16px', textAlign: 'right' }}>Target</th>
              <th style={{ padding: '16px', textAlign: 'right' }}>Expected Return</th>
              <th style={{ padding: '16px', textAlign: 'center' }}>Confidence</th>
              <th style={{ padding: '16px', textAlign: 'center' }}>Risk</th>
              <th style={{ padding: '16px', textAlign: 'center' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {recommendations.map((rec) => (
              <RecommendationListRow
                key={rec.id}
                rec={rec}
                isInWatchlist={isInWatchlist(rec.ticker)}
                onWatchlistToggle={onWatchlistToggle}
                onViewAnalysis={onViewAnalysis}
              />
            ))}
          </tbody>
        </table>
      </Box>
    </Paper>
  );
};

export default RecommendationsList;
