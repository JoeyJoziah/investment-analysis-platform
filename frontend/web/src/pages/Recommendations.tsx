import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Box,
  Button,
  Chip,
  IconButton,
  TextField,
  MenuItem,
  InputAdornment,
  Paper,
  Slider,
  FormControl,
  InputLabel,
  Select,
  ToggleButton,
  ToggleButtonGroup,
  Rating,
  LinearProgress,
  Tooltip,
  Badge,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  TrendingUp,
  TrendingDown,
  Info as InfoIcon,
  Bookmark as BookmarkIcon,
  BookmarkBorder as BookmarkBorderIcon,
  Assessment,
  Timeline,
  Security,
  Speed,
  ViewList,
  ViewModule,
  Refresh,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { fetchRecommendations } from '../store/slices/recommendationsSlice';
import { addToWatchlist, removeFromWatchlist } from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';

interface Recommendation {
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

const Recommendations: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { recommendations, loading: isLoading } = useAppSelector((state) => state.recommendations);
  const { watchlist } = useAppSelector((state) => state.portfolio);
  
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({
    recommendation: 'all',
    sector: 'all',
    risk: 'all',
    timeHorizon: 'all',
    minConfidence: 0,
    sortBy: 'confidence',
  });

  useEffect(() => {
    dispatch(fetchRecommendations({}));
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchRecommendations({}));
    dispatch(
      addNotification({
        type: 'info',
        message: 'Recommendations refreshed',
      })
    );
  };

  // Check if a ticker is in watchlist
  const isInWatchlist = (ticker: string) => {
    return watchlist?.items?.some(
      (item) => item.symbol.toUpperCase() === ticker.toUpperCase()
    ) ?? false;
  };

  const handleWatchlistToggle = async (ticker: string) => {
    try {
      if (isInWatchlist(ticker)) {
        await dispatch(removeFromWatchlist(ticker)).unwrap();
        dispatch(
          addNotification({
            type: 'info',
            message: `${ticker} removed from watchlist`,
          })
        );
      } else {
        await dispatch(addToWatchlist({ symbol: ticker })).unwrap();
        dispatch(
          addNotification({
            type: 'success',
            message: `${ticker} added to watchlist`,
          })
        );
      }
    } catch (error) {
      dispatch(
        addNotification({
          type: 'error',
          message: `Failed to update watchlist`,
        })
      );
    }
  };

  const getRecommendationColor = (recommendation: string) => {
    switch (recommendation) {
      case 'STRONG_BUY':
        return 'success';
      case 'BUY':
        return 'success';
      case 'HOLD':
        return 'warning';
      case 'SELL':
        return 'error';
      case 'STRONG_SELL':
        return 'error';
      default:
        return 'default';
    }
  };

  const getRiskColor = (risk: string) => {
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

  const filteredRecommendations = (recommendations as unknown as Recommendation[])
    .filter((rec) => {
      if (searchQuery && !rec.ticker.includes(searchQuery.toUpperCase()) &&
          !rec.companyName.toLowerCase().includes(searchQuery.toLowerCase())) {
        return false;
      }
      if (filters.recommendation !== 'all' && rec.recommendation !== filters.recommendation) {
        return false;
      }
      if (filters.sector !== 'all' && rec.sector !== filters.sector) {
        return false;
      }
      if (filters.risk !== 'all' && rec.risk !== filters.risk) {
        return false;
      }
      if (filters.timeHorizon !== 'all' && rec.timeHorizon !== filters.timeHorizon) {
        return false;
      }
      if (rec.confidence < filters.minConfidence) {
        return false;
      }
      return true;
    })
    .sort((a, b) => {
      switch (filters.sortBy) {
        case 'confidence':
          return b.confidence - a.confidence;
        case 'expectedReturn':
          return b.expectedReturn - a.expectedReturn;
        case 'ticker':
          return a.ticker.localeCompare(b.ticker);
        default:
          return 0;
      }
    });

  const uniqueSectors = [...new Set((recommendations as unknown as Recommendation[]).map((r) => r.sector))];

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight="bold">
            AI Recommendations
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(_, newMode) => newMode && setViewMode(newMode)}
              size="small"
            >
              <ToggleButton value="grid">
                <ViewModule />
              </ToggleButton>
              <ToggleButton value="list">
                <ViewList />
              </ToggleButton>
            </ToggleButtonGroup>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {/* Filters */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={3}>
              <TextField
                fullWidth
                placeholder="Search ticker or company..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <SearchIcon />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Recommendation</InputLabel>
                <Select
                  value={filters.recommendation}
                  label="Recommendation"
                  onChange={(e) => setFilters({ ...filters, recommendation: e.target.value })}
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="STRONG_BUY">Strong Buy</MenuItem>
                  <MenuItem value="BUY">Buy</MenuItem>
                  <MenuItem value="HOLD">Hold</MenuItem>
                  <MenuItem value="SELL">Sell</MenuItem>
                  <MenuItem value="STRONG_SELL">Strong Sell</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Sector</InputLabel>
                <Select
                  value={filters.sector}
                  label="Sector"
                  onChange={(e) => setFilters({ ...filters, sector: e.target.value })}
                >
                  <MenuItem value="all">All Sectors</MenuItem>
                  {uniqueSectors.map((sector) => (
                    <MenuItem key={sector} value={sector}>
                      {sector}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={1.5}>
              <FormControl fullWidth size="small">
                <InputLabel>Risk</InputLabel>
                <Select
                  value={filters.risk}
                  label="Risk"
                  onChange={(e) => setFilters({ ...filters, risk: e.target.value })}
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="LOW">Low</MenuItem>
                  <MenuItem value="MEDIUM">Medium</MenuItem>
                  <MenuItem value="HIGH">High</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={1.5}>
              <FormControl fullWidth size="small">
                <InputLabel>Time Horizon</InputLabel>
                <Select
                  value={filters.timeHorizon}
                  label="Time Horizon"
                  onChange={(e) => setFilters({ ...filters, timeHorizon: e.target.value })}
                >
                  <MenuItem value="all">All</MenuItem>
                  <MenuItem value="SHORT">Short Term</MenuItem>
                  <MenuItem value="MEDIUM">Medium Term</MenuItem>
                  <MenuItem value="LONG">Long Term</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={2}>
              <FormControl fullWidth size="small">
                <InputLabel>Sort By</InputLabel>
                <Select
                  value={filters.sortBy}
                  label="Sort By"
                  onChange={(e) => setFilters({ ...filters, sortBy: e.target.value })}
                >
                  <MenuItem value="confidence">Confidence</MenuItem>
                  <MenuItem value="expectedReturn">Expected Return</MenuItem>
                  <MenuItem value="ticker">Ticker</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <Box sx={{ px: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Minimum Confidence: {filters.minConfidence}%
                </Typography>
                <Slider
                  value={filters.minConfidence}
                  onChange={(_, value) => setFilters({ ...filters, minConfidence: value as number })}
                  valueLabelDisplay="auto"
                  min={0}
                  max={100}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 50, label: '50%' },
                    { value: 100, label: '100%' },
                  ]}
                />
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Results Count */}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Showing {filteredRecommendations.length} of {recommendations.length} recommendations
        </Typography>
      </Box>

      {/* Recommendations Grid/List */}
      {viewMode === 'grid' ? (
        <Grid container spacing={3}>
          {filteredRecommendations.map((rec: Recommendation) => (
            <Grid item xs={12} sm={6} md={4} lg={3} key={rec.id}>
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
                      onClick={() => handleWatchlistToggle(rec.ticker)}
                    >
                      {isInWatchlist(rec.ticker) ? (
                        <BookmarkIcon color="primary" />
                      ) : (
                        <BookmarkBorderIcon />
                      )}
                    </IconButton>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={rec.recommendation.replace('_', ' ')}
                      color={getRecommendationColor(rec.recommendation) as any}
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
                    onClick={() => navigate(`/analysis/${rec.ticker}`)}
                  >
                    View Analysis
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      ) : (
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
                {filteredRecommendations.map((rec: Recommendation) => (
                  <tr key={rec.id} style={{ borderBottom: '1px solid rgba(0,0,0,0.12)' }}>
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
                        color={getRecommendationColor(rec.recommendation) as any}
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
                        onClick={() => handleWatchlistToggle(rec.ticker)}
                      >
                        {isInWatchlist(rec.ticker) ? (
                          <BookmarkIcon fontSize="small" color="primary" />
                        ) : (
                          <BookmarkBorderIcon fontSize="small" />
                        )}
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => navigate(`/analysis/${rec.ticker}`)}
                      >
                        <InfoIcon fontSize="small" />
                      </IconButton>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Box>
        </Paper>
      )}
    </Container>
  );
};

export default Recommendations;