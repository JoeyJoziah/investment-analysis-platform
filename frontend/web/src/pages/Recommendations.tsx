import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Typography,
  Box,
  Button,
  ToggleButton,
  ToggleButtonGroup,
  LinearProgress,
} from '@mui/material';
import {
  ViewList,
  ViewModule,
  Refresh,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { fetchRecommendations } from '../store/slices/recommendationsSlice';
import { addToWatchlist, removeFromWatchlist } from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';
import RecommendationsList from '../components/recommendations/RecommendationsList';
import type { Recommendation } from '../components/recommendations/RecommendationsList';
import RecommendationsFilter from '../components/recommendations/RecommendationsFilter';
import type { RecommendationFilters } from '../components/recommendations/RecommendationsFilter';

const DEFAULT_FILTERS: RecommendationFilters = {
  recommendation: 'all',
  sector: 'all',
  risk: 'all',
  timeHorizon: 'all',
  minConfidence: 0,
  sortBy: 'confidence',
};

const Recommendations: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { recommendations, loading: isLoading } = useAppSelector((state) => state.recommendations);
  const { watchlist } = useAppSelector((state) => state.portfolio);

  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState<RecommendationFilters>(DEFAULT_FILTERS);

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
    } catch {
      dispatch(
        addNotification({
          type: 'error',
          message: `Failed to update watchlist`,
        })
      );
    }
  };

  const typedRecommendations = recommendations as unknown as Recommendation[];

  const filteredRecommendations = typedRecommendations
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

  const uniqueSectors = [...new Set(typedRecommendations.map((r) => r.sector))];

  const handleClearFilters = () => {
    setSearchQuery('');
    setFilters(DEFAULT_FILTERS);
  };

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

        <RecommendationsFilter
          searchQuery={searchQuery}
          onSearchQueryChange={setSearchQuery}
          filters={filters}
          onFiltersChange={setFilters}
          uniqueSectors={uniqueSectors}
        />

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Showing {filteredRecommendations.length} of {recommendations.length} recommendations
        </Typography>
      </Box>

      <RecommendationsList
        recommendations={filteredRecommendations}
        totalCount={recommendations.length}
        viewMode={viewMode}
        isInWatchlist={isInWatchlist}
        onWatchlistToggle={handleWatchlistToggle}
        onViewAnalysis={(ticker) => navigate(`/analysis/${ticker}`)}
        onClearFilters={handleClearFilters}
      />
    </Container>
  );
};

export default Recommendations;
