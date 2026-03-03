import React from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Button,
  IconButton,
  TextField,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  BookmarkBorder,
  Bookmark,
  Share,
  Refresh,
  SearchOutlined,
} from '@mui/icons-material';
import type { StockQuote } from '../../store/slices/stockSlice';

export interface AnalysisHeaderProps {
  quote: StockQuote;
  isInWatchlist: boolean;
  searchTicker: string;
  onWatchlistToggle: () => void;
  onSearch: () => void;
  onRefresh: () => void;
  onSearchTickerChange: (value: string) => void;
  formatCurrency: (value: number) => string;
  formatPercent: (value: number) => string;
  formatLargeNumber: (value: number) => string;
}

export const AnalysisHeader: React.FC<AnalysisHeaderProps> = ({
  quote,
  isInWatchlist,
  searchTicker,
  onWatchlistToggle,
  onSearch,
  onRefresh,
  onSearchTickerChange,
  formatCurrency,
  formatPercent,
  formatLargeNumber,
}) => {
  return (
    <Box sx={{ mb: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="h4" fontWeight="bold">
              {quote.ticker}
            </Typography>
            <IconButton onClick={onWatchlistToggle}>
              {isInWatchlist ? (
                <Bookmark color="primary" />
              ) : (
                <BookmarkBorder />
              )}
            </IconButton>
            <IconButton>
              <Share />
            </IconButton>
          </Box>
          <Typography variant="h6" color="text.secondary">
            {quote.companyName}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <TextField
            size="small"
            label="Compare with"
            value={searchTicker}
            onChange={(e) => onSearchTickerChange(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === 'Enter' && onSearch()}
            sx={{ width: 150 }}
          />
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={onRefresh}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Price Info */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 2, mb: 2 }}>
              <Typography variant="h3" fontWeight="bold">
                {formatCurrency(quote.price)}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {quote.change >= 0 ? (
                  <TrendingUp color="success" />
                ) : (
                  <TrendingDown color="error" />
                )}
                <Typography
                  variant="h5"
                  color={quote.change >= 0 ? 'success.main' : 'error.main'}
                >
                  {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)} ({formatPercent(quote.changePercent)})
                </Typography>
              </Box>
            </Box>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Open
                </Typography>
                <Typography variant="body2">{formatCurrency(quote.open)}</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Day Range
                </Typography>
                <Typography variant="body2">
                  {formatCurrency(quote.low)} - {formatCurrency(quote.high)}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Volume
                </Typography>
                <Typography variant="body2">{formatLargeNumber(quote.volume)}</Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="caption" color="text.secondary">
                  Avg Volume
                </Typography>
                <Typography variant="body2">{formatLargeNumber(quote.avgVolume)}</Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2, height: '100%' }}>
            <Typography variant="h6" gutterBottom>
              Key Statistics
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  Market Cap
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {formatLargeNumber(quote.marketCap)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  P/E Ratio
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {quote.peRatio?.toFixed(2) || '-'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  52W Range
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  ${quote.week52Low.toFixed(2)} - ${quote.week52High.toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  Dividend Yield
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {quote.dividendYield ? `${(quote.dividendYield * 100).toFixed(2)}%` : '-'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  Beta
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {quote.beta?.toFixed(2) || '-'}
                </Typography>
              </Grid>
              <Grid item xs={6}>
                <Typography variant="caption" color="text.secondary">
                  EPS
                </Typography>
                <Typography variant="body2" fontWeight="bold">
                  {quote.eps ? formatCurrency(quote.eps) : '-'}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export interface EmptyTickerProps {
  searchTicker: string;
  onSearch: () => void;
  onSearchTickerChange: (value: string) => void;
}

export const EmptyTickerView: React.FC<EmptyTickerProps> = ({
  searchTicker,
  onSearch,
  onSearchTickerChange,
}) => {
  return (
    <Container maxWidth="xl">
      <Box sx={{ textAlign: 'center', py: 8 }}>
        <Box sx={{ mb: 2, opacity: 0.5 }}>
          <SearchOutlined sx={{ fontSize: 64, color: 'text.secondary' }} />
        </Box>
        <Typography variant="h5" gutterBottom>
          Enter a stock ticker above to begin analysis
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 3, maxWidth: 480, mx: 'auto' }}>
          Search for any stock symbol to view detailed charts, technical indicators, fundamentals, news, and more.
        </Typography>
        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
          <TextField
            label="Ticker Symbol"
            placeholder="e.g. AAPL, MSFT, TSLA"
            value={searchTicker}
            onChange={(e) => onSearchTickerChange(e.target.value.toUpperCase())}
            onKeyDown={(e) => e.key === 'Enter' && onSearch()}
          />
          <Button variant="contained" onClick={onSearch} disabled={!searchTicker.trim()}>
            Analyze
          </Button>
        </Box>
      </Box>
    </Container>
  );
};

export interface LoadingErrorViewProps {
  isLoading: boolean;
  error: string | null;
  hasQuote: boolean;
}

export const LoadingErrorView: React.FC<LoadingErrorViewProps> = ({
  isLoading,
  error,
  hasQuote,
}) => {
  if (isLoading) {
    return <LinearProgress />;
  }

  if (error) {
    return (
      <Container maxWidth="xl">
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  if (!hasQuote) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h6" sx={{ mt: 3 }}>
          Loading stock data...
        </Typography>
      </Container>
    );
  }

  return null;
};

export default { AnalysisHeader, EmptyTickerView, LoadingErrorView };
