import React, { memo, useCallback, useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  Button,
  useTheme,
  useMediaQuery,
  alpha,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Notifications as NotificationsIcon,
  Search as SearchIcon,
  WifiOff as WifiOffIcon,
} from '@mui/icons-material';

// Import dashboard components
import MetricCard from './MetricCard';
import PerformanceSection from './PerformanceSection';
import RecommendationsPanel from '../panels/RecommendationsPanel';
import HoldingsSection from './HoldingsSection';
import AllocationPanel from '../panels/AllocationPanel';
import MarketOverviewPanel from '../panels/MarketOverviewPanel';
import NewsFeedPanel from '../panels/NewsFeedPanel';
import { WebSocketIndicator } from '../WebSocketIndicator';

// Types
import {
  PortfolioMetrics,
  Recommendation,
  Position,
  NewsArticle,
  MarketIndex,
} from '../../types';

interface DashboardLayoutProps {
  portfolioMetrics: PortfolioMetrics | null;
  recommendations: Recommendation[];
  positions: Position[];
  news: NewsArticle[];
  marketIndices: MarketIndex[];
  performanceData: Array<{ date: string; value: number }>;
  allocationData: Array<{ sector: string; value: number; percentage: number }>;
  isLoading: boolean;
  error: string | null;
  isConnected: boolean;
  onRefresh: () => void;
  onSearch: () => void;
  onNotifications: () => void;
}

/**
 * DashboardLayout - Main portfolio dashboard container
 *
 * Implements responsive grid layout with:
 * - Desktop (1200px+): Full 12-column grid
 * - Tablet (768-1199px): Condensed 2-column
 * - Mobile (<768px): Single column stack
 *
 * WCAG 2.1 AA Compliant
 */
const DashboardLayout: React.FC<DashboardLayoutProps> = ({
  portfolioMetrics,
  recommendations,
  positions,
  news,
  marketIndices,
  performanceData,
  allocationData,
  isLoading,
  error,
  isConnected,
  onRefresh,
  onSearch,
  onNotifications,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'lg'));
  const isDesktop = useMediaQuery(theme.breakpoints.up('lg'));

  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    await onRefresh();
    setRefreshing(false);
  }, [onRefresh]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ctrl/Cmd + K for search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        onSearch();
      }
      // Ctrl/Cmd + R for refresh (when not in browser refresh)
      if ((e.ctrlKey || e.metaKey) && e.key === 'r' && e.shiftKey) {
        e.preventDefault();
        handleRefresh();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onSearch, handleRefresh]);

  // Format values for display
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value);
  };

  const formatPercent = (value: number): string => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  // Error state
  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert
          severity="error"
          action={
            <Button color="inherit" size="small" onClick={handleRefresh}>
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box
      component="main"
      id="main-content"
      sx={{
        flexGrow: 1,
        p: { xs: 2, sm: 3 },
        minHeight: '100vh',
        backgroundColor: theme.palette.background.default,
      }}
      role="main"
      aria-label="Portfolio Dashboard"
    >
      {/* Skip link for accessibility */}
      <a
        href="#holdings"
        className="skip-link"
        style={{
          position: 'absolute',
          left: '-9999px',
          zIndex: 999,
        }}
      >
        Skip to holdings table
      </a>

      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        <Box>
          <Typography
            variant="h4"
            component="h1"
            sx={{
              fontWeight: 600,
              fontSize: { xs: '1.5rem', sm: '2rem' },
            }}
          >
            Portfolio Dashboard
          </Typography>
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 0.5 }}
          >
            {new Date().toLocaleDateString('en-US', {
              weekday: 'long',
              year: 'numeric',
              month: 'long',
              day: 'numeric',
            })}
          </Typography>
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <WebSocketIndicator connected={isConnected} />

          {!isConnected && (
            <Tooltip title="Connection lost. Reconnecting...">
              <WifiOffIcon
                color="warning"
                sx={{ mr: 1 }}
                aria-label="Connection lost"
              />
            </Tooltip>
          )}

          <Tooltip title="Search stocks (Ctrl+K)">
            <IconButton
              onClick={onSearch}
              aria-label="Search stocks"
              size={isMobile ? 'small' : 'medium'}
            >
              <SearchIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Notifications">
            <IconButton
              onClick={onNotifications}
              aria-label="View notifications"
              size={isMobile ? 'small' : 'medium'}
            >
              <NotificationsIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Refresh data (Ctrl+Shift+R)">
            <IconButton
              onClick={handleRefresh}
              disabled={refreshing}
              aria-label="Refresh dashboard data"
              size={isMobile ? 'small' : 'medium'}
            >
              <RefreshIcon
                sx={{
                  animation: refreshing ? 'spin 1s linear infinite' : 'none',
                  '@keyframes spin': {
                    '0%': { transform: 'rotate(0deg)' },
                    '100%': { transform: 'rotate(360deg)' },
                  },
                }}
              />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Loading indicator */}
      {(isLoading || refreshing) && (
        <LinearProgress
          sx={{ mb: 2 }}
          aria-label="Loading dashboard data"
        />
      )}

      {/* Main Grid Layout */}
      <Grid container spacing={{ xs: 2, sm: 3 }}>
        {/* Metric Cards Row */}
        <Grid item xs={12}>
          <Grid container spacing={{ xs: 1.5, sm: 2, md: 3 }}>
            <Grid item xs={6} md={3}>
              <MetricCard
                title="Portfolio Value"
                value={formatCurrency(portfolioMetrics?.totalValue || 0)}
                change={portfolioMetrics?.dayGainPercent || 0}
                changeValue={portfolioMetrics?.dayGain || 0}
                icon="portfolio"
                color={theme.palette.primary.main}
                isLoading={isLoading}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <MetricCard
                title="Total Return"
                value={formatPercent(portfolioMetrics?.totalGainPercent || 0)}
                change={portfolioMetrics?.totalGainPercent || 0}
                changeValue={portfolioMetrics?.totalGain || 0}
                icon="return"
                color={theme.palette.success.main}
                isLoading={isLoading}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <MetricCard
                title="Day P&L"
                value={formatCurrency(portfolioMetrics?.dayGain || 0)}
                change={portfolioMetrics?.dayGainPercent || 0}
                icon="pnl"
                color={theme.palette.info.main}
                isLoading={isLoading}
              />
            </Grid>
            <Grid item xs={6} md={3}>
              <MetricCard
                title="AI Sentiment"
                value="Bullish"
                sentiment={78}
                icon="sentiment"
                color={theme.palette.warning.main}
                isLoading={isLoading}
              />
            </Grid>
          </Grid>
        </Grid>

        {/* Performance Chart + Recommendations Row */}
        <Grid item xs={12} lg={8}>
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
              height: '100%',
              minHeight: 400,
            }}
            elevation={1}
          >
            <PerformanceSection
              data={performanceData}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4} id="recommendations">
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
              height: '100%',
              minHeight: { xs: 'auto', lg: 400 },
            }}
            elevation={1}
          >
            <RecommendationsPanel
              recommendations={recommendations}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        {/* Holdings Table + Allocation Row */}
        <Grid item xs={12} lg={8} id="holdings">
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
            }}
            elevation={1}
          >
            <HoldingsSection
              positions={positions}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
              height: '100%',
            }}
            elevation={1}
          >
            <AllocationPanel
              data={allocationData}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        {/* Market Overview + News Row */}
        <Grid item xs={12} md={4}>
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
              height: '100%',
            }}
            elevation={1}
          >
            <MarketOverviewPanel
              indices={marketIndices}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper
            sx={{
              p: { xs: 2, sm: 3 },
            }}
            elevation={1}
          >
            <NewsFeedPanel
              news={news}
              isLoading={isLoading}
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default memo(DashboardLayout);
