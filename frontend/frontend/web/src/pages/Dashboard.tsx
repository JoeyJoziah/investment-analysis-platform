import React, { useEffect, useState, useCallback, useMemo } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  Button,
  useTheme,
  alpha,
  Skeleton,
  Fade,
  Zoom
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Assessment,
  Notifications,
  Speed,
  AttachMoney,
  PieChart,
  Timeline,
  Refresh,
  Warning,
  CheckCircle
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { useMarketRealTimeData } from '../hooks/useRealTimeData';
import { useErrorHandler } from '../hooks/useErrorHandler';
import { fetchDashboardData } from '../store/slices/dashboardSlice';
import { fetchRecommendations } from '../store/slices/recommendationsSlice';
import StockChart from '../components/charts/StockChart';
import RecommendationCard from '../components/cards/RecommendationCard';
import MarketHeatmap from '../components/charts/MarketHeatmap';
import PortfolioSummary from '../components/cards/PortfolioSummary';
import NewsCard from '../components/cards/NewsCard';
import CostMonitor from '../components/monitoring/CostMonitor';

const Dashboard: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const [refreshing, setRefreshing] = useState(false);
  
  // Initialize real-time data and error handling
  const { isConnected } = useMarketRealTimeData();
  const { handleAsyncError } = useErrorHandler({ context: 'Dashboard' });
  
  const {
    marketOverview,
    topRecommendations,
    portfolioSummary,
    recentNews,
    marketSentiment,
    costMetrics,
    loading,
    error
  } = useAppSelector((state) => state.dashboard);

  // Load initial data
  useEffect(() => {
    const loadDashboardData = async () => {
      await handleAsyncError(async () => {
        await dispatch(fetchDashboardData()).unwrap();
        await dispatch(fetchRecommendations({ limit: 5 })).unwrap();
      }, 'Failed to load dashboard data');
    };
    
    loadDashboardData();
  }, [dispatch, handleAsyncError]);

  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await handleAsyncError(async () => {
        await Promise.all([
          dispatch(fetchDashboardData()).unwrap(),
          dispatch(fetchRecommendations({ limit: 5 })).unwrap()
        ]);
      }, 'Failed to refresh dashboard');
    } finally {
      setRefreshing(false);
    }
  }, [dispatch, handleAsyncError]);

  // Memoized market status for performance
  const marketStatus = useMemo(() => {
    const now = new Date();
    const currentHour = now.getHours();
    const isWeekend = now.getDay() === 0 || now.getDay() === 6;
    const isMarketHours = currentHour >= 9 && currentHour < 16; // Simplified market hours
    
    return {
      isOpen: !isWeekend && isMarketHours,
      status: isWeekend ? 'Weekend' : isMarketHours ? 'Open' : 'Closed',
    };
  }, []);

  const MetricCard = ({ title, value, change, icon, color }: any) => (
    <Card sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography color="textSecondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ mb: 1 }}>
              {value}
            </Typography>
            {change !== undefined && (
              <Box display="flex" alignItems="center">
                {change >= 0 ? (
                  <TrendingUp sx={{ color: theme.palette.success.main, mr: 0.5 }} />
                ) : (
                  <TrendingDown sx={{ color: theme.palette.error.main, mr: 0.5 }} />
                )}
                <Typography
                  variant="body2"
                  sx={{
                    color: change >= 0 ? theme.palette.success.main : theme.palette.error.main
                  }}
                >
                  {Math.abs(change)}%
                </Typography>
              </Box>
            )}
          </Box>
          <Box
            sx={{
              backgroundColor: alpha(color || theme.palette.primary.main, 0.1),
              borderRadius: 2,
              p: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (error) {
    return (
      <Box p={3}>
        <Alert severity="error" action={
          <Button color="inherit" size="small" onClick={handleRefresh}>
            Retry
          </Button>
        }>
          {error}
        </Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Box>
          <Typography variant="h4" component="h1" gutterBottom>
            Investment Dashboard
          </Typography>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="body1" color="textSecondary">
              {new Date().toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </Typography>
            <Chip
              icon={marketStatus.isOpen ? <CheckCircle /> : <Warning />}
              label={`Market ${marketStatus.status}`}
              color={marketStatus.isOpen ? 'success' : 'default'}
              size="small"
              variant="outlined"
            />
            <Chip
              icon={isConnected ? <CheckCircle /> : <Warning />}
              label={isConnected ? 'Live Data' : 'Offline'}
              color={isConnected ? 'success' : 'error'}
              size="small"
              variant="outlined"
            />
          </Box>
        </Box>
        <Box display="flex" gap={2} alignItems="center">
          <Tooltip title="Data connection status">
            <Box display="flex" alignItems="center" gap={1}>
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  backgroundColor: isConnected ? 'success.main' : 'error.main',
                  animation: isConnected ? 'pulse 2s infinite' : 'none',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                    '100%': { opacity: 1 },
                  },
                }}
              />
              <Typography variant="caption" color="text.secondary">
                {isConnected ? 'Real-time' : 'Delayed'}
              </Typography>
            </Box>
          </Tooltip>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <Refresh sx={{ 
                animation: refreshing ? 'spin 1s linear infinite' : 'none',
                '@keyframes spin': {
                  '0%': { transform: 'rotate(0deg)' },
                  '100%': { transform: 'rotate(360deg)' },
                }
              }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="View notifications">
            <IconButton>
              <Notifications />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {loading && <LinearProgress sx={{ mb: 2 }} />}

      <Grid container spacing={3}>
        {/* Key Metrics */}
        {loading ? (
          // Loading skeleton for metrics
          [...Array(4)].map((_, index) => (
            <Grid item xs={12} md={3} key={index}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Skeleton variant="text" height={20} width="60%" />
                  <Skeleton variant="text" height={40} width="80%" sx={{ my: 1 }} />
                  <Skeleton variant="text" height={16} width="40%" />
                </CardContent>
              </Card>
            </Grid>
          ))
        ) : (
          <>
            <Grid item xs={12} md={3}>
              <Fade in timeout={500}>
                <div>
                  <MetricCard
                    title="Portfolio Value"
                    value={portfolioSummary?.totalValue ? 
                      `$${portfolioSummary.totalValue.toLocaleString()}` : '$0'}
                    change={portfolioSummary?.dayChange}
                    icon={<AttachMoney sx={{ color: theme.palette.primary.main }} />}
                    color={theme.palette.primary.main}
                  />
                </div>
              </Fade>
            </Grid>
            <Grid item xs={12} md={3}>
              <Fade in timeout={700}>
                <div>
                  <MetricCard
                    title="Total Return"
                    value={portfolioSummary?.totalReturn ? 
                      `${portfolioSummary.totalReturn.toFixed(2)}%` : '0%'}
                    change={portfolioSummary?.totalReturn}
                    icon={<Timeline sx={{ color: theme.palette.success.main }} />}
                    color={theme.palette.success.main}
                  />
                </div>
              </Fade>
            </Grid>
            <Grid item xs={12} md={3}>
              <Fade in timeout={900}>
                <div>
                  <MetricCard
                    title="Market Sentiment"
                    value={marketSentiment?.overall || 'Neutral'}
                    icon={<Speed sx={{ color: theme.palette.info.main }} />}
                    color={theme.palette.info.main}
                  />
                </div>
              </Fade>
            </Grid>
            <Grid item xs={12} md={3}>
              <Fade in timeout={1100}>
                <div>
                  <MetricCard
                    title="Active Positions"
                    value={portfolioSummary?.activePositions || 0}
                    icon={<PieChart sx={{ color: theme.palette.warning.main }} />}
                    color={theme.palette.warning.main}
                  />
                </div>
              </Fade>
            </Grid>
          </>
        )}

        {/* Market Overview */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Market Overview</Typography>
              <Box display="flex" gap={1}>
                {marketOverview?.indices?.map((index: any) => (
                  <Chip
                    key={index.symbol}
                    label={`${index.symbol}: ${index.change >= 0 ? '+' : ''}${index.change}%`}
                    color={index.change >= 0 ? 'success' : 'error'}
                    size="small"
                  />
                ))}
              </Box>
            </Box>
            <MarketHeatmap data={marketOverview?.heatmap} />
          </Paper>
        </Grid>

        {/* Top Recommendations */}
        <Grid item xs={12} lg={4}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Top Recommendations</Typography>
              <Button 
                size="small" 
                startIcon={<Assessment />}
                href="/recommendations"
              >
                View All
              </Button>
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {topRecommendations?.slice(0, 3).map((rec: any) => (
                <RecommendationCard key={rec.ticker} recommendation={rec} compact />
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Portfolio Performance Chart */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Performance
            </Typography>
            <StockChart
              data={portfolioSummary?.performanceHistory}
              height={300}
              showVolume={false}
            />
          </Paper>
        </Grid>

        {/* Portfolio Summary */}
        <Grid item xs={12} lg={4}>
          <PortfolioSummary summary={portfolioSummary} />
        </Grid>

        {/* Recent News */}
        <Grid item xs={12} lg={8}>
          <Paper sx={{ p: 3 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
              <Typography variant="h6">Market News & Insights</Typography>
              <Button size="small">View All</Button>
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {recentNews?.slice(0, 5).map((news: any, index: number) => (
                <NewsCard key={index} news={news} />
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Cost Monitor */}
        <Grid item xs={12} lg={4}>
          <CostMonitor metrics={costMetrics} />
        </Grid>

        {/* Market Sectors Performance */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Sector Performance
            </Typography>
            <Grid container spacing={2}>
              {marketOverview?.sectors?.map((sector: any) => (
                <Grid item xs={6} sm={4} md={3} key={sector.name}>
                  <Box
                    sx={{
                      p: 2,
                      borderRadius: 1,
                      backgroundColor: alpha(
                        sector.change >= 0 ? 
                          theme.palette.success.main : 
                          theme.palette.error.main,
                        0.1
                      ),
                      textAlign: 'center'
                    }}
                  >
                    <Typography variant="body2" color="textSecondary">
                      {sector.name}
                    </Typography>
                    <Typography
                      variant="h6"
                      sx={{
                        color: sector.change >= 0 ? 
                          theme.palette.success.main : 
                          theme.palette.error.main
                      }}
                    >
                      {sector.change >= 0 ? '+' : ''}{sector.change}%
                    </Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;