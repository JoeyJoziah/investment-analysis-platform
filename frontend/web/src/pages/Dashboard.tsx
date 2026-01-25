import React, { useEffect, useState } from 'react';
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
  alpha
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
  Refresh
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
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

  useEffect(() => {
    dispatch(fetchDashboardData());
    dispatch(fetchRecommendations({ limit: 5 }));
  }, [dispatch]);

  const handleRefresh = async () => {
    setRefreshing(true);
    await dispatch(fetchDashboardData());
    await dispatch(fetchRecommendations({ limit: 5 }));
    setRefreshing(false);
  };

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
          <Typography variant="body1" color="textSecondary">
            {new Date().toLocaleDateString('en-US', { 
              weekday: 'long', 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}
          </Typography>
        </Box>
        <Box display="flex" gap={2}>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} disabled={refreshing}>
              <Refresh />
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
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Portfolio Value"
            value={portfolioSummary?.totalValue ? 
              `$${portfolioSummary.totalValue.toLocaleString()}` : '$0'}
            change={portfolioSummary?.dayChange}
            icon={<AttachMoney sx={{ color: theme.palette.primary.main }} />}
            color={theme.palette.primary.main}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Total Return"
            value={portfolioSummary?.totalReturn ? 
              `${portfolioSummary.totalReturn.toFixed(2)}%` : '0%'}
            change={portfolioSummary?.totalReturn}
            icon={<Timeline sx={{ color: theme.palette.success.main }} />}
            color={theme.palette.success.main}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Market Sentiment"
            value={marketSentiment?.overall || 'Neutral'}
            icon={<Speed sx={{ color: theme.palette.info.main }} />}
            color={theme.palette.info.main}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <MetricCard
            title="Active Positions"
            value={portfolioSummary?.activePositions || 0}
            icon={<PieChart sx={{ color: theme.palette.warning.main }} />}
            color={theme.palette.warning.main}
          />
        </Grid>

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
              data={portfolioSummary?.performanceHistory?.map((item: { date: string; value: number }) => ({
                date: item.date,
                close: item.value,
              }))}
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