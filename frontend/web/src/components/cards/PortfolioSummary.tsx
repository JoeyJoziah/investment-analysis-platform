import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  LinearProgress,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  Button,
  IconButton,
  Tooltip,
  useTheme,
  alpha,
  Grid,
  Paper
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  PieChart,
  ShowChart,
  Timeline,
  AttachMoney,
  ArrowUpward,
  ArrowDownward,
  MoreVert,
  Assessment,
  Equalizer,
  Speed
} from '@mui/icons-material';
import { PieChart as RechartssPieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';

interface PortfolioSummaryProps {
  summary?: {
    totalValue: number;
    totalCost: number;
    totalReturn: number;
    totalReturnPercent: number;
    dayChange: number;
    dayChangePercent: number;
    weekChange: number;
    monthChange: number;
    yearChange: number;
    activePositions: number;
    performanceHistory?: Array<{ date: string; value: number }>;
    topGainers?: Array<{
      ticker: string;
      name: string;
      change: number;
      changePercent: number;
      value: number;
    }>;
    topLosers?: Array<{
      ticker: string;
      name: string;
      change: number;
      changePercent: number;
      value: number;
    }>;
    allocation?: Array<{
      sector: string;
      value: number;
      percentage: number;
    }>;
    riskMetrics?: {
      sharpeRatio: number;
      beta: number;
      standardDeviation: number;
      maxDrawdown: number;
    };
    diversificationScore?: number;
    cashBalance?: number;
    marginUsed?: number;
  };
  compact?: boolean;
}

const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({ summary, compact = false }) => {
  const theme = useTheme();

  if (!summary) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body2" color="textSecondary" align="center">
            No portfolio data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    const formatted = value.toFixed(2);
    return value >= 0 ? `+${formatted}%` : `${formatted}%`;
  };

  const getChangeColor = (value: number) => {
    return value >= 0 ? theme.palette.success.main : theme.palette.error.main;
  };

  // Prepare allocation data for pie chart
  const pieChartData = summary.allocation || [
    { sector: 'Technology', value: 35, percentage: 35 },
    { sector: 'Healthcare', value: 20, percentage: 20 },
    { sector: 'Finance', value: 15, percentage: 15 },
    { sector: 'Consumer', value: 15, percentage: 15 },
    { sector: 'Energy', value: 10, percentage: 10 },
    { sector: 'Other', value: 5, percentage: 5 }
  ];

  const COLORS = [
    theme.palette.primary.main,
    theme.palette.secondary.main,
    theme.palette.success.main,
    theme.palette.warning.main,
    theme.palette.info.main,
    theme.palette.error.main
  ];

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload[0]) {
      return (
        <Paper sx={{ p: 1 }}>
          <Typography variant="caption">
            {payload[0].name}: {payload[0].value}%
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  if (compact) {
    return (
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Portfolio Summary
          </Typography>
          <Box display="flex" justifyContent="space-between" mb={2}>
            <Box>
              <Typography variant="body2" color="textSecondary">
                Total Value
              </Typography>
              <Typography variant="h5">
                {formatCurrency(summary.totalValue)}
              </Typography>
            </Box>
            <Box textAlign="right">
              <Typography variant="body2" color="textSecondary">
                Total Return
              </Typography>
              <Typography
                variant="h6"
                sx={{ color: getChangeColor(summary.totalReturn) }}
              >
                {formatPercentage(summary.totalReturnPercent)}
              </Typography>
            </Box>
          </Box>
          <LinearProgress
            variant="determinate"
            value={Math.min(Math.abs(summary.totalReturnPercent), 100)}
            sx={{
              height: 8,
              borderRadius: 4,
              bgcolor: alpha(theme.palette.grey[300], 0.3),
              '& .MuiLinearProgress-bar': {
                bgcolor: getChangeColor(summary.totalReturn)
              }
            }}
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">Portfolio Overview</Typography>
          <IconButton size="small" aria-label="Portfolio options menu">
            <MoreVert />
          </IconButton>
        </Box>

        {/* Main metrics */}
        <Box mb={3}>
          <Box display="flex" justifyContent="space-between" alignItems="baseline" mb={1}>
            <Typography variant="body2" color="textSecondary">
              Total Value
            </Typography>
            <Typography variant="h4">
              {formatCurrency(summary.totalValue)}
            </Typography>
          </Box>
          
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Box display="flex" alignItems="center" gap={1}>
              {summary.dayChange >= 0 ? (
                <TrendingUp sx={{ color: theme.palette.success.main }} />
              ) : (
                <TrendingDown sx={{ color: theme.palette.error.main }} />
              )}
              <Box>
                <Typography
                  variant="body1"
                  sx={{ color: getChangeColor(summary.dayChange) }}
                >
                  {formatCurrency(Math.abs(summary.dayChange))}
                </Typography>
                <Typography variant="caption" color="textSecondary">
                  Today {formatPercentage(summary.dayChangePercent)}
                </Typography>
              </Box>
            </Box>
            
            <Box textAlign="right">
              <Typography
                variant="body1"
                sx={{ color: getChangeColor(summary.totalReturn) }}
              >
                {formatCurrency(Math.abs(summary.totalReturn))}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Total {formatPercentage(summary.totalReturnPercent)}
              </Typography>
            </Box>
          </Box>

          <LinearProgress
            variant="determinate"
            value={Math.min(Math.abs(summary.totalReturnPercent), 100)}
            sx={{
              height: 8,
              borderRadius: 4,
              bgcolor: alpha(theme.palette.grey[300], 0.3),
              '& .MuiLinearProgress-bar': {
                bgcolor: getChangeColor(summary.totalReturn),
                borderRadius: 4
              }
            }}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Performance periods */}
        <Box mb={3}>
          <Typography variant="subtitle2" gutterBottom>
            Performance
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Box>
                <Typography variant="caption" color="textSecondary">
                  1 Week
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: getChangeColor(summary.weekChange || 0) }}
                >
                  {formatPercentage(summary.weekChange || 0)}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box>
                <Typography variant="caption" color="textSecondary">
                  1 Month
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: getChangeColor(summary.monthChange || 0) }}
                >
                  {formatPercentage(summary.monthChange || 0)}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box>
                <Typography variant="caption" color="textSecondary">
                  1 Year
                </Typography>
                <Typography
                  variant="body2"
                  sx={{ color: getChangeColor(summary.yearChange || 0) }}
                >
                  {formatPercentage(summary.yearChange || 0)}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box>
                <Typography variant="caption" color="textSecondary">
                  Positions
                </Typography>
                <Typography variant="body2">
                  {summary.activePositions}
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Allocation pie chart */}
        {pieChartData.length > 0 && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Allocation
            </Typography>
            <ResponsiveContainer width="100%" height={200}>
              <RechartssPieChart>
                <Pie
                  data={pieChartData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={2}
                  dataKey="percentage"
                >
                  {pieChartData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip content={<CustomTooltip />} />
              </RechartssPieChart>
            </ResponsiveContainer>
            <Box display="flex" flexWrap="wrap" gap={1} justifyContent="center">
              {pieChartData.map((sector, index) => (
                <Chip
                  key={sector.sector}
                  label={`${sector.sector}: ${sector.percentage}%`}
                  size="small"
                  sx={{
                    bgcolor: alpha(COLORS[index % COLORS.length], 0.1),
                    color: COLORS[index % COLORS.length]
                  }}
                />
              ))}
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        {/* Top movers */}
        {(summary.topGainers || summary.topLosers) && (
          <Box mb={3}>
            <Typography variant="subtitle2" gutterBottom>
              Top Movers
            </Typography>
            
            {summary.topGainers && summary.topGainers.length > 0 && (
              <Box mb={2}>
                <Typography variant="caption" color="textSecondary">
                  Gainers
                </Typography>
                <List dense>
                  {summary.topGainers.slice(0, 3).map((gainer) => (
                    <ListItem key={gainer.ticker} disableGutters>
                      <ListItemAvatar>
                        <Avatar sx={{ 
                          bgcolor: alpha(theme.palette.success.main, 0.1),
                          width: 32,
                          height: 32
                        }}>
                          <ArrowUpward sx={{ 
                            color: theme.palette.success.main,
                            fontSize: 18
                          }} />
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={gainer.ticker}
                        secondary={gainer.name}
                        primaryTypographyProps={{ variant: 'body2' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                      <Typography
                        variant="body2"
                        sx={{ color: theme.palette.success.main }}
                      >
                        {formatPercentage(gainer.changePercent)}
                      </Typography>
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}

            {summary.topLosers && summary.topLosers.length > 0 && (
              <Box>
                <Typography variant="caption" color="textSecondary">
                  Losers
                </Typography>
                <List dense>
                  {summary.topLosers.slice(0, 3).map((loser) => (
                    <ListItem key={loser.ticker} disableGutters>
                      <ListItemAvatar>
                        <Avatar sx={{ 
                          bgcolor: alpha(theme.palette.error.main, 0.1),
                          width: 32,
                          height: 32
                        }}>
                          <ArrowDownward sx={{ 
                            color: theme.palette.error.main,
                            fontSize: 18
                          }} />
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={loser.ticker}
                        secondary={loser.name}
                        primaryTypographyProps={{ variant: 'body2' }}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                      <Typography
                        variant="body2"
                        sx={{ color: theme.palette.error.main }}
                      >
                        {formatPercentage(loser.changePercent)}
                      </Typography>
                    </ListItem>
                  ))}
                </List>
              </Box>
            )}
          </Box>
        )}

        {/* Risk metrics */}
        {summary.riskMetrics && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                Risk Metrics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <Speed sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                    <Typography variant="caption" color="textSecondary">
                      Sharpe Ratio
                    </Typography>
                  </Box>
                  <Typography variant="body2">
                    {summary.riskMetrics.sharpeRatio.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <Equalizer sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                    <Typography variant="caption" color="textSecondary">
                      Beta
                    </Typography>
                  </Box>
                  <Typography variant="body2">
                    {summary.riskMetrics.beta.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <ShowChart sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                    <Typography variant="caption" color="textSecondary">
                      Std Dev
                    </Typography>
                  </Box>
                  <Typography variant="body2">
                    {summary.riskMetrics.standardDeviation.toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Box display="flex" alignItems="center" gap={0.5}>
                    <TrendingDown sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                    <Typography variant="caption" color="textSecondary">
                      Max DD
                    </Typography>
                  </Box>
                  <Typography variant="body2">
                    {summary.riskMetrics.maxDrawdown.toFixed(2)}%
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          </>
        )}

        {/* Action buttons */}
        <Box mt={3} display="flex" gap={1}>
          <Button
            variant="outlined"
            size="small"
            startIcon={<Assessment />}
            fullWidth
          >
            Details
          </Button>
          <Button
            variant="contained"
            size="small"
            startIcon={<AttachMoney />}
            fullWidth
          >
            Rebalance
          </Button>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PortfolioSummary;