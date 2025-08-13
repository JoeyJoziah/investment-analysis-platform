import React from 'react';
import {
  Card,
  CardContent,
  Box,
  Typography,
  LinearProgress,
  Chip,
  Alert,
  AlertTitle,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Tooltip,
  Button,
  Collapse,
  useTheme,
  alpha
} from '@mui/material';
import {
  AttachMoney,
  TrendingUp,
  Warning,
  CheckCircle,
  Error,
  Info,
  ExpandMore,
  ExpandLess,
  Refresh,
  Settings,
  Speed,
  DataUsage,
  CloudQueue,
  Api,
  Storage,
  Schedule,
  ShowChart
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

interface CostMetricsProps {
  metrics?: {
    currentMonthCost: number;
    projectedMonthCost: number;
    dailyAverage: number;
    monthlyBudget: number;
    apiUsage: {
      provider: string;
      used: number;
      limit: number;
      cost: number;
      resetDate?: string;
    }[];
    costBreakdown: {
      category: string;
      amount: number;
      percentage: number;
    }[];
    alerts?: {
      type: 'warning' | 'error' | 'info';
      message: string;
    }[];
    lastUpdated?: string;
    costTrend?: {
      date: string;
      cost: number;
    }[];
    savingsMode?: boolean;
    emergencyMode?: boolean;
  };
  onRefresh?: () => void;
  onSettings?: () => void;
}

const CostMonitor: React.FC<CostMetricsProps> = ({ metrics, onRefresh, onSettings }) => {
  const theme = useTheme();
  const [expanded, setExpanded] = React.useState(false);

  if (!metrics) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body2" color="textSecondary" align="center">
            No cost metrics available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const budgetUsagePercent = (metrics.currentMonthCost / metrics.monthlyBudget) * 100;
  const isOverBudget = budgetUsagePercent > 100;
  const isNearLimit = budgetUsagePercent > 80;

  const getStatusColor = () => {
    if (isOverBudget) return theme.palette.error.main;
    if (isNearLimit) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  const getStatusIcon = () => {
    if (isOverBudget) return <Error />;
    if (isNearLimit) return <Warning />;
    return <CheckCircle />;
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

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
            {payload[0].name}: {formatCurrency(payload[0].value)}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  const getApiStatusColor = (usage: number, limit: number) => {
    const percent = (usage / limit) * 100;
    if (percent >= 90) return theme.palette.error.main;
    if (percent >= 70) return theme.palette.warning.main;
    return theme.palette.success.main;
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Box display="flex" alignItems="center" gap={1}>
            <AttachMoney color="primary" />
            <Typography variant="h6">Cost Monitor</Typography>
          </Box>
          <Box display="flex" gap={0.5}>
            <Tooltip title="Refresh metrics">
              <IconButton size="small" onClick={onRefresh}>
                <Refresh />
              </IconButton>
            </Tooltip>
            <Tooltip title="Settings">
              <IconButton size="small" onClick={onSettings}>
                <Settings />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Status indicators */}
        <Box display="flex" gap={1} mb={2}>
          {metrics.savingsMode && (
            <Chip
              label="Savings Mode"
              size="small"
              icon={<DataUsage />}
              sx={{
                bgcolor: alpha(theme.palette.info.main, 0.1),
                color: theme.palette.info.main
              }}
            />
          )}
          {metrics.emergencyMode && (
            <Chip
              label="Emergency Mode"
              size="small"
              icon={<Warning />}
              sx={{
                bgcolor: alpha(theme.palette.error.main, 0.1),
                color: theme.palette.error.main
              }}
            />
          )}
        </Box>

        {/* Budget overview */}
        <Box mb={3}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
            <Typography variant="body2" color="textSecondary">
              Monthly Budget
            </Typography>
            <Box display="flex" alignItems="center" gap={0.5}>
              {getStatusIcon()}
              <Typography variant="h6" sx={{ color: getStatusColor() }}>
                {formatCurrency(metrics.currentMonthCost)} / {formatCurrency(metrics.monthlyBudget)}
              </Typography>
            </Box>
          </Box>
          <LinearProgress
            variant="determinate"
            value={Math.min(budgetUsagePercent, 100)}
            sx={{
              height: 10,
              borderRadius: 5,
              bgcolor: alpha(theme.palette.grey[300], 0.3),
              '& .MuiLinearProgress-bar': {
                bgcolor: getStatusColor(),
                borderRadius: 5
              }
            }}
          />
          <Box display="flex" justifyContent="space-between" mt={1}>
            <Typography variant="caption" color="textSecondary">
              {budgetUsagePercent.toFixed(1)}% used
            </Typography>
            <Typography variant="caption" color="textSecondary">
              {Math.max(0, metrics.monthlyBudget - metrics.currentMonthCost).toFixed(2)} remaining
            </Typography>
          </Box>
        </Box>

        {/* Alerts */}
        {metrics.alerts && metrics.alerts.length > 0 && (
          <Box mb={2}>
            {metrics.alerts.map((alert, index) => (
              <Alert 
                key={index} 
                severity={alert.type}
                sx={{ mb: 1 }}
                icon={alert.type === 'warning' ? <Warning /> : alert.type === 'error' ? <Error /> : <Info />}
              >
                {alert.message}
              </Alert>
            ))}
          </Box>
        )}

        {/* Key metrics */}
        <Grid container spacing={2} mb={3}>
          <Grid item xs={6}>
            <Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <TrendingUp sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                <Typography variant="caption" color="textSecondary">
                  Projected Month
                </Typography>
              </Box>
              <Typography variant="h6">
                {formatCurrency(metrics.projectedMonthCost)}
              </Typography>
              {metrics.projectedMonthCost > metrics.monthlyBudget && (
                <Typography variant="caption" sx={{ color: theme.palette.error.main }}>
                  Over budget by {formatCurrency(metrics.projectedMonthCost - metrics.monthlyBudget)}
                </Typography>
              )}
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box>
              <Box display="flex" alignItems="center" gap={0.5}>
                <Schedule sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                <Typography variant="caption" color="textSecondary">
                  Daily Average
                </Typography>
              </Box>
              <Typography variant="h6">
                {formatCurrency(metrics.dailyAverage)}
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Target: {formatCurrency(metrics.monthlyBudget / 30)}
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* API Usage */}
        <Box mb={3}>
          <Typography variant="subtitle2" gutterBottom>
            API Usage
          </Typography>
          <List dense>
            {metrics.apiUsage.map((api) => (
              <ListItem key={api.provider} disableGutters>
                <ListItemIcon sx={{ minWidth: 32 }}>
                  <Api sx={{ fontSize: 18, color: getApiStatusColor(api.used, api.limit) }} />
                </ListItemIcon>
                <ListItemText
                  primary={
                    <Box display="flex" justifyContent="space-between">
                      <Typography variant="body2">{api.provider}</Typography>
                      <Typography variant="caption" color="textSecondary">
                        {api.used}/{api.limit} calls
                      </Typography>
                    </Box>
                  }
                  secondary={
                    <Box>
                      <LinearProgress
                        variant="determinate"
                        value={Math.min((api.used / api.limit) * 100, 100)}
                        sx={{
                          height: 4,
                          borderRadius: 2,
                          bgcolor: alpha(theme.palette.grey[300], 0.3),
                          '& .MuiLinearProgress-bar': {
                            bgcolor: getApiStatusColor(api.used, api.limit),
                            borderRadius: 2
                          }
                        }}
                      />
                      <Box display="flex" justifyContent="space-between" mt={0.5}>
                        <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
                          Cost: {formatCurrency(api.cost)}
                        </Typography>
                        {api.resetDate && (
                          <Typography variant="caption" sx={{ fontSize: '0.65rem' }}>
                            Resets: {api.resetDate}
                          </Typography>
                        )}
                      </Box>
                    </Box>
                  }
                />
              </ListItem>
            ))}
          </List>
        </Box>

        {/* Expandable section */}
        <Box>
          <Button
            fullWidth
            size="small"
            onClick={() => setExpanded(!expanded)}
            endIcon={expanded ? <ExpandLess /> : <ExpandMore />}
          >
            {expanded ? 'Hide Details' : 'Show Details'}
          </Button>
          
          <Collapse in={expanded}>
            <Box mt={2}>
              {/* Cost breakdown pie chart */}
              {metrics.costBreakdown && metrics.costBreakdown.length > 0 && (
                <Box mb={3}>
                  <Typography variant="subtitle2" gutterBottom>
                    Cost Breakdown
                  </Typography>
                  <ResponsiveContainer width="100%" height={200}>
                    <PieChart>
                      <Pie
                        data={metrics.costBreakdown}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={2}
                        dataKey="amount"
                      >
                        {metrics.costBreakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                  <Box display="flex" flexWrap="wrap" gap={1} justifyContent="center">
                    {metrics.costBreakdown.map((item, index) => (
                      <Chip
                        key={item.category}
                        label={`${item.category}: ${formatCurrency(item.amount)}`}
                        size="small"
                        sx={{
                          bgcolor: alpha(COLORS[index % COLORS.length], 0.1),
                          color: COLORS[index % COLORS.length],
                          fontSize: '0.7rem'
                        }}
                      />
                    ))}
                  </Box>
                </Box>
              )}

              {/* Cost trend chart */}
              {metrics.costTrend && metrics.costTrend.length > 0 && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Daily Cost Trend
                  </Typography>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={metrics.costTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                      <XAxis 
                        dataKey="date" 
                        tick={{ fontSize: 10 }}
                        stroke={theme.palette.text.secondary}
                      />
                      <YAxis 
                        tick={{ fontSize: 10 }}
                        stroke={theme.palette.text.secondary}
                        tickFormatter={(value) => `$${value}`}
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Bar 
                        dataKey="cost" 
                        fill={theme.palette.primary.main}
                        radius={[4, 4, 0, 0]}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Box>
              )}

              {/* Last updated */}
              {metrics.lastUpdated && (
                <Box mt={2} textAlign="center">
                  <Typography variant="caption" color="textSecondary">
                    Last updated: {metrics.lastUpdated}
                  </Typography>
                </Box>
              )}
            </Box>
          </Collapse>
        </Box>
      </CardContent>
    </Card>
  );
};

export default CostMonitor;