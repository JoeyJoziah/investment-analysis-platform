import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  LinearProgress,
  Grid,
  Tooltip,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { Warning as WarningIcon } from '@mui/icons-material';

interface RiskComponent {
  symbol: string;
  riskContribution: number;
  volatility: number;
  beta?: number;
  concentration?: number;
}

interface RiskDecompositionProps {
  components: RiskComponent[];
  totalRisk: number;
  diversificationScore?: number;
  title?: string;
}

const getRiskColor = (percentage: number): string => {
  if (percentage > 40) return '#d32f2f'; // Red for high
  if (percentage > 25) return '#f57c00'; // Orange for medium
  return '#388e3c'; // Green for low
};

export const RiskDecomposition: React.FC<RiskDecompositionProps> = ({
  components,
  totalRisk,
  diversificationScore = 60,
  title = 'Risk Decomposition Analysis',
}) => {
  const chartData = useMemo(() => {
    return components
      .map((component) => ({
        symbol: component.symbol,
        riskContribution: Math.round(component.riskContribution * 10000) / 100,
        volatility: Math.round(component.volatility * 10000) / 100,
      }))
      .sort((a, b) => b.riskContribution - a.riskContribution);
  }, [components]);

  const highRiskComponents = useMemo(() => {
    return components.filter((c) => c.riskContribution > 0.25);
  }, [components]);

  const concentrationRisk = useMemo(() => {
    const top3 = components
      .sort((a, b) => b.riskContribution - a.riskContribution)
      .slice(0, 3)
      .reduce((sum, c) => sum + c.riskContribution, 0);
    return Math.round(top3 * 10000) / 100;
  }, [components]);

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom fontWeight="bold">
        {title}
      </Typography>

      {/* Main Chart */}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          margin={{ top: 20, right: 20, bottom: 20, left: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="symbol" />
          <YAxis
            label={{ value: 'Risk Contribution (%)', angle: -90, position: 'insideLeft' }}
          />
          <RechartsTooltip
            formatter={(value) => `${(value as number).toFixed(2)}%`}
            labelFormatter={(label) => `${label}`}
          />
          <Legend />
          <Bar name="Risk Contribution" dataKey="riskContribution" fill="#8884d8">
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getRiskColor(entry.riskContribution)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Risk Metrics Grid */}
      <Grid container spacing={2} sx={{ mt: 3 }}>
        {/* Total Risk */}
        <Grid item xs={12} sm={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="caption">
                Portfolio Volatility
              </Typography>
              <Typography variant="h5" fontWeight="bold" sx={{ mt: 1 }}>
                {(totalRisk * 100).toFixed(2)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(totalRisk * 100, 100)}
                sx={{ mt: 2 }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Diversification Score */}
        <Grid item xs={12} sm={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="caption">
                Diversification Score
              </Typography>
              <Typography variant="h5" fontWeight="bold" sx={{ mt: 1 }}>
                {Math.round(diversificationScore)}/100
              </Typography>
              <LinearProgress
                variant="determinate"
                value={diversificationScore}
                sx={{
                  mt: 2,
                  backgroundColor: '#e0e0e0',
                  '& .MuiLinearProgress-bar': {
                    backgroundColor:
                      diversificationScore > 70
                        ? '#388e3c'
                        : diversificationScore > 50
                          ? '#f57c00'
                          : '#d32f2f',
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        {/* Concentration Risk */}
        <Grid item xs={12} sm={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="caption">
                Top 3 Concentration
              </Typography>
              <Typography variant="h5" fontWeight="bold" sx={{ mt: 1 }}>
                {concentrationRisk.toFixed(2)}%
              </Typography>
              <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                Risk from 3 largest positions
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Number of Components */}
        <Grid item xs={12} sm={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="caption">
                Portfolio Holdings
              </Typography>
              <Typography variant="h5" fontWeight="bold" sx={{ mt: 1 }}>
                {components.length}
              </Typography>
              <Typography variant="caption" color="textSecondary" sx={{ mt: 1, display: 'block' }}>
                Total number of positions
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Risk Component Details */}
      <Box sx={{ mt: 4 }}>
        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
          Risk Components by Position
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
          {chartData.map((component) => {
            const riskPercent = (component.riskContribution / totalRisk) * 100;
            const original = components.find((c) => c.symbol === component.symbol);

            return (
              <Box key={component.symbol}>
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mb: 0.5,
                  }}
                >
                  <Typography variant="body2" fontWeight="bold">
                    {component.symbol}
                  </Typography>
                  <Tooltip title={`${riskPercent.toFixed(1)}% of total portfolio risk`}>
                    <Typography variant="body2" fontWeight="bold">
                      {component.riskContribution.toFixed(2)}%
                    </Typography>
                  </Tooltip>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.min(riskPercent, 100)}
                  sx={{
                    backgroundColor: '#f0f0f0',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getRiskColor(component.riskContribution),
                    },
                  }}
                />
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    mt: 0.5,
                    fontSize: '0.75rem',
                    color: 'text.secondary',
                  }}
                >
                  <span>Volatility: {component.volatility.toFixed(2)}%</span>
                  {original?.beta && <span>Beta: {original.beta.toFixed(2)}</span>}
                </Box>
              </Box>
            );
          })}
        </Box>
      </Box>

      {/* Warnings */}
      {highRiskComponents.length > 0 && (
        <Box
          sx={{
            mt: 3,
            p: 2,
            backgroundColor: '#fff3e0',
            borderLeft: '4px solid #f57c00',
            borderRadius: '4px',
            display: 'flex',
            gap: 2,
          }}
        >
          <WarningIcon sx={{ color: '#f57c00', mt: 0.5 }} />
          <Box>
            <Typography variant="subtitle2" fontWeight="bold" color="#e65100">
              High Risk Components
            </Typography>
            <Typography variant="body2" color="#e65100" sx={{ mt: 1 }}>
              {highRiskComponents.map((c) => c.symbol).join(', ')} contribute more than 25% of
              portfolio risk. Consider rebalancing to reduce concentration risk.
            </Typography>
          </Box>
        </Box>
      )}

      <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
        Risk decomposition shows how much each position contributes to overall portfolio
        volatility. Diversification score measures how well-distributed risk is across holdings.
        Lower concentration and higher diversification generally indicate more stable portfolios.
      </Typography>
    </Paper>
  );
};

export default RiskDecomposition;
