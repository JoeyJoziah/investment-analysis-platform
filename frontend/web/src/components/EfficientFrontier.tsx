import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
} from '@mui/material';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Line,
  ComposedChart,
} from 'recharts';

interface EfficientFrontierProps {
  frontier: Array<{
    risk: number;
    return: number;
  }>;
  currentPortfolio: {
    risk: number;
    return: number;
  };
  optimalPortfolio?: {
    risk: number;
    return: number;
  };
  title?: string;
}

export const EfficientFrontier: React.FC<EfficientFrontierProps> = ({
  frontier,
  currentPortfolio,
  optimalPortfolio,
  title = 'ML-Based Efficient Frontier',
}) => {
  const chartData = useMemo(() => {
    return frontier.map((point, index) => ({
      risk: Math.round(point.risk * 10000) / 100, // Convert to percentage
      return: Math.round(point.return * 10000) / 100, // Convert to percentage
      index,
    }));
  }, [frontier]);

  const currentPoint = useMemo(() => {
    return {
      risk: Math.round(currentPortfolio.risk * 10000) / 100,
      return: Math.round(currentPortfolio.return * 10000) / 100,
    };
  }, [currentPortfolio]);

  const optimalPoint = useMemo(() => {
    if (!optimalPortfolio) return null;
    return {
      risk: Math.round(optimalPortfolio.risk * 10000) / 100,
      return: Math.round(optimalPortfolio.return * 10000) / 100,
    };
  }, [optimalPortfolio]);

  // Calculate improvement potential
  const improvementRisk = optimalPoint
    ? Math.round((currentPoint.risk - optimalPoint.risk) * 100) / 100
    : 0;
  const improvementReturn = optimalPoint
    ? Math.round((optimalPoint.return - currentPoint.return) * 100) / 100
    : 0;

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom fontWeight="bold">
        {title}
      </Typography>

      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart
          data={chartData}
          margin={{ top: 20, right: 20, bottom: 20, left: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="risk"
            label={{ value: 'Risk (Volatility %)', position: 'insideBottom', offset: -10 }}
            type="number"
          />
          <YAxis
            label={{ value: 'Expected Return (%)', angle: -90, position: 'insideLeft' }}
          />
          <RechartsTooltip
            formatter={(value) => `${(value as number).toFixed(2)}%`}
            labelFormatter={(label) => `Risk: ${(label as number).toFixed(2)}%`}
          />
          <Legend />

          {/* Efficient Frontier Line */}
          <Scatter
            name="Efficient Frontier"
            data={chartData}
            stroke="#8884d8"
            fill="#8884d8"
            line
            isAnimationActive={false}
          />

          {/* Current Portfolio Point */}
          <Scatter
            name="Current Portfolio"
            data={[currentPoint]}
            stroke="#ff7300"
            fill="#ff7300"
            shape="diamond"
            isAnimationActive={false}
            fillOpacity={0.8}
          />

          {/* Optimal Portfolio Point */}
          {optimalPoint && (
            <Scatter
              name="Optimal Portfolio"
              data={[optimalPoint]}
              stroke="#82ca9d"
              fill="#82ca9d"
              shape="star"
              isAnimationActive={false}
              fillOpacity={0.9}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* Portfolio Metrics */}
      <Grid container spacing={2} sx={{ mt: 3 }}>
        <Grid item xs={12} sm={6}>
          <Card variant="outlined">
            <CardContent>
              <Typography color="textSecondary" gutterBottom variant="caption">
                Current Portfolio
              </Typography>
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Risk:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {currentPoint.risk.toFixed(2)}%
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Expected Return:</Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {currentPoint.return.toFixed(2)}%
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {optimalPoint && (
          <>
            <Grid item xs={12} sm={6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography color="textSecondary" gutterBottom variant="caption">
                    Optimal Portfolio
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Risk:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {optimalPoint.risk.toFixed(2)}%
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="body2">Expected Return:</Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {optimalPoint.return.toFixed(2)}%
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            <Grid item xs={12}>
              <Card variant="outlined" sx={{ backgroundColor: '#f5f5f5' }}>
                <CardContent>
                  <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                    Improvement Potential
                  </Typography>
                  <Box sx={{ mt: 2, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2 }}>
                    <Box>
                      <Typography variant="caption" color="textSecondary">
                        Risk Reduction
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        sx={{ color: improvementRisk > 0 ? 'success.main' : 'text.primary' }}
                      >
                        {improvementRisk > 0 ? '-' : ''}
                        {Math.abs(improvementRisk).toFixed(2)}%
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="textSecondary">
                        Return Increase
                      </Typography>
                      <Typography
                        variant="h6"
                        fontWeight="bold"
                        sx={{ color: improvementReturn > 0 ? 'success.main' : 'text.primary' }}
                      >
                        {improvementReturn > 0 ? '+' : ''}
                        {improvementReturn.toFixed(2)}%
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          </>
        )}
      </Grid>

      <Typography variant="body2" color="text.secondary" sx={{ mt: 3 }}>
        The efficient frontier represents the set of optimal portfolios that offer the highest
        expected return for a given level of risk. A portfolio above the frontier is theoretically
        impossible; portfolios below the frontier are suboptimal.
      </Typography>
    </Paper>
  );
};

export default EfficientFrontier;
