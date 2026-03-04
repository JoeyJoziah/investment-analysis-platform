import React from 'react';
import {
  Grid,
  Typography,
} from '@mui/material';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip as RechartsTooltip,
} from 'recharts';
import CorrelationMatrix from './CorrelationMatrix';
import EfficientFrontier from './EfficientFrontier';
import RiskDecomposition from './RiskDecomposition';
import type { Position, PortfolioMetrics } from '../../store/slices/portfolioSlice';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

export interface AllocationTabProps {
  metrics: PortfolioMetrics | null;
}

export const AllocationTabContent: React.FC<AllocationTabProps> = ({ metrics }) => {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Typography variant="h6" gutterBottom>
          Sector Allocation
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <RechartsPieChart>
            <Pie
              data={Object.entries(metrics?.diversification?.sector || {}).map(
                ([sector, value]) => ({ name: sector, value })
              )}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {Object.entries(metrics?.diversification?.sector || {}).map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <RechartsTooltip />
            <Legend />
          </RechartsPieChart>
        </ResponsiveContainer>
      </Grid>
      <Grid item xs={12} md={6}>
        <Typography variant="h6" gutterBottom>
          Asset Type Allocation
        </Typography>
        <ResponsiveContainer width="100%" height={300}>
          <RechartsPieChart>
            <Pie
              data={Object.entries(metrics?.diversification?.asset || {}).map(
                ([asset, value]) => ({ name: asset, value })
              )}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#82ca9d"
              dataKey="value"
            >
              {Object.entries(metrics?.diversification?.asset || {}).map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <RechartsTooltip />
            <Legend />
          </RechartsPieChart>
        </ResponsiveContainer>
      </Grid>
    </Grid>
  );
};

export interface RiskAnalysisTabProps {
  metrics: PortfolioMetrics | null;
  positions: Position[];
  totalValue: number;
  diversificationScore: number;
}

export const RiskAnalysisTabContent: React.FC<RiskAnalysisTabProps> = ({
  metrics,
  positions,
  totalValue,
  diversificationScore,
}) => {
  return (
    <Grid container spacing={3}>
      {/* Correlation Matrix */}
      <Grid item xs={12} lg={6}>
        <CorrelationMatrix
          correlations={(metrics as any)?.correlationMatrix || {}}
          title="Asset Correlation Matrix"
        />
      </Grid>

      {/* Efficient Frontier */}
      <Grid item xs={12} lg={6}>
        <EfficientFrontier
          frontier={(metrics as any)?.efficientFrontier?.points || []}
          currentPortfolio={(metrics as any)?.efficientFrontier?.currentPosition || { risk: 0.15, return: 0.12 }}
          optimalPortfolio={(metrics as any)?.efficientFrontier?.optimalPosition}
          title="ML-Based Efficient Frontier"
        />
      </Grid>

      {/* Risk Decomposition */}
      <Grid item xs={12}>
        <RiskDecomposition
          components={positions.map((p) => ({
            symbol: p.ticker,
            riskContribution: p.marketValue / (totalValue || 1),
            volatility: (p.currentPrice * 0.15) / 100,
            beta: 1.0,
          }))}
          totalRisk={0.15}
          diversificationScore={diversificationScore}
          title="Risk Decomposition Analysis"
        />
      </Grid>
    </Grid>
  );
};

export default { AllocationTabContent, RiskAnalysisTabContent };
