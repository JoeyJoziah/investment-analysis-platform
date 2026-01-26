import React, { memo, useMemo } from 'react';
import {
  Box,
  Typography,
  Skeleton,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from 'recharts';

interface AllocationData {
  sector: string;
  value: number;
  percentage: number;
}

interface AllocationPanelProps {
  data: AllocationData[];
  isLoading?: boolean;
  type?: 'sector' | 'asset';
}

/**
 * AllocationPanel - Portfolio allocation donut chart
 *
 * Features:
 * - Interactive donut chart
 * - Center total display
 * - Sector legend with percentages
 * - Hover highlights
 *
 * WCAG 2.1 AA Compliant
 */
const AllocationPanel: React.FC<AllocationPanelProps> = ({
  data = [],
  isLoading = false,
  type = 'sector',
}) => {
  const theme = useTheme();

  // Sector colors palette
  const SECTOR_COLORS = useMemo(
    () => [
      theme.palette.primary.main,      // Blue - Technology
      theme.palette.success.main,      // Green - Healthcare
      '#7B1FA2',                        // Purple - Finance
      theme.palette.warning.main,       // Orange - Consumer
      theme.palette.error.main,         // Red - Energy
      '#0288D1',                         // Light Blue - Industrials
      '#C2185B',                         // Pink - Materials
      '#689F38',                         // Light Green - Utilities
      '#5D4037',                         // Brown - Real Estate
      '#455A64',                         // Blue Grey - Communication
    ],
    [theme.palette]
  );

  // Calculate total value
  const totalValue = useMemo(
    () => data.reduce((sum, item) => sum + item.value, 0),
    [data]
  );

  // Format currency
  const formatCurrency = (value: number): string => {
    if (value >= 1000000) {
      return `$${(value / 1000000).toFixed(2)}M`;
    }
    if (value >= 1000) {
      return `$${(value / 1000).toFixed(0)}K`;
    }
    return `$${value.toFixed(0)}`;
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const item = payload[0].payload;
      return (
        <Box
          sx={{
            backgroundColor: theme.palette.background.paper,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            p: 1.5,
            boxShadow: theme.shadows[4],
          }}
        >
          <Typography variant="subtitle2" fontWeight={600}>
            {item.sector}
          </Typography>
          <Typography
            variant="body2"
            sx={{ fontFamily: '"SF Mono", Monaco, monospace' }}
          >
            {formatCurrency(item.value)}
          </Typography>
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ fontFamily: '"SF Mono", Monaco, monospace' }}
          >
            {item.percentage.toFixed(1)}% of portfolio
          </Typography>
        </Box>
      );
    }
    return null;
  };

  // Custom legend
  const renderLegend = (props: any) => {
    const { payload } = props;

    return (
      <Box
        sx={{
          display: 'flex',
          flexWrap: 'wrap',
          justifyContent: 'center',
          gap: 1,
          mt: 2,
        }}
      >
        {payload.map((entry: any, index: number) => (
          <Chip
            key={entry.value}
            label={`${entry.value}: ${data[index]?.percentage.toFixed(1)}%`}
            size="small"
            sx={{
              backgroundColor: alpha(entry.color, 0.1),
              color: entry.color,
              fontWeight: 500,
              fontSize: '0.7rem',
              '& .MuiChip-label': {
                px: 1,
              },
            }}
            aria-label={`${entry.value}: ${data[index]?.percentage.toFixed(1)} percent, ${formatCurrency(data[index]?.value || 0)}`}
          />
        ))}
      </Box>
    );
  };

  // Custom center label
  const CenterLabel = () => (
    <text
      x="50%"
      y="50%"
      textAnchor="middle"
      dominantBaseline="middle"
      style={{
        fill: theme.palette.text.primary,
        fontWeight: 600,
        fontSize: '14px',
      }}
    >
      <tspan
        x="50%"
        dy="-0.5em"
        style={{
          fill: theme.palette.text.secondary,
          fontSize: '11px',
          fontWeight: 400,
        }}
      >
        Total
      </tspan>
      <tspan
        x="50%"
        dy="1.2em"
        style={{
          fontFamily: '"SF Mono", Monaco, monospace',
          fontSize: '16px',
        }}
      >
        {formatCurrency(totalValue)}
      </tspan>
    </text>
  );

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Skeleton variant="text" width={120} height={28} sx={{ mb: 2 }} />
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: 220,
          }}
        >
          <Skeleton variant="circular" width={180} height={180} />
        </Box>
        <Box
          sx={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 1,
            mt: 2,
          }}
        >
          {[...Array(5)].map((_, i) => (
            <Skeleton
              key={i}
              variant="rectangular"
              width={80}
              height={24}
              sx={{ borderRadius: 1 }}
            />
          ))}
        </Box>
      </Box>
    );
  }

  // Empty state
  if (data.length === 0) {
    return (
      <Box>
        <Typography variant="h6" component="h2" gutterBottom>
          {type === 'sector' ? 'Sector' : 'Asset'} Allocation
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: 220,
            color: 'text.secondary',
          }}
        >
          <Typography variant="body1">No allocation data</Typography>
          <Typography variant="caption">
            Add positions to see your portfolio allocation
          </Typography>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Typography variant="h6" component="h2" gutterBottom>
        {type === 'sector' ? 'Sector' : 'Asset'} Allocation
      </Typography>

      {/* Donut Chart */}
      <Box
        role="img"
        aria-label={`Portfolio allocation chart showing ${data.length} ${type}s. Total value ${formatCurrency(totalValue)}.`}
        sx={{
          height: 220,
          position: 'relative',
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={85}
              paddingAngle={2}
              dataKey="percentage"
              nameKey="sector"
              animationDuration={800}
              animationEasing="ease-out"
            >
              {data.map((entry, index) => (
                <Cell
                  key={entry.sector}
                  fill={SECTOR_COLORS[index % SECTOR_COLORS.length]}
                  stroke={theme.palette.background.paper}
                  strokeWidth={2}
                  style={{
                    cursor: 'pointer',
                    transition: 'all 0.2s ease-out',
                  }}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend content={renderLegend} />
            {/* Center label via SVG */}
            <CenterLabel />
          </PieChart>
        </ResponsiveContainer>
      </Box>

      {/* Diversification score */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          mt: 2,
          pt: 2,
          borderTop: `1px solid ${theme.palette.divider}`,
        }}
      >
        <Box sx={{ textAlign: 'center' }}>
          <Typography
            variant="caption"
            color="text.secondary"
            display="block"
          >
            Diversification Score
          </Typography>
          <Typography
            variant="h6"
            sx={{
              fontWeight: 700,
              color:
                data.length >= 5
                  ? theme.palette.success.main
                  : data.length >= 3
                    ? theme.palette.warning.main
                    : theme.palette.error.main,
            }}
          >
            {Math.min(data.length * 15 + 25, 100)}/100
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {data.length >= 5
              ? 'Well diversified'
              : data.length >= 3
                ? 'Moderately diversified'
                : 'Consider diversifying'}
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default memo(AllocationPanel);
