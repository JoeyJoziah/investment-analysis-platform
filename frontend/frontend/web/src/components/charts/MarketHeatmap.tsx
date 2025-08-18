import React, { useMemo } from 'react';
import { Box, Typography, Tooltip, Paper, useTheme, alpha } from '@mui/material';
import { Treemap, ResponsiveContainer } from 'recharts';

interface MarketHeatmapProps {
  data?: Array<{
    name: string;
    ticker?: string;
    value: number;
    change: number;
    volume?: number;
    sector?: string;
  }>;
  height?: number;
  groupBy?: 'sector' | 'market_cap' | 'volume';
}

const MarketHeatmap: React.FC<MarketHeatmapProps> = ({
  data = [],
  height = 400,
  groupBy = 'sector'
}) => {
  const theme = useTheme();

  const processedData = useMemo(() => {
    if (!data || data.length === 0) {
      // Sample data for demonstration
      return {
        name: 'Market',
        children: [
          {
            name: 'Technology',
            children: [
              { name: 'AAPL', value: 3000, change: 2.5, ticker: 'AAPL' },
              { name: 'MSFT', value: 2800, change: 1.8, ticker: 'MSFT' },
              { name: 'GOOGL', value: 2200, change: -0.5, ticker: 'GOOGL' },
              { name: 'META', value: 1500, change: 3.2, ticker: 'META' },
              { name: 'NVDA', value: 1800, change: 4.5, ticker: 'NVDA' }
            ]
          },
          {
            name: 'Healthcare',
            children: [
              { name: 'JNJ', value: 1600, change: 0.8, ticker: 'JNJ' },
              { name: 'PFE', value: 1200, change: -1.2, ticker: 'PFE' },
              { name: 'UNH', value: 1400, change: 1.5, ticker: 'UNH' },
              { name: 'CVS', value: 800, change: -0.3, ticker: 'CVS' }
            ]
          },
          {
            name: 'Finance',
            children: [
              { name: 'JPM', value: 1800, change: 1.2, ticker: 'JPM' },
              { name: 'BAC', value: 1300, change: 0.5, ticker: 'BAC' },
              { name: 'WFC', value: 900, change: -0.8, ticker: 'WFC' },
              { name: 'GS', value: 1100, change: 2.1, ticker: 'GS' }
            ]
          },
          {
            name: 'Consumer',
            children: [
              { name: 'AMZN', value: 2500, change: 1.9, ticker: 'AMZN' },
              { name: 'TSLA', value: 2000, change: -2.3, ticker: 'TSLA' },
              { name: 'WMT', value: 1500, change: 0.6, ticker: 'WMT' },
              { name: 'HD', value: 1200, change: 1.1, ticker: 'HD' }
            ]
          },
          {
            name: 'Energy',
            children: [
              { name: 'XOM', value: 1400, change: -1.5, ticker: 'XOM' },
              { name: 'CVX', value: 1200, change: -0.9, ticker: 'CVX' },
              { name: 'COP', value: 700, change: -2.1, ticker: 'COP' }
            ]
          },
          {
            name: 'Industrial',
            children: [
              { name: 'BA', value: 1000, change: 0.3, ticker: 'BA' },
              { name: 'CAT', value: 900, change: 1.7, ticker: 'CAT' },
              { name: 'GE', value: 600, change: -0.4, ticker: 'GE' }
            ]
          }
        ]
      };
    }

    // Group data by the specified criteria
    const grouped: { [key: string]: any[] } = {};
    
    data.forEach(item => {
      const groupKey = item.sector || 'Other';
      if (!grouped[groupKey]) {
        grouped[groupKey] = [];
      }
      grouped[groupKey].push({
        name: item.ticker || item.name,
        value: Math.abs(item.value), // Ensure positive values for size
        change: item.change,
        ticker: item.ticker,
        originalValue: item.value
      });
    });

    return {
      name: 'Market',
      children: Object.keys(grouped).map(key => ({
        name: key,
        children: grouped[key]
      }))
    };
  }, [data, groupBy]);

  const getColor = (change: number) => {
    const intensity = Math.min(Math.abs(change) / 5, 1); // Cap at 5% for max intensity
    
    if (change > 0) {
      // Green gradient for positive changes
      return alpha(theme.palette.success.main, 0.3 + intensity * 0.7);
    } else if (change < 0) {
      // Red gradient for negative changes
      return alpha(theme.palette.error.main, 0.3 + intensity * 0.7);
    } else {
      // Gray for no change
      return alpha(theme.palette.grey[500], 0.5);
    }
  };

  const CustomTreemapContent = ({ 
    root, 
    depth, 
    x, 
    y, 
    width, 
    height, 
    name, 
    change,
    ticker
  }: any) => {
    const fontSize = width > 80 && height > 40 ? 12 : 10;
    const showDetails = width > 60 && height > 50;
    
    return (
      <Tooltip
        title={
          <Box>
            <Typography variant="body2">{ticker || name}</Typography>
            <Typography variant="caption">
              Change: {change >= 0 ? '+' : ''}{change?.toFixed(2)}%
            </Typography>
          </Box>
        }
      >
        <g>
          <rect
            x={x}
            y={y}
            width={width}
            height={height}
            style={{
              fill: getColor(change || 0),
              stroke: theme.palette.divider,
              strokeWidth: 2,
              strokeOpacity: 0.3,
              cursor: 'pointer'
            }}
          />
          {showDetails && (
            <>
              <text
                x={x + width / 2}
                y={y + height / 2 - 8}
                textAnchor="middle"
                fill={theme.palette.text.primary}
                fontSize={fontSize}
                fontWeight="bold"
              >
                {ticker || name}
              </text>
              <text
                x={x + width / 2}
                y={y + height / 2 + 8}
                textAnchor="middle"
                fill={theme.palette.text.primary}
                fontSize={fontSize - 2}
              >
                {change >= 0 ? '+' : ''}{change?.toFixed(2)}%
              </text>
            </>
          )}
        </g>
      </Tooltip>
    );
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="subtitle1">
          Market Heatmap
        </Typography>
        <Box display="flex" gap={2} alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <Box
              sx={{
                width: 12,
                height: 12,
                bgcolor: theme.palette.success.main,
                borderRadius: 0.5
              }}
            />
            <Typography variant="caption">Gains</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Box
              sx={{
                width: 12,
                height: 12,
                bgcolor: theme.palette.error.main,
                borderRadius: 0.5
              }}
            />
            <Typography variant="caption">Losses</Typography>
          </Box>
        </Box>
      </Box>
      
      <ResponsiveContainer width="100%" height={height}>
        <Treemap
          data={[processedData]}
          dataKey="value"
          aspectRatio={4 / 3}
          stroke={theme.palette.divider}
          content={<CustomTreemapContent />}
        />
      </ResponsiveContainer>

      <Box mt={2} display="flex" flexWrap="wrap" gap={1}>
        <Typography variant="caption" color="textSecondary">
          Size represents market cap • Color intensity shows % change
        </Typography>
      </Box>
    </Box>
  );
};

export default MarketHeatmap;