/**
 * Lazy-loaded chart components
 *
 * Charts are heavy components that include Recharts library dependencies.
 * By lazy-loading them, we reduce the initial bundle size and only load
 * the charting library when actually needed.
 */
import React, { Suspense, lazy } from 'react';
import { Box, Skeleton } from '@mui/material';

// Lazy load chart components
const LazyStockChart = lazy(() => import('./StockChart'));
const LazyMarketHeatmap = lazy(() => import('./MarketHeatmap'));
const LazySparkline = lazy(() => import('./Sparkline'));

// Chart skeleton fallback
interface ChartSkeletonProps {
  height?: number;
}

const ChartSkeleton: React.FC<ChartSkeletonProps> = ({ height = 400 }) => (
  <Box sx={{ width: '100%', height }}>
    <Skeleton variant="rectangular" height="100%" sx={{ borderRadius: 1 }} />
  </Box>
);

// Re-export with Suspense wrapper for easy consumption
interface StockChartProps {
  data?: Array<{
    date: string;
    open?: number;
    high?: number;
    low?: number;
    close: number;
    volume?: number;
    ma20?: number;
    ma50?: number;
    ma200?: number;
    rsi?: number;
    macd?: number;
    signal?: number;
  }>;
  height?: number;
  showVolume?: boolean;
  showIndicators?: boolean;
  chartType?: 'line' | 'area' | 'candlestick';
  ticker?: string;
}

export const StockChart: React.FC<StockChartProps> = (props) => (
  <Suspense fallback={<ChartSkeleton height={props.height} />}>
    <LazyStockChart {...props} />
  </Suspense>
);

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

export const MarketHeatmap: React.FC<MarketHeatmapProps> = (props) => (
  <Suspense fallback={<ChartSkeleton height={props.height} />}>
    <LazyMarketHeatmap {...props} />
  </Suspense>
);

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
}

export const Sparkline: React.FC<SparklineProps> = (props) => (
  <Suspense fallback={<ChartSkeleton height={props.height || 50} />}>
    <LazySparkline {...props} />
  </Suspense>
);

// Default export for direct lazy loading
export default {
  StockChart,
  MarketHeatmap,
  Sparkline,
};
