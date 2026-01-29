import React, { useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Tooltip,
} from '@mui/material';

interface CorrelationMatrixProps {
  correlations: Record<string, Record<string, number>>;
  title?: string;
}

const getHeatmapColor = (value: number): string => {
  // Value range: -1 to 1
  // Red for negative correlation, green for positive
  const normalized = (value + 1) / 2; // Convert to 0-1 range

  if (normalized < 0.5) {
    // Red shades for negative
    const intensity = 1 - normalized * 2;
    return `rgba(255, ${Math.round(100 + (1 - intensity) * 155)}, 100, 0.7)`;
  } else {
    // Green shades for positive
    const intensity = (normalized - 0.5) * 2;
    return `rgba(${Math.round(255 - intensity * 155)}, ${Math.round(100 + intensity * 155)}, 100, 0.7)`;
  }
};

export const CorrelationMatrix: React.FC<CorrelationMatrixProps> = ({
  correlations,
  title = 'Asset Correlation Matrix',
}) => {
  const symbols = useMemo(() => Object.keys(correlations), [correlations]);

  const getCorrelationValue = (symbol1: string, symbol2: string): number => {
    if (symbol1 === symbol2) return 1.0;
    return correlations[symbol1]?.[symbol2] ?? 0;
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom fontWeight="bold">
        {title}
      </Typography>

      <Box sx={{ overflowX: 'auto', mt: 3 }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: `60px repeat(${symbols.length}, 80px)`,
            gap: 0.5,
            minWidth: 'fit-content',
          }}
        >
          {/* Column headers */}
          <Box />
          {symbols.map((symbol) => (
            <Box
              key={`header-${symbol}`}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 'bold',
                fontSize: '0.85rem',
                writingMode: 'vertical-rl',
                textOrientation: 'mixed',
                transform: 'rotate(180deg)',
              }}
            >
              {symbol}
            </Box>
          ))}

          {/* Row headers and cells */}
          {symbols.map((rowSymbol) => (
            <React.Fragment key={`row-${rowSymbol}`}>
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 'bold',
                  fontSize: '0.85rem',
                }}
              >
                {rowSymbol}
              </Box>

              {symbols.map((colSymbol) => {
                const correlation = getCorrelationValue(rowSymbol, colSymbol);
                const color = getHeatmapColor(correlation);

                return (
                  <Tooltip
                    key={`cell-${rowSymbol}-${colSymbol}`}
                    title={`${rowSymbol} - ${colSymbol}: ${correlation.toFixed(3)}`}
                  >
                    <Box
                      sx={{
                        background: color,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        minHeight: '40px',
                        minWidth: '40px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '0.75rem',
                        fontWeight: rowSymbol === colSymbol ? 'bold' : 'normal',
                        border: '1px solid #ddd',
                        '&:hover': {
                          boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
                          transform: 'scale(1.05)',
                        },
                        transition: 'all 0.2s ease',
                      }}
                    >
                      {correlation.toFixed(2)}
                    </Box>
                  </Tooltip>
                );
              })}
            </React.Fragment>
          ))}
        </Box>
      </Box>

      {/* Legend */}
      <Box sx={{ mt: 4, display: 'flex', alignItems: 'center', gap: 2 }}>
        <Typography variant="caption" fontWeight="bold">
          Legend:
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              backgroundColor: getHeatmapColor(-1),
              borderRadius: '2px',
            }}
          />
          <Typography variant="caption">-1 (Negative)</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              backgroundColor: getHeatmapColor(0),
              borderRadius: '2px',
            }}
          />
          <Typography variant="caption">0 (Neutral)</Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 20,
              height: 20,
              backgroundColor: getHeatmapColor(1),
              borderRadius: '2px',
            }}
          />
          <Typography variant="caption">+1 (Positive)</Typography>
        </Box>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
        Higher positive correlations indicate assets that move together. Negative correlations
        indicate diversification benefits. Aim for a balanced portfolio with low average
        correlation.
      </Typography>
    </Paper>
  );
};

export default CorrelationMatrix;
