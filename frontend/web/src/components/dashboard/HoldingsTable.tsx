import React, { memo, useMemo, useCallback } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Skeleton,
  useTheme,
  useMediaQuery,
  alpha,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
} from '@mui/icons-material';
import { Position } from '../../types';
import HoldingsActions from './HoldingsActions';

export type SortDirection = 'asc' | 'desc';
export type SortColumn =
  | 'ticker'
  | 'quantity'
  | 'averagePrice'
  | 'currentPrice'
  | 'marketValue'
  | 'totalGainPercent'
  | 'dayGainPercent';

export interface Column {
  id: SortColumn;
  label: string;
  align: 'left' | 'right' | 'center';
  minWidth?: number;
  hideOnMobile?: boolean;
  format?: (value: any, position: Position) => React.ReactNode;
}

interface HoldingsTableProps {
  positions: Position[];
  sortBy: SortColumn;
  sortDirection: SortDirection;
  showAll: boolean;
  maxRows: number;
  onSort: (column: SortColumn) => void;
  onRowClick: (ticker: string) => void;
  onSetAlert?: (ticker: string) => void;
  isLoading?: boolean;
}

/**
 * HoldingsTable - Sortable table displaying portfolio positions.
 *
 * Renders column headers with sort controls, position rows with formatted
 * values, inline action buttons on desktop, and a loading skeleton state.
 */
const HoldingsTable: React.FC<HoldingsTableProps> = ({
  positions,
  sortBy,
  sortDirection,
  showAll,
  maxRows,
  onSort,
  onRowClick,
  onSetAlert,
  isLoading = false,
}) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  const formatPercent = (value: number, showIcon = true): React.ReactNode => {
    const isPositive = value >= 0;
    const formattedValue = `${isPositive ? '+' : ''}${value.toFixed(2)}%`;

    return (
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'flex-end',
          gap: 0.5,
        }}
      >
        {showIcon && (
          isPositive ? (
            <TrendingUpIcon
              sx={{ fontSize: 16, color: theme.palette.success.main }}
              aria-hidden="true"
            />
          ) : (
            <TrendingDownIcon
              sx={{ fontSize: 16, color: theme.palette.error.main }}
              aria-hidden="true"
            />
          )
        )}
        <Typography
          component="span"
          sx={{
            color: isPositive
              ? theme.palette.success.main
              : theme.palette.error.main,
            fontWeight: 600,
            fontFamily: '"SF Mono", Monaco, monospace',
            fontSize: 'inherit',
          }}
        >
          {formattedValue}
        </Typography>
      </Box>
    );
  };

  const columns: Column[] = useMemo(
    () => [
      {
        id: 'ticker',
        label: 'Symbol',
        align: 'left',
        minWidth: 120,
        format: (_, position: Position) => (
          <Box>
            <Typography
              variant="body2"
              sx={{
                fontWeight: 700,
                fontFamily: '"SF Mono", Monaco, monospace',
              }}
            >
              {position.ticker}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{
                display: 'block',
                maxWidth: 120,
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {position.companyName}
            </Typography>
          </Box>
        ),
      },
      {
        id: 'quantity',
        label: 'Shares',
        align: 'right',
        minWidth: 80,
        hideOnMobile: true,
        format: (value) => (
          <Typography
            variant="body2"
            sx={{ fontFamily: '"SF Mono", Monaco, monospace' }}
          >
            {value.toLocaleString()}
          </Typography>
        ),
      },
      {
        id: 'averagePrice',
        label: 'Avg Cost',
        align: 'right',
        minWidth: 90,
        hideOnMobile: true,
        format: (value) => formatCurrency(value),
      },
      {
        id: 'currentPrice',
        label: 'Price',
        align: 'right',
        minWidth: 90,
        format: (value) => formatCurrency(value),
      },
      {
        id: 'marketValue',
        label: 'Value',
        align: 'right',
        minWidth: 100,
        format: (value) => formatCurrency(value),
      },
      {
        id: 'totalGainPercent',
        label: 'Total P&L',
        align: 'right',
        minWidth: 100,
        hideOnMobile: true,
        format: (value, position) => (
          <Box>
            {formatPercent(value)}
            <Typography
              variant="caption"
              display="block"
              color="text.secondary"
              sx={{
                fontFamily: '"SF Mono", Monaco, monospace',
                textAlign: 'right',
              }}
            >
              {formatCurrency(position.totalGain)}
            </Typography>
          </Box>
        ),
      },
      {
        id: 'dayGainPercent',
        label: 'Day P&L',
        align: 'right',
        minWidth: 100,
        format: (value, position) => (
          <Box>
            {formatPercent(value)}
            {!isMobile && (
              <Typography
                variant="caption"
                display="block"
                color="text.secondary"
                sx={{
                  fontFamily: '"SF Mono", Monaco, monospace',
                  textAlign: 'right',
                }}
              >
                {formatCurrency(position.dayGain)}
              </Typography>
            )}
          </Box>
        ),
      },
    ],
    [isMobile, theme.palette.success.main, theme.palette.error.main]
  );

  const visibleColumns = useMemo(
    () => columns.filter((col) => !isMobile || !col.hideOnMobile),
    [columns, isMobile]
  );

  const sortedPositions = useMemo(() => {
    const sorted = [...positions].sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc'
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }

      return sortDirection === 'asc'
        ? (aValue as number) - (bValue as number)
        : (bValue as number) - (aValue as number);
    });

    return showAll ? sorted : sorted.slice(0, maxRows);
  }, [positions, sortBy, sortDirection, showAll, maxRows]);

  const handleRowKeyDown = useCallback(
    (e: React.KeyboardEvent, ticker: string) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onRowClick(ticker);
      }
    },
    [onRowClick]
  );

  if (isLoading) {
    return (
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              {visibleColumns.map((col) => (
                <TableCell key={col.id}>
                  <Skeleton variant="text" width={60} />
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {[...Array(5)].map((_, index) => (
              <TableRow key={index}>
                {visibleColumns.map((col) => (
                  <TableCell key={col.id}>
                    <Skeleton variant="text" />
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  }

  return (
    <TableContainer
      sx={{
        overflowX: 'auto',
        '&::-webkit-scrollbar': {
          height: 8,
        },
        '&::-webkit-scrollbar-track': {
          backgroundColor: alpha(theme.palette.grey[500], 0.1),
          borderRadius: 4,
        },
        '&::-webkit-scrollbar-thumb': {
          backgroundColor: alpha(theme.palette.grey[500], 0.3),
          borderRadius: 4,
        },
      }}
    >
      <Table
        aria-label={`Holdings table with ${positions.length} positions, sortable by column headers`}
        size={isMobile ? 'small' : 'medium'}
      >
        <TableHead>
          <TableRow>
            {visibleColumns.map((column) => (
              <TableCell
                key={column.id}
                align={column.align}
                sx={{
                  minWidth: column.minWidth,
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  borderBottom: `2px solid ${theme.palette.divider}`,
                }}
              >
                <TableSortLabel
                  active={sortBy === column.id}
                  direction={sortBy === column.id ? sortDirection : 'asc'}
                  onClick={() => onSort(column.id)}
                >
                  {column.label}
                </TableSortLabel>
              </TableCell>
            ))}
            {!isMobile && (
              <TableCell
                align="center"
                sx={{
                  width: 80,
                  fontWeight: 600,
                  fontSize: '0.75rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em',
                  borderBottom: `2px solid ${theme.palette.divider}`,
                }}
              >
                Actions
              </TableCell>
            )}
          </TableRow>
        </TableHead>
        <TableBody>
          {sortedPositions.map((position) => (
            <TableRow
              key={position.id}
              hover
              onClick={() => onRowClick(position.ticker)}
              sx={{
                cursor: 'pointer',
                '&:focus-visible': {
                  outline: `2px solid ${theme.palette.primary.main}`,
                  outlineOffset: -2,
                },
              }}
              tabIndex={0}
              onKeyDown={(e) => handleRowKeyDown(e, position.ticker)}
              aria-label={`${position.ticker}, ${position.companyName}, value ${formatCurrency(position.marketValue)}, ${position.dayGainPercent >= 0 ? 'up' : 'down'} ${Math.abs(position.dayGainPercent).toFixed(2)} percent today`}
            >
              {visibleColumns.map((column) => (
                <TableCell key={column.id} align={column.align}>
                  {column.format
                    ? column.format(position[column.id], position)
                    : position[column.id]}
                </TableCell>
              ))}
              {!isMobile && (
                <HoldingsActions
                  ticker={position.ticker}
                  onViewDetails={onRowClick}
                  onSetAlert={onSetAlert}
                />
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export default memo(HoldingsTable);
