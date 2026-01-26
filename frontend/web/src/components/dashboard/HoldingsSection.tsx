import React, { memo, useState, useCallback, useMemo } from 'react';
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
  IconButton,
  Tooltip,
  Button,
  Skeleton,
  useTheme,
  useMediaQuery,
  alpha,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  MoreVert as MoreIcon,
  Add as AddIcon,
  Visibility as ViewIcon,
  NotificationsActive as AlertIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Position } from '../../types';
import Sparkline from '../charts/Sparkline';

interface HoldingsSectionProps {
  positions: Position[];
  isLoading?: boolean;
  maxRows?: number;
  onAddPosition?: () => void;
}

type SortDirection = 'asc' | 'desc';
type SortColumn =
  | 'ticker'
  | 'quantity'
  | 'averagePrice'
  | 'currentPrice'
  | 'marketValue'
  | 'totalGainPercent'
  | 'dayGainPercent';

interface Column {
  id: SortColumn;
  label: string;
  align: 'left' | 'right' | 'center';
  minWidth?: number;
  hideOnMobile?: boolean;
  format?: (value: any, position: Position) => React.ReactNode;
}

/**
 * HoldingsSection - Portfolio holdings table
 *
 * Features:
 * - Sortable columns
 * - Color-coded P&L (with icons for accessibility)
 * - Sparkline trends
 * - Quick action buttons
 * - Responsive column hiding
 * - WCAG 2.1 AA compliant
 */
const HoldingsSection: React.FC<HoldingsSectionProps> = ({
  positions = [],
  isLoading = false,
  maxRows = 10,
  onAddPosition,
}) => {
  const theme = useTheme();
  const navigate = useNavigate();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.between('sm', 'md'));

  const [sortBy, setSortBy] = useState<SortColumn>('marketValue');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [showAll, setShowAll] = useState(false);

  // Format currency
  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value);
  };

  // Format percent with color
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

  // Column definitions
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

  // Filter columns for mobile
  const visibleColumns = useMemo(
    () => columns.filter((col) => !isMobile || !col.hideOnMobile),
    [columns, isMobile]
  );

  // Handle sort
  const handleSort = useCallback(
    (column: SortColumn) => {
      const isAsc = sortBy === column && sortDirection === 'asc';
      setSortDirection(isAsc ? 'desc' : 'asc');
      setSortBy(column);
    },
    [sortBy, sortDirection]
  );

  // Sort positions
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

  // Handle row click
  const handleRowClick = useCallback(
    (ticker: string) => {
      navigate(`/stocks/${ticker}`);
    },
    [navigate]
  );

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Skeleton variant="text" width={120} height={28} />
          <Skeleton variant="rectangular" width={100} height={32} sx={{ borderRadius: 1 }} />
        </Box>
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
      </Box>
    );
  }

  // Empty state
  if (positions.length === 0) {
    return (
      <Box>
        <Typography variant="h6" component="h2" gutterBottom>
          Holdings
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            py: 6,
            color: 'text.secondary',
          }}
        >
          <Typography variant="body1" gutterBottom>
            No positions in your portfolio
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={onAddPosition}
            sx={{ mt: 2 }}
          >
            Add Position
          </Button>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography variant="h6" component="h2">
          Holdings
        </Typography>
        <Button
          variant="outlined"
          size="small"
          startIcon={<AddIcon />}
          onClick={onAddPosition}
        >
          Add Position
        </Button>
      </Box>

      {/* Table */}
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
                    onClick={() => handleSort(column.id)}
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
                onClick={() => handleRowClick(position.ticker)}
                sx={{
                  cursor: 'pointer',
                  '&:focus-visible': {
                    outline: `2px solid ${theme.palette.primary.main}`,
                    outlineOffset: -2,
                  },
                }}
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    handleRowClick(position.ticker);
                  }
                }}
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
                  <TableCell align="center">
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 0.5 }}>
                      <Tooltip title="View details">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleRowClick(position.ticker);
                          }}
                          aria-label={`View details for ${position.ticker}`}
                        >
                          <ViewIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Set alert">
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            // Handle set alert
                          }}
                          aria-label={`Set alert for ${position.ticker}`}
                        >
                          <AlertIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </TableCell>
                )}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      {/* Show more/less button */}
      {positions.length > maxRows && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Button
            size="small"
            onClick={() => setShowAll(!showAll)}
            aria-expanded={showAll}
            aria-controls="holdings-table"
          >
            {showAll
              ? 'Show Less'
              : `Show All (${positions.length - maxRows} more)`}
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default memo(HoldingsSection);
