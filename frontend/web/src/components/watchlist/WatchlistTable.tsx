import React from 'react';
import {
  Box,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Skeleton,
  Tooltip,
  Typography,
  Paper,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  MoreVert,
  NotificationsActive,
  NotificationsOff,
} from '@mui/icons-material';
import { WatchlistItem } from '../../store/slices/portfolioSlice';

// --- Formatters ---

export const formatCurrency = (value: number | null): string => {
  if (value === null || value === undefined) return '-';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
  }).format(value);
};

export const formatPercent = (value: number | null): string => {
  if (value === null || value === undefined) return '-';
  return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
};

export const formatLargeNumber = (value: number | null): string => {
  if (value === null || value === undefined) return '-';
  if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return value.toFixed(0);
};

// --- Table Skeleton ---

const TableSkeleton: React.FC = () => (
  <>
    {[1, 2, 3, 4, 5].map((i) => (
      <TableRow key={i}>
        <TableCell><Skeleton width={60} /></TableCell>
        <TableCell><Skeleton width={150} /></TableCell>
        <TableCell align="right"><Skeleton width={80} /></TableCell>
        <TableCell align="right"><Skeleton width={100} /></TableCell>
        <TableCell align="right"><Skeleton width={80} /></TableCell>
        <TableCell align="center"><Skeleton width={30} /></TableCell>
        <TableCell align="center"><Skeleton width={40} /></TableCell>
      </TableRow>
    ))}
  </>
);

// --- Watchlist Table ---

export interface WatchlistTableProps {
  items: WatchlistItem[];
  isLoading: boolean;
  onNavigateToAnalysis: (symbol: string) => void;
  onToggleAlert: (item: WatchlistItem) => void;
  onMenuOpen: (event: React.MouseEvent<HTMLElement>, item: WatchlistItem) => void;
}

const WatchlistTable: React.FC<WatchlistTableProps> = ({
  items,
  isLoading,
  onNavigateToAnalysis,
  onToggleAlert,
  onMenuOpen,
}) => (
  <TableContainer component={Paper}>
    <Table>
      <TableHead>
        <TableRow>
          <TableCell>Symbol</TableCell>
          <TableCell>Company</TableCell>
          <TableCell align="right">Price</TableCell>
          <TableCell align="right">Change</TableCell>
          <TableCell align="right">Target Price</TableCell>
          <TableCell align="center">Alerts</TableCell>
          <TableCell align="center">Actions</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {isLoading && items.length === 0 ? (
          <TableSkeleton />
        ) : (
          items.map((item) => (
            <TableRow key={item.id} hover>
              <TableCell>
                <Button
                  variant="text"
                  onClick={() => onNavigateToAnalysis(item.symbol)}
                  sx={{ fontWeight: 'bold' }}
                >
                  {item.symbol}
                </Button>
              </TableCell>
              <TableCell>{item.company_name}</TableCell>
              <TableCell align="right">{formatCurrency(item.current_price)}</TableCell>
              <TableCell align="right">
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'flex-end',
                    gap: 0.5,
                  }}
                >
                  {item.price_change !== null && (
                    <>
                      {item.price_change >= 0 ? (
                        <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                      ) : (
                        <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                      )}
                      <Typography
                        color={item.price_change >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatCurrency(Math.abs(item.price_change))} (
                        {formatPercent(item.price_change_percent)})
                      </Typography>
                    </>
                  )}
                  {item.price_change === null && (
                    <Typography color="text.secondary">-</Typography>
                  )}
                </Box>
              </TableCell>
              <TableCell align="right">
                {item.target_price ? (
                  <Tooltip
                    title={`${
                      item.current_price && item.target_price > item.current_price
                        ? 'Above'
                        : 'Below'
                    } current price`}
                  >
                    <Typography
                      color={
                        item.current_price && item.target_price > item.current_price
                          ? 'success.main'
                          : 'error.main'
                      }
                    >
                      {formatCurrency(item.target_price)}
                    </Typography>
                  </Tooltip>
                ) : (
                  <Typography color="text.secondary">-</Typography>
                )}
              </TableCell>
              <TableCell align="center">
                <Tooltip title={item.alert_enabled ? 'Alerts enabled' : 'Alerts disabled'}>
                  <IconButton
                    size="small"
                    color={item.alert_enabled ? 'primary' : 'default'}
                    onClick={() => onToggleAlert(item)}
                  >
                    {item.alert_enabled ? <NotificationsActive /> : <NotificationsOff />}
                  </IconButton>
                </Tooltip>
              </TableCell>
              <TableCell align="center">
                <IconButton size="small" onClick={(e) => onMenuOpen(e, item)}>
                  <MoreVert />
                </IconButton>
              </TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  </TableContainer>
);

export default WatchlistTable;
