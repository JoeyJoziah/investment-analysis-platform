import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Chip,
} from '@mui/material';
import {
  Edit as EditIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import {
  ResponsiveContainer,
  Area,
  AreaChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip as RechartsTooltip,
} from 'recharts';
import type { Position, Transaction, PortfolioMetrics } from '../../store/slices/portfolioSlice';

export interface PositionsTabProps {
  positions: Position[];
  formatCurrency: (value: number) => string;
  formatPercent: (value: number) => string;
  onEdit: (position: Position) => void;
  onDelete: (positionId: string) => void;
}

export const PositionsTabContent: React.FC<PositionsTabProps> = ({
  positions,
  formatCurrency,
  formatPercent,
  onEdit,
  onDelete,
}) => {
  return (
    <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Symbol</TableCell>
            <TableCell>Company</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Avg Cost</TableCell>
            <TableCell align="right">Current Price</TableCell>
            <TableCell align="right">Market Value</TableCell>
            <TableCell align="right">Total Gain</TableCell>
            <TableCell align="right">Day Gain</TableCell>
            <TableCell align="center">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {positions.map((position) => (
            <TableRow key={position.id}>
              <TableCell>
                <Typography variant="subtitle2" fontWeight="bold">
                  {position.ticker}
                </Typography>
              </TableCell>
              <TableCell>{position.companyName}</TableCell>
              <TableCell align="right">{position.quantity}</TableCell>
              <TableCell align="right">{formatCurrency(position.averagePrice)}</TableCell>
              <TableCell align="right">{formatCurrency(position.currentPrice)}</TableCell>
              <TableCell align="right">{formatCurrency(position.marketValue)}</TableCell>
              <TableCell align="right">
                <Box sx={{ color: position.totalGain >= 0 ? 'success.main' : 'error.main' }}>
                  {formatCurrency(position.totalGain)}
                  <br />
                  <Typography variant="caption">
                    {formatPercent(position.totalGainPercent)}
                  </Typography>
                </Box>
              </TableCell>
              <TableCell align="right">
                <Box sx={{ color: position.dayGain >= 0 ? 'success.main' : 'error.main' }}>
                  {formatCurrency(position.dayGain)}
                  <br />
                  <Typography variant="caption">
                    {formatPercent(position.dayGainPercent)}
                  </Typography>
                </Box>
              </TableCell>
              <TableCell align="center">
                <IconButton
                  size="small"
                  onClick={() => onEdit(position)}
                  aria-label={`Edit ${position.ticker} position`}
                >
                  <EditIcon fontSize="small" />
                </IconButton>
                <IconButton
                  size="small"
                  onClick={() => onDelete(position.id)}
                  color="error"
                  aria-label={`Delete ${position.ticker} position`}
                >
                  <DeleteIcon fontSize="small" />
                </IconButton>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export interface PerformanceTabProps {
  metrics: PortfolioMetrics | null;
}

export const PerformanceTabContent: React.FC<PerformanceTabProps> = ({ metrics }) => {
  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h6" gutterBottom>
          Portfolio Performance
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={metrics?.performance?.daily || []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" />
            <YAxis />
            <RechartsTooltip />
            <Area type="monotone" dataKey="value" stroke="#8884d8" fill="#8884d8" />
          </AreaChart>
        </ResponsiveContainer>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Risk Metrics
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Sharpe Ratio</Typography>
              <Typography fontWeight="bold">
                {metrics?.riskMetrics?.sharpeRatio?.toFixed(2) || '-'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Beta</Typography>
              <Typography fontWeight="bold">
                {metrics?.riskMetrics?.beta?.toFixed(2) || '-'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Alpha</Typography>
              <Typography fontWeight="bold">
                {metrics?.riskMetrics?.alpha?.toFixed(2) || '-'}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Standard Deviation</Typography>
              <Typography fontWeight="bold">
                {metrics?.riskMetrics?.standardDeviation?.toFixed(2) || '-'}%
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography>Max Drawdown</Typography>
              <Typography fontWeight="bold" color="error.main">
                {metrics?.riskMetrics?.maxDrawdown?.toFixed(2) || '-'}%
              </Typography>
            </Box>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
};

export interface TransactionsTabProps {
  transactions: Transaction[];
  formatCurrency: (value: number) => string;
}

export const TransactionsTabContent: React.FC<TransactionsTabProps> = ({
  transactions,
  formatCurrency,
}) => {
  return (
    <TableContainer>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Date</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Symbol</TableCell>
            <TableCell align="right">Quantity</TableCell>
            <TableCell align="right">Price</TableCell>
            <TableCell align="right">Total Amount</TableCell>
            <TableCell>Notes</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {transactions.map((transaction) => (
            <TableRow key={transaction.id}>
              <TableCell>
                {new Date(transaction.date).toLocaleDateString()}
              </TableCell>
              <TableCell>
                <Chip
                  label={transaction.type}
                  color={transaction.type === 'BUY' ? 'success' : 'error'}
                  size="small"
                />
              </TableCell>
              <TableCell>{transaction.ticker}</TableCell>
              <TableCell align="right">{transaction.quantity}</TableCell>
              <TableCell align="right">{formatCurrency(transaction.price)}</TableCell>
              <TableCell align="right">
                {formatCurrency(transaction.totalAmount)}
              </TableCell>
              <TableCell>{transaction.notes || '-'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

export interface AnalysisTabProps {
  positions: Position[];
  formatCurrency: (value: number) => string;
  formatPercent: (value: number) => string;
}

export const AnalysisTabContent: React.FC<AnalysisTabProps> = ({
  positions,
  formatCurrency,
  formatPercent,
}) => {
  return (
    <>
      <Typography variant="h6" gutterBottom>
        Portfolio Analysis
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Top Performers
            </Typography>
            {[...positions]
              .sort((a, b) => b.totalGainPercent - a.totalGainPercent)
              .slice(0, 5)
              .map((position) => (
                <Box
                  key={position.id}
                  sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}
                >
                  <Typography>{position.ticker}</Typography>
                  <Typography color="success.main">
                    {formatPercent(position.totalGainPercent)}
                  </Typography>
                </Box>
              ))}
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Worst Performers
            </Typography>
            {[...positions]
              .sort((a, b) => a.totalGainPercent - b.totalGainPercent)
              .slice(0, 5)
              .map((position) => (
                <Box
                  key={position.id}
                  sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}
                >
                  <Typography>{position.ticker}</Typography>
                  <Typography color="error.main">
                    {formatPercent(position.totalGainPercent)}
                  </Typography>
                </Box>
              ))}
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              Largest Positions
            </Typography>
            {[...positions]
              .sort((a, b) => b.marketValue - a.marketValue)
              .slice(0, 5)
              .map((position) => (
                <Box
                  key={position.id}
                  sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}
                >
                  <Typography>{position.ticker}</Typography>
                  <Typography>{formatCurrency(position.marketValue)}</Typography>
                </Box>
              ))}
          </Paper>
        </Grid>
      </Grid>
    </>
  );
};

export default {
  PositionsTabContent,
  PerformanceTabContent,
  TransactionsTabContent,
  AnalysisTabContent,
};
