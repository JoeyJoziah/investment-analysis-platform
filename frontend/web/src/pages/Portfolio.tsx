import React, { useEffect, useState } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Chip,
  Tabs,
  Tab,
  LinearProgress,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Alert,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  PieChart,
  Timeline,
  Assessment,
  Download,
  Refresh,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchPortfolio,
  fetchTransactions,
  addTransaction,
  deletePosition,
} from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';
import {
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Area,
  AreaChart,
} from 'recharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Portfolio: React.FC = () => {
  const dispatch = useAppDispatch();
  const { positions, transactions, metrics, isLoading, error } = useAppSelector(
    (state) => state.portfolio
  );
  const [tabValue, setTabValue] = useState(0);
  const [addTransactionOpen, setAddTransactionOpen] = useState(false);
  const [selectedPosition, setSelectedPosition] = useState<any>(null);
  const [transactionForm, setTransactionForm] = useState({
    ticker: '',
    type: 'BUY' as 'BUY' | 'SELL',
    quantity: 0,
    price: 0,
    notes: '',
  });

  useEffect(() => {
    dispatch(fetchPortfolio());
    dispatch(fetchTransactions({}));
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchPortfolio());
    dispatch(fetchTransactions({}));
    dispatch(
      addNotification({
        type: 'info',
        message: 'Portfolio data refreshed',
      })
    );
  };

  const handleAddTransaction = async () => {
    try {
      await dispatch(
        addTransaction({
          ...transactionForm,
          totalAmount: transactionForm.quantity * transactionForm.price,
          date: new Date().toISOString(),
        })
      ).unwrap();
      
      setAddTransactionOpen(false);
      setTransactionForm({
        ticker: '',
        type: 'BUY',
        quantity: 0,
        price: 0,
        notes: '',
      });
      
      dispatch(
        addNotification({
          type: 'success',
          message: `Transaction added successfully`,
        })
      );
      
      dispatch(fetchPortfolio());
    } catch (error) {
      dispatch(
        addNotification({
          type: 'error',
          message: 'Failed to add transaction',
        })
      );
    }
  };

  const handleDeletePosition = async (positionId: string) => {
    if (window.confirm('Are you sure you want to delete this position?')) {
      try {
        await dispatch(deletePosition(positionId)).unwrap();
        dispatch(
          addNotification({
            type: 'success',
            message: 'Position deleted successfully',
          })
        );
      } catch (error) {
        dispatch(
          addNotification({
            type: 'error',
            message: 'Failed to delete position',
          })
        );
      }
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" fontWeight="bold">
          Portfolio
        </Typography>
        <Box>
          <Button
            variant="outlined"
            startIcon={<Refresh />}
            onClick={handleRefresh}
            sx={{ mr: 2 }}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setAddTransactionOpen(true)}
          >
            Add Transaction
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {/* Portfolio Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="caption">
                    Total Value
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatCurrency(metrics?.totalValue || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={metrics?.dayGainPercent && metrics.dayGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(metrics?.dayGainPercent || 0)} Today
                  </Typography>
                </Box>
                <AccountBalance sx={{ fontSize: 40, color: 'primary.main', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="caption">
                    Total Gain/Loss
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatCurrency(metrics?.totalGain || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={metrics?.totalGainPercent && metrics.totalGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(metrics?.totalGainPercent || 0)}
                  </Typography>
                </Box>
                {metrics?.totalGain && metrics.totalGain >= 0 ? (
                  <TrendingUp sx={{ fontSize: 40, color: 'success.main', opacity: 0.3 }} />
                ) : (
                  <TrendingDown sx={{ fontSize: 40, color: 'error.main', opacity: 0.3 }} />
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="caption">
                    Day Gain/Loss
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatCurrency(metrics?.dayGain || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={metrics?.dayGainPercent && metrics.dayGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(metrics?.dayGainPercent || 0)}
                  </Typography>
                </Box>
                <ShowChart sx={{ fontSize: 40, color: 'info.main', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <Box>
                  <Typography color="text.secondary" gutterBottom variant="caption">
                    Cash Balance
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {formatCurrency(metrics?.cashBalance || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Buying Power: {formatCurrency(metrics?.buyingPower || 0)}
                  </Typography>
                </Box>
                <AccountBalance sx={{ fontSize: 40, color: 'warning.main', opacity: 0.3 }} />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="Positions" />
          <Tab label="Performance" />
          <Tab label="Allocation" />
          <Tab label="Transactions" />
          <Tab label="Analysis" />
        </Tabs>

        {/* Positions Tab */}
        <TabPanel value={tabValue} index={0}>
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
                      <IconButton size="small" onClick={() => setSelectedPosition(position)}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => handleDeletePosition(position.id)}
                        color="error"
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Performance Tab */}
        <TabPanel value={tabValue} index={1}>
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
        </TabPanel>

        {/* Allocation Tab */}
        <TabPanel value={tabValue} index={2}>
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
        </TabPanel>

        {/* Transactions Tab */}
        <TabPanel value={tabValue} index={3}>
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
        </TabPanel>

        {/* Analysis Tab */}
        <TabPanel value={tabValue} index={4}>
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
        </TabPanel>
      </Paper>

      {/* Add Transaction Dialog */}
      <Dialog open={addTransactionOpen} onClose={() => setAddTransactionOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Transaction</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 2 }}>
            <TextField
              label="Ticker Symbol"
              value={transactionForm.ticker}
              onChange={(e) =>
                setTransactionForm({ ...transactionForm, ticker: e.target.value.toUpperCase() })
              }
              fullWidth
            />
            <TextField
              select
              label="Type"
              value={transactionForm.type}
              onChange={(e) =>
                setTransactionForm({ ...transactionForm, type: e.target.value as 'BUY' | 'SELL' })
              }
              fullWidth
            >
              <MenuItem value="BUY">Buy</MenuItem>
              <MenuItem value="SELL">Sell</MenuItem>
            </TextField>
            <TextField
              label="Quantity"
              type="number"
              value={transactionForm.quantity}
              onChange={(e) =>
                setTransactionForm({ ...transactionForm, quantity: Number(e.target.value) })
              }
              fullWidth
            />
            <TextField
              label="Price per Share"
              type="number"
              value={transactionForm.price}
              onChange={(e) =>
                setTransactionForm({ ...transactionForm, price: Number(e.target.value) })
              }
              fullWidth
              InputProps={{
                startAdornment: '$',
              }}
            />
            <TextField
              label="Notes (Optional)"
              value={transactionForm.notes}
              onChange={(e) =>
                setTransactionForm({ ...transactionForm, notes: e.target.value })
              }
              fullWidth
              multiline
              rows={2}
            />
            <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Total Amount: {formatCurrency(transactionForm.quantity * transactionForm.price)}
              </Typography>
            </Box>
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAddTransactionOpen(false)}>Cancel</Button>
          <Button
            onClick={handleAddTransaction}
            variant="contained"
            disabled={!transactionForm.ticker || transactionForm.quantity === 0 || transactionForm.price === 0}
          >
            Add Transaction
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default Portfolio;