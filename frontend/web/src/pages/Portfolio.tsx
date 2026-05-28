import React, { useEffect, useState, useMemo, useCallback } from 'react';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Tabs,
  Tab,
  LinearProgress,
  Alert,
  Badge,
} from '@mui/material';
import {
  Add as AddIcon,
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Refresh,
  Cable as WebSocketIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchPortfolio,
  fetchTransactions,
  addTransaction,
  deletePosition,
  Position,
} from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';
import { usePortfolioWebSocket } from '../hooks/usePortfolioWebSocket';
import { env } from '../utils/env';
import {
  PositionsTabContent,
  PerformanceTabContent,
  TransactionsTabContent,
  AnalysisTabContent,
} from '../components/portfolio/PortfolioTabs';
import { AllocationTabContent, RiskAnalysisTabContent } from '../components/portfolio/PortfolioChart';
import {
  AddTransactionDialog,
  DeleteConfirmDialog,
  TransactionFormData,
} from '../components/portfolio/PortfolioActions';

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
  const user = useAppSelector((state) => state.app.user);
  const [tabValue, setTabValue] = useState(0);
  const [addTransactionOpen, setAddTransactionOpen] = useState(false);
  const [, setSelectedPosition] = useState<Position | null>(null);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [positionToDelete, setPositionToDelete] = useState<string | null>(null);
  const [transactionForm, setTransactionForm] = useState<TransactionFormData>({
    ticker: '',
    type: 'BUY',
    quantity: 0,
    price: 0,
    notes: '',
  });

  // Get symbols for WebSocket subscription
  const symbols = useMemo(() => positions.map((p) => p.ticker), [positions]);
  const portfolioId = useMemo(
    () => (user?.id ? `portfolio-${user.id}` : 'default-portfolio'),
    [user?.id]
  );

  // Set up WebSocket for real-time price updates
  const { isConnected, priceUpdates, latency } = usePortfolioWebSocket(
    portfolioId,
    symbols,
    true
  );

  // Update positions with real-time prices
  const updatedPositions = useMemo(() => {
    return positions.map((position) => {
      const priceUpdate = priceUpdates.get(position.ticker);
      if (priceUpdate) {
        return {
          ...position,
          currentPrice: priceUpdate.price,
          dayGain: priceUpdate.change || 0,
          dayGainPercent: priceUpdate.change_percent || 0,
          marketValue: position.quantity * priceUpdate.price,
        };
      }
      return position;
    });
  }, [positions, priceUpdates]);

  // Update metrics with real-time data
  const updatedMetrics = useMemo(() => {
    if (!metrics || updatedPositions.length === 0) return metrics;

    const totalValue = updatedPositions.reduce((sum, p) => sum + (p.marketValue || 0), 0);
    const totalGain = updatedPositions.reduce((sum, p) => sum + (p.totalGain || 0), 0);
    const dayGain = updatedPositions.reduce((sum, p) => sum + (p.dayGain || 0), 0);

    return {
      ...metrics,
      totalValue,
      totalGain,
      totalGainPercent: (totalGain / (totalValue - totalGain)) * 100,
      dayGain,
      dayGainPercent: (dayGain / totalValue) * 100,
    };
  }, [metrics, updatedPositions]);

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
    } catch {
      dispatch(
        addNotification({
          type: 'error',
          message: 'Failed to add transaction',
        })
      );
    }
  };

  const handleDeleteClick = useCallback((positionId: string) => {
    setPositionToDelete(positionId);
    setDeleteConfirmOpen(true);
  }, []);

  const handleDeleteConfirm = useCallback(async () => {
    if (!positionToDelete) return;
    setDeleteConfirmOpen(false);
    try {
      await dispatch(deletePosition(positionToDelete)).unwrap();
      dispatch(
        addNotification({
          type: 'success',
          message: 'Position deleted successfully',
        })
      );
    } catch {
      dispatch(
        addNotification({
          type: 'error',
          message: 'Failed to delete position',
        })
      );
    }
    setPositionToDelete(null);
  }, [positionToDelete, dispatch]);

  const handleDeleteCancel = useCallback(() => {
    setDeleteConfirmOpen(false);
    setPositionToDelete(null);
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Typography variant="h4" fontWeight="bold">
            Portfolio
          </Typography>
          {env.ENABLE_WEBSOCKETS && (
            <Badge
              badgeContent={isConnected ? 'LIVE' : 'OFFLINE'}
              color={isConnected ? 'success' : 'error'}
              sx={{
                '& .MuiBadge-badge': {
                  position: 'relative',
                  transform: 'none',
                  top: 0,
                  right: 0,
                },
              }}
            >
              <WebSocketIcon
                sx={{
                  color: isConnected ? 'success.main' : 'error.main',
                  animation: isConnected ? 'pulse 1s infinite' : 'none',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                    '100%': { opacity: 1 },
                  },
                }}
              />
            </Badge>
          )}
          {latency > 0 && (
            <Typography variant="caption" color="textSecondary">
              Latency: {latency}ms
            </Typography>
          )}
        </Box>
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
        <Alert severity="info" sx={{ mb: 2 }}>
          {/40[34]|not found|no data|network/i.test(error)
            ? 'No portfolio data is available yet — create a portfolio or add positions to get started.'
            : error}
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
                    {formatCurrency(updatedMetrics?.totalValue || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={updatedMetrics?.dayGainPercent && updatedMetrics.dayGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(updatedMetrics?.dayGainPercent || 0)} Today
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
                    {formatCurrency(updatedMetrics?.totalGain || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={updatedMetrics?.totalGainPercent && updatedMetrics.totalGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(updatedMetrics?.totalGainPercent || 0)}
                  </Typography>
                </Box>
                {updatedMetrics?.totalGain && updatedMetrics.totalGain >= 0 ? (
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
                    {formatCurrency(updatedMetrics?.dayGain || 0)}
                  </Typography>
                  <Typography
                    variant="body2"
                    color={updatedMetrics?.dayGainPercent && updatedMetrics.dayGainPercent >= 0 ? 'success.main' : 'error.main'}
                  >
                    {formatPercent(updatedMetrics?.dayGainPercent || 0)}
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
          <Tab label="Risk Analysis" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <PositionsTabContent
            positions={updatedPositions}
            formatCurrency={formatCurrency}
            formatPercent={formatPercent}
            onEdit={setSelectedPosition}
            onDelete={handleDeleteClick}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <PerformanceTabContent metrics={updatedMetrics} />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <AllocationTabContent metrics={updatedMetrics} />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <TransactionsTabContent
            transactions={transactions}
            formatCurrency={formatCurrency}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <AnalysisTabContent
            positions={updatedPositions}
            formatCurrency={formatCurrency}
            formatPercent={formatPercent}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <RiskAnalysisTabContent
            metrics={updatedMetrics}
            positions={updatedPositions}
            totalValue={updatedMetrics?.totalValue || 0}
            diversificationScore={(metrics as any)?.diversificationScore || 65}
          />
        </TabPanel>
      </Paper>

      <AddTransactionDialog
        open={addTransactionOpen}
        onClose={() => setAddTransactionOpen(false)}
        onSubmit={handleAddTransaction}
        transactionForm={transactionForm}
        onFormChange={setTransactionForm}
        formatCurrency={formatCurrency}
      />

      <DeleteConfirmDialog
        open={deleteConfirmOpen}
        onCancel={handleDeleteCancel}
        onConfirm={handleDeleteConfirm}
      />
    </Container>
  );
};

export default Portfolio;
