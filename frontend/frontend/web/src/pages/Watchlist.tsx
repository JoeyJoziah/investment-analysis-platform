import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  InputAdornment,
  Chip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
} from '@mui/material';
import {
  Add,
  Delete,
  MoreVert,
  Search,
  TrendingUp,
  TrendingDown,
  Refresh,
  Analytics,
  AddAlert,
  RemoveRedEye,
  SortByAlpha,
  FilterList,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchWatchlist,
  removeFromWatchlist,
  addToWatchlist,
} from '../store/slices/portfolioSlice';
import { fetchStockData } from '../store/slices/stockSlice';
import { addNotification } from '../store/slices/appSlice';

interface WatchlistStock {
  ticker: string;
  companyName: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  marketCap: number;
  dayHigh: number;
  dayLow: number;
  alerts: number;
}

const Watchlist: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { watchlist } = useAppSelector((state) => state.portfolio);
  
  const [watchlistData, setWatchlistData] = useState<WatchlistStock[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'ticker' | 'change' | 'volume'>('ticker');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [newTicker, setNewTicker] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    dispatch(fetchWatchlist());
    loadWatchlistData();
  }, [dispatch]);

  const loadWatchlistData = async () => {
    setIsLoading(true);
    // Simulate loading watchlist data
    const mockData: WatchlistStock[] = watchlist.map((ticker) => ({
      ticker,
      companyName: `${ticker} Company`,
      price: Math.random() * 1000,
      change: (Math.random() - 0.5) * 20,
      changePercent: (Math.random() - 0.5) * 10,
      volume: Math.floor(Math.random() * 10000000),
      marketCap: Math.floor(Math.random() * 1000000000000),
      dayHigh: Math.random() * 1000,
      dayLow: Math.random() * 900,
      alerts: Math.floor(Math.random() * 3),
    }));
    setWatchlistData(mockData);
    setIsLoading(false);
  };

  const handleRefresh = () => {
    loadWatchlistData();
    dispatch(
      addNotification({
        type: 'info',
        message: 'Watchlist refreshed',
      })
    );
  };

  const handleAddStock = async () => {
    if (newTicker && !watchlist.includes(newTicker)) {
      await dispatch(addToWatchlist(newTicker));
      setNewTicker('');
      setAddDialogOpen(false);
      dispatch(
        addNotification({
          type: 'success',
          message: `${newTicker} added to watchlist`,
        })
      );
      loadWatchlistData();
    }
  };

  const handleRemoveStock = async (ticker: string) => {
    await dispatch(removeFromWatchlist(ticker));
    dispatch(
      addNotification({
        type: 'info',
        message: `${ticker} removed from watchlist`,
      })
    );
    loadWatchlistData();
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, ticker: string) => {
    setAnchorEl(event.currentTarget);
    setSelectedStock(ticker);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedStock(null);
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

  const formatLargeNumber = (value: number) => {
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
  };

  const filteredData = watchlistData
    .filter((stock) =>
      stock.ticker.toLowerCase().includes(searchQuery.toLowerCase()) ||
      stock.companyName.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'ticker':
          return a.ticker.localeCompare(b.ticker);
        case 'change':
          return b.changePercent - a.changePercent;
        case 'volume':
          return b.volume - a.volume;
        default:
          return 0;
      }
    });

  const totalValue = watchlistData.reduce((sum, stock) => sum + stock.price, 0);
  const gainers = watchlistData.filter((stock) => stock.change > 0).length;
  const losers = watchlistData.filter((stock) => stock.change < 0).length;

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h4" fontWeight="bold">
            Watchlist
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setAddDialogOpen(true)}
            >
              Add Stock
            </Button>
          </Box>
        </Box>

        {/* Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Total Stocks
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {watchlistData.length}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Total Value
                </Typography>
                <Typography variant="h5" fontWeight="bold">
                  {formatCurrency(totalValue)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Gainers
                </Typography>
                <Typography variant="h4" fontWeight="bold" color="success.main">
                  {gainers}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Losers
                </Typography>
                <Typography variant="h4" fontWeight="bold" color="error.main">
                  {losers}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Search and Filter */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <TextField
              fullWidth
              placeholder="Search stocks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
            <Button
              startIcon={<SortByAlpha />}
              onClick={() => {
                const options: Array<'ticker' | 'change' | 'volume'> = ['ticker', 'change', 'volume'];
                const currentIndex = options.indexOf(sortBy);
                setSortBy(options[(currentIndex + 1) % options.length]);
              }}
            >
              Sort: {sortBy}
            </Button>
          </Box>
        </Paper>

        {/* Watchlist Table */}
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Company</TableCell>
                <TableCell align="right">Price</TableCell>
                <TableCell align="right">Change</TableCell>
                <TableCell align="right">Volume</TableCell>
                <TableCell align="right">Market Cap</TableCell>
                <TableCell align="right">Day Range</TableCell>
                <TableCell align="center">Alerts</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {filteredData.map((stock) => (
                <TableRow key={stock.ticker} hover>
                  <TableCell>
                    <Button
                      variant="text"
                      onClick={() => navigate(`/analysis/${stock.ticker}`)}
                      sx={{ fontWeight: 'bold' }}
                    >
                      {stock.ticker}
                    </Button>
                  </TableCell>
                  <TableCell>{stock.companyName}</TableCell>
                  <TableCell align="right">{formatCurrency(stock.price)}</TableCell>
                  <TableCell align="right">
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 0.5 }}>
                      {stock.change >= 0 ? (
                        <TrendingUp sx={{ fontSize: 16, color: 'success.main' }} />
                      ) : (
                        <TrendingDown sx={{ fontSize: 16, color: 'error.main' }} />
                      )}
                      <Typography color={stock.change >= 0 ? 'success.main' : 'error.main'}>
                        {formatCurrency(Math.abs(stock.change))} ({formatPercent(stock.changePercent)})
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell align="right">{formatLargeNumber(stock.volume)}</TableCell>
                  <TableCell align="right">{formatLargeNumber(stock.marketCap)}</TableCell>
                  <TableCell align="right">
                    {formatCurrency(stock.dayLow)} - {formatCurrency(stock.dayHigh)}
                  </TableCell>
                  <TableCell align="center">
                    {stock.alerts > 0 ? (
                      <Chip label={stock.alerts} size="small" color="primary" />
                    ) : (
                      '-'
                    )}
                  </TableCell>
                  <TableCell align="center">
                    <IconButton
                      size="small"
                      onClick={(e) => handleMenuOpen(e, stock.ticker)}
                    >
                      <MoreVert />
                    </IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {filteredData.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" color="text.secondary">
              {searchQuery ? 'No stocks found matching your search' : 'Your watchlist is empty'}
            </Typography>
            {!searchQuery && (
              <Button
                variant="contained"
                sx={{ mt: 2 }}
                onClick={() => setAddDialogOpen(true)}
              >
                Add Your First Stock
              </Button>
            )}
          </Box>
        )}

        {/* Context Menu */}
        <Menu
          anchorEl={anchorEl}
          open={Boolean(anchorEl)}
          onClose={handleMenuClose}
        >
          <MenuItem onClick={() => {
            if (selectedStock) {
              navigate(`/analysis/${selectedStock}`);
            }
            handleMenuClose();
          }}>
            <RemoveRedEye sx={{ mr: 1 }} /> View Analysis
          </MenuItem>
          <MenuItem onClick={() => {
            // Add alert logic
            handleMenuClose();
          }}>
            <AddAlert sx={{ mr: 1 }} /> Set Alert
          </MenuItem>
          <MenuItem onClick={() => {
            if (selectedStock) {
              handleRemoveStock(selectedStock);
            }
            handleMenuClose();
          }}>
            <Delete sx={{ mr: 1 }} /> Remove from Watchlist
          </MenuItem>
        </Menu>

        {/* Add Stock Dialog */}
        <Dialog open={addDialogOpen} onClose={() => setAddDialogOpen(false)}>
          <DialogTitle>Add Stock to Watchlist</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Ticker Symbol"
              fullWidth
              variant="outlined"
              value={newTicker}
              onChange={(e) => setNewTicker(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleAddStock()}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleAddStock} variant="contained">
              Add
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default Watchlist;