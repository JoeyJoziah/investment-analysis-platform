import React, { useEffect, useState, useCallback } from 'react';
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
  CircularProgress,
  Skeleton,
  Switch,
  FormControlLabel,
  Tooltip,
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
  Edit,
  NotificationsActive,
  NotificationsOff,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchWatchlist,
  removeFromWatchlist,
  addToWatchlist,
  updateWatchlistItem,
  WatchlistItem,
} from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';

const Watchlist: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { watchlist, watchlistLoading, watchlistError } = useAppSelector(
    (state) => state.portfolio
  );

  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'symbol' | 'change' | 'price'>('symbol');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedItem, setSelectedItem] = useState<WatchlistItem | null>(null);
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [newTicker, setNewTicker] = useState('');
  const [newTargetPrice, setNewTargetPrice] = useState<string>('');
  const [newNotes, setNewNotes] = useState('');
  const [editTargetPrice, setEditTargetPrice] = useState<string>('');
  const [editNotes, setEditNotes] = useState('');
  const [editAlertEnabled, setEditAlertEnabled] = useState(false);

  useEffect(() => {
    dispatch(fetchWatchlist());
  }, [dispatch]);

  const handleRefresh = useCallback(() => {
    dispatch(fetchWatchlist());
    dispatch(
      addNotification({
        type: 'info',
        message: 'Watchlist refreshed',
      })
    );
  }, [dispatch]);

  const handleAddStock = async () => {
    if (!newTicker.trim()) return;

    // Check if stock already exists in watchlist
    const exists = watchlist?.items.some(
      (item) => item.symbol.toUpperCase() === newTicker.toUpperCase()
    );
    if (exists) {
      dispatch(
        addNotification({
          type: 'warning',
          message: `${newTicker} is already in your watchlist`,
        })
      );
      return;
    }

    try {
      await dispatch(
        addToWatchlist({
          symbol: newTicker.toUpperCase(),
          targetPrice: newTargetPrice ? parseFloat(newTargetPrice) : undefined,
          notes: newNotes || undefined,
        })
      ).unwrap();

      setNewTicker('');
      setNewTargetPrice('');
      setNewNotes('');
      setAddDialogOpen(false);
      dispatch(
        addNotification({
          type: 'success',
          message: `${newTicker.toUpperCase()} added to watchlist`,
        })
      );
      // Refresh to get updated data
      dispatch(fetchWatchlist());
    } catch (error: any) {
      dispatch(
        addNotification({
          type: 'error',
          message: error || 'Failed to add stock to watchlist',
        })
      );
    }
  };

  const handleRemoveStock = async (symbol: string) => {
    try {
      await dispatch(removeFromWatchlist(symbol)).unwrap();
      dispatch(
        addNotification({
          type: 'info',
          message: `${symbol} removed from watchlist`,
        })
      );
    } catch (error: any) {
      dispatch(
        addNotification({
          type: 'error',
          message: error || 'Failed to remove stock from watchlist',
        })
      );
    }
  };

  const handleEditItem = (item: WatchlistItem) => {
    setSelectedItem(item);
    setEditTargetPrice(item.target_price?.toString() || '');
    setEditNotes(item.notes || '');
    setEditAlertEnabled(item.alert_enabled);
    setEditDialogOpen(true);
    handleMenuClose();
  };

  const handleSaveEdit = async () => {
    if (!selectedItem || !watchlist) return;

    try {
      await dispatch(
        updateWatchlistItem({
          watchlistId: watchlist.id,
          itemId: selectedItem.id,
          updates: {
            target_price: editTargetPrice ? parseFloat(editTargetPrice) : null,
            notes: editNotes || null,
            alert_enabled: editAlertEnabled,
          },
        })
      ).unwrap();

      setEditDialogOpen(false);
      setSelectedItem(null);
      dispatch(
        addNotification({
          type: 'success',
          message: `${selectedItem.symbol} updated`,
        })
      );
      // Refresh to get updated data
      dispatch(fetchWatchlist());
    } catch (error: any) {
      dispatch(
        addNotification({
          type: 'error',
          message: error || 'Failed to update watchlist item',
        })
      );
    }
  };

  const handleToggleAlert = async (item: WatchlistItem) => {
    if (!watchlist) return;

    try {
      await dispatch(
        updateWatchlistItem({
          watchlistId: watchlist.id,
          itemId: item.id,
          updates: {
            alert_enabled: !item.alert_enabled,
          },
        })
      ).unwrap();
      dispatch(
        addNotification({
          type: 'info',
          message: `Alerts ${!item.alert_enabled ? 'enabled' : 'disabled'} for ${item.symbol}`,
        })
      );
    } catch (error: any) {
      dispatch(
        addNotification({
          type: 'error',
          message: error || 'Failed to toggle alert',
        })
      );
    }
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, item: WatchlistItem) => {
    setAnchorEl(event.currentTarget);
    setSelectedItem(item);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  const formatCurrency = (value: number | null) => {
    if (value === null || value === undefined) return '-';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number | null) => {
    if (value === null || value === undefined) return '-';
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatLargeNumber = (value: number | null) => {
    if (value === null || value === undefined) return '-';
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
  };

  // Get items from watchlist
  const watchlistItems = watchlist?.items || [];

  // Filter and sort
  const filteredData = watchlistItems
    .filter(
      (item) =>
        item.symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.company_name.toLowerCase().includes(searchQuery.toLowerCase())
    )
    .sort((a, b) => {
      switch (sortBy) {
        case 'symbol':
          return a.symbol.localeCompare(b.symbol);
        case 'change':
          return (b.price_change_percent || 0) - (a.price_change_percent || 0);
        case 'price':
          return (b.current_price || 0) - (a.current_price || 0);
        default:
          return 0;
      }
    });

  // Calculate summary stats
  const totalValue = watchlistItems.reduce(
    (sum, item) => sum + (item.current_price || 0),
    0
  );
  const gainers = watchlistItems.filter(
    (item) => item.price_change !== null && item.price_change > 0
  ).length;
  const losers = watchlistItems.filter(
    (item) => item.price_change !== null && item.price_change < 0
  ).length;
  const alertsEnabled = watchlistItems.filter((item) => item.alert_enabled).length;

  // Loading skeleton
  const TableSkeleton = () => (
    <>
      {[1, 2, 3, 4, 5].map((i) => (
        <TableRow key={i}>
          <TableCell>
            <Skeleton width={60} />
          </TableCell>
          <TableCell>
            <Skeleton width={150} />
          </TableCell>
          <TableCell align="right">
            <Skeleton width={80} />
          </TableCell>
          <TableCell align="right">
            <Skeleton width={100} />
          </TableCell>
          <TableCell align="right">
            <Skeleton width={80} />
          </TableCell>
          <TableCell align="center">
            <Skeleton width={30} />
          </TableCell>
          <TableCell align="center">
            <Skeleton width={40} />
          </TableCell>
        </TableRow>
      ))}
    </>
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3 }}>
        <Box
          sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}
        >
          <Typography variant="h4" fontWeight="bold">
            Watchlist
          </Typography>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              variant="outlined"
              startIcon={watchlistLoading ? <CircularProgress size={18} /> : <Refresh />}
              onClick={handleRefresh}
              disabled={watchlistLoading}
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

        {/* Error Alert */}
        {watchlistError && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => {}}>
            {watchlistError}
          </Alert>
        )}

        {/* Summary Cards */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Total Stocks
                </Typography>
                <Typography variant="h4" fontWeight="bold">
                  {watchlistLoading ? <Skeleton width={40} /> : watchlistItems.length}
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
                  {watchlistLoading ? <Skeleton width={40} /> : gainers}
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
                  {watchlistLoading ? <Skeleton width={40} /> : losers}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Alerts Active
                </Typography>
                <Typography variant="h4" fontWeight="bold" color="primary.main">
                  {watchlistLoading ? <Skeleton width={40} /> : alertsEnabled}
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
                const options: Array<'symbol' | 'change' | 'price'> = [
                  'symbol',
                  'change',
                  'price',
                ];
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
                <TableCell align="right">Target Price</TableCell>
                <TableCell align="center">Alerts</TableCell>
                <TableCell align="center">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {watchlistLoading && watchlistItems.length === 0 ? (
                <TableSkeleton />
              ) : (
                filteredData.map((item) => (
                  <TableRow key={item.id} hover>
                    <TableCell>
                      <Button
                        variant="text"
                        onClick={() => navigate(`/analysis/${item.symbol}`)}
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
                          onClick={() => handleToggleAlert(item)}
                        >
                          {item.alert_enabled ? <NotificationsActive /> : <NotificationsOff />}
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                    <TableCell align="center">
                      <IconButton size="small" onClick={(e) => handleMenuOpen(e, item)}>
                        <MoreVert />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </TableContainer>

        {!watchlistLoading && filteredData.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" color="text.secondary">
              {searchQuery
                ? 'No stocks found matching your search'
                : 'Your watchlist is empty'}
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
        <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleMenuClose}>
          <MenuItem
            onClick={() => {
              if (selectedItem) {
                navigate(`/analysis/${selectedItem.symbol}`);
              }
              handleMenuClose();
            }}
          >
            <RemoveRedEye sx={{ mr: 1 }} /> View Analysis
          </MenuItem>
          <MenuItem
            onClick={() => {
              if (selectedItem) {
                handleEditItem(selectedItem);
              }
            }}
          >
            <Edit sx={{ mr: 1 }} /> Edit Item
          </MenuItem>
          <MenuItem
            onClick={() => {
              if (selectedItem) {
                handleToggleAlert(selectedItem);
              }
              handleMenuClose();
            }}
          >
            <AddAlert sx={{ mr: 1 }} />{' '}
            {selectedItem?.alert_enabled ? 'Disable Alert' : 'Enable Alert'}
          </MenuItem>
          <MenuItem
            onClick={() => {
              if (selectedItem) {
                handleRemoveStock(selectedItem.symbol);
              }
              handleMenuClose();
            }}
          >
            <Delete sx={{ mr: 1 }} /> Remove from Watchlist
          </MenuItem>
        </Menu>

        {/* Add Stock Dialog */}
        <Dialog
          open={addDialogOpen}
          onClose={() => setAddDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
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
              sx={{ mb: 2 }}
            />
            <TextField
              margin="dense"
              label="Target Price (optional)"
              fullWidth
              variant="outlined"
              type="number"
              value={newTargetPrice}
              onChange={(e) => setNewTargetPrice(e.target.value)}
              InputProps={{
                startAdornment: <InputAdornment position="start">$</InputAdornment>,
              }}
              sx={{ mb: 2 }}
            />
            <TextField
              margin="dense"
              label="Notes (optional)"
              fullWidth
              variant="outlined"
              multiline
              rows={2}
              value={newNotes}
              onChange={(e) => setNewNotes(e.target.value)}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setAddDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleAddStock} variant="contained" disabled={!newTicker.trim()}>
              Add
            </Button>
          </DialogActions>
        </Dialog>

        {/* Edit Item Dialog */}
        <Dialog
          open={editDialogOpen}
          onClose={() => setEditDialogOpen(false)}
          maxWidth="sm"
          fullWidth
        >
          <DialogTitle>Edit {selectedItem?.symbol}</DialogTitle>
          <DialogContent>
            <TextField
              autoFocus
              margin="dense"
              label="Target Price"
              fullWidth
              variant="outlined"
              type="number"
              value={editTargetPrice}
              onChange={(e) => setEditTargetPrice(e.target.value)}
              InputProps={{
                startAdornment: <InputAdornment position="start">$</InputAdornment>,
              }}
              sx={{ mb: 2, mt: 1 }}
            />
            <TextField
              margin="dense"
              label="Notes"
              fullWidth
              variant="outlined"
              multiline
              rows={3}
              value={editNotes}
              onChange={(e) => setEditNotes(e.target.value)}
              sx={{ mb: 2 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={editAlertEnabled}
                  onChange={(e) => setEditAlertEnabled(e.target.checked)}
                />
              }
              label="Enable price alerts"
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleSaveEdit} variant="contained">
              Save Changes
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default Watchlist;
