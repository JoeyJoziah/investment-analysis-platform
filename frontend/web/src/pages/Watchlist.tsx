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
  Button,
  TextField,
  InputAdornment,
  Alert,
  CircularProgress,
  Skeleton,
} from '@mui/material';
import {
  Add,
  Search,
  Refresh,
  SortByAlpha,
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
import WatchlistTable from '../components/watchlist/WatchlistTable';
import {
  WatchlistContextMenu,
  AddStockDialog,
  EditItemDialog,
  WatchlistEmptyState,
} from '../components/watchlist/WatchlistActions';

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
      dispatch(fetchWatchlist());
    } catch (error: unknown) {
      const message = typeof error === 'string' ? error : 'Failed to add stock to watchlist';
      dispatch(
        addNotification({
          type: 'error',
          message,
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
    } catch (error: unknown) {
      const message = typeof error === 'string' ? error : 'Failed to remove stock from watchlist';
      dispatch(
        addNotification({
          type: 'error',
          message,
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
    setAnchorEl(null);
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
      dispatch(fetchWatchlist());
    } catch (error: unknown) {
      const message = typeof error === 'string' ? error : 'Failed to update watchlist item';
      dispatch(
        addNotification({
          type: 'error',
          message,
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
    } catch (error: unknown) {
      const message = typeof error === 'string' ? error : 'Failed to toggle alert';
      dispatch(
        addNotification({
          type: 'error',
          message,
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
  const gainers = watchlistItems.filter(
    (item) => item.price_change !== null && item.price_change > 0
  ).length;
  const losers = watchlistItems.filter(
    (item) => item.price_change !== null && item.price_change < 0
  ).length;
  const alertsEnabled = watchlistItems.filter((item) => item.alert_enabled).length;

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

        {watchlistError && (
          <Alert severity="info" sx={{ mb: 2 }} onClose={() => {}}>
            Your watchlist isn&apos;t available yet — there&apos;s no saved watchlist data in this environment.
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
        <WatchlistTable
          items={filteredData}
          isLoading={watchlistLoading}
          onNavigateToAnalysis={(symbol) => navigate(`/analysis/${symbol}`)}
          onToggleAlert={handleToggleAlert}
          onMenuOpen={handleMenuOpen}
        />

        {!watchlistLoading && filteredData.length === 0 && (
          <WatchlistEmptyState
            searchQuery={searchQuery}
            onAddStock={() => setAddDialogOpen(true)}
          />
        )}

        {/* Context Menu */}
        <WatchlistContextMenu
          anchorEl={anchorEl}
          selectedItem={selectedItem}
          onClose={handleMenuClose}
          onViewAnalysis={(symbol) => navigate(`/analysis/${symbol}`)}
          onEditItem={handleEditItem}
          onToggleAlert={handleToggleAlert}
          onRemoveStock={handleRemoveStock}
        />

        {/* Add Stock Dialog */}
        <AddStockDialog
          open={addDialogOpen}
          ticker={newTicker}
          targetPrice={newTargetPrice}
          notes={newNotes}
          onClose={() => setAddDialogOpen(false)}
          onTickerChange={setNewTicker}
          onTargetPriceChange={setNewTargetPrice}
          onNotesChange={setNewNotes}
          onSubmit={handleAddStock}
        />

        {/* Edit Item Dialog */}
        <EditItemDialog
          open={editDialogOpen}
          selectedItem={selectedItem}
          targetPrice={editTargetPrice}
          notes={editNotes}
          alertEnabled={editAlertEnabled}
          onClose={() => setEditDialogOpen(false)}
          onTargetPriceChange={setEditTargetPrice}
          onNotesChange={setEditNotes}
          onAlertEnabledChange={setEditAlertEnabled}
          onSave={handleSaveEdit}
        />
      </Box>
    </Container>
  );
};

export default Watchlist;
