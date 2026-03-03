import React from 'react';
import {
  Box,
  Button,
  TextField,
  InputAdornment,
  Switch,
  FormControlLabel,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
} from '@mui/material';
import {
  Delete,
  RemoveRedEye,
  AddAlert,
  Edit,
} from '@mui/icons-material';
import { WatchlistItem } from '../../store/slices/portfolioSlice';

// --- Context Menu ---

export interface WatchlistContextMenuProps {
  anchorEl: HTMLElement | null;
  selectedItem: WatchlistItem | null;
  onClose: () => void;
  onViewAnalysis: (symbol: string) => void;
  onEditItem: (item: WatchlistItem) => void;
  onToggleAlert: (item: WatchlistItem) => void;
  onRemoveStock: (symbol: string) => void;
}

export const WatchlistContextMenu: React.FC<WatchlistContextMenuProps> = ({
  anchorEl,
  selectedItem,
  onClose,
  onViewAnalysis,
  onEditItem,
  onToggleAlert,
  onRemoveStock,
}) => (
  <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={onClose}>
    <MenuItem
      onClick={() => {
        if (selectedItem) {
          onViewAnalysis(selectedItem.symbol);
        }
        onClose();
      }}
    >
      <RemoveRedEye sx={{ mr: 1 }} /> View Analysis
    </MenuItem>
    <MenuItem
      onClick={() => {
        if (selectedItem) {
          onEditItem(selectedItem);
        }
      }}
    >
      <Edit sx={{ mr: 1 }} /> Edit Item
    </MenuItem>
    <MenuItem
      onClick={() => {
        if (selectedItem) {
          onToggleAlert(selectedItem);
        }
        onClose();
      }}
    >
      <AddAlert sx={{ mr: 1 }} />{' '}
      {selectedItem?.alert_enabled ? 'Disable Alert' : 'Enable Alert'}
    </MenuItem>
    <MenuItem
      onClick={() => {
        if (selectedItem) {
          onRemoveStock(selectedItem.symbol);
        }
        onClose();
      }}
    >
      <Delete sx={{ mr: 1 }} /> Remove from Watchlist
    </MenuItem>
  </Menu>
);

// --- Add Stock Dialog ---

export interface AddStockDialogProps {
  open: boolean;
  ticker: string;
  targetPrice: string;
  notes: string;
  onClose: () => void;
  onTickerChange: (ticker: string) => void;
  onTargetPriceChange: (price: string) => void;
  onNotesChange: (notes: string) => void;
  onSubmit: () => void;
}

export const AddStockDialog: React.FC<AddStockDialogProps> = ({
  open,
  ticker,
  targetPrice,
  notes,
  onClose,
  onTickerChange,
  onTargetPriceChange,
  onNotesChange,
  onSubmit,
}) => (
  <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
    <DialogTitle>Add Stock to Watchlist</DialogTitle>
    <DialogContent>
      <TextField
        autoFocus
        margin="dense"
        label="Ticker Symbol"
        fullWidth
        variant="outlined"
        value={ticker}
        onChange={(e) => onTickerChange(e.target.value.toUpperCase())}
        onKeyDown={(e) => e.key === 'Enter' && onSubmit()}
        sx={{ mb: 2 }}
      />
      <TextField
        margin="dense"
        label="Target Price (optional)"
        fullWidth
        variant="outlined"
        type="number"
        value={targetPrice}
        onChange={(e) => onTargetPriceChange(e.target.value)}
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
        value={notes}
        onChange={(e) => onNotesChange(e.target.value)}
      />
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Cancel</Button>
      <Button onClick={onSubmit} variant="contained" disabled={!ticker.trim()}>
        Add
      </Button>
    </DialogActions>
  </Dialog>
);

// --- Edit Item Dialog ---

export interface EditItemDialogProps {
  open: boolean;
  selectedItem: WatchlistItem | null;
  targetPrice: string;
  notes: string;
  alertEnabled: boolean;
  onClose: () => void;
  onTargetPriceChange: (price: string) => void;
  onNotesChange: (notes: string) => void;
  onAlertEnabledChange: (enabled: boolean) => void;
  onSave: () => void;
}

export const EditItemDialog: React.FC<EditItemDialogProps> = ({
  open,
  selectedItem,
  targetPrice,
  notes,
  alertEnabled,
  onClose,
  onTargetPriceChange,
  onNotesChange,
  onAlertEnabledChange,
  onSave,
}) => (
  <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
    <DialogTitle>Edit {selectedItem?.symbol}</DialogTitle>
    <DialogContent>
      <TextField
        autoFocus
        margin="dense"
        label="Target Price"
        fullWidth
        variant="outlined"
        type="number"
        value={targetPrice}
        onChange={(e) => onTargetPriceChange(e.target.value)}
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
        value={notes}
        onChange={(e) => onNotesChange(e.target.value)}
        sx={{ mb: 2 }}
      />
      <FormControlLabel
        control={
          <Switch
            checked={alertEnabled}
            onChange={(e) => onAlertEnabledChange(e.target.checked)}
          />
        }
        label="Enable price alerts"
      />
    </DialogContent>
    <DialogActions>
      <Button onClick={onClose}>Cancel</Button>
      <Button onClick={onSave} variant="contained">
        Save Changes
      </Button>
    </DialogActions>
  </Dialog>
);

// --- Empty State ---

export interface WatchlistEmptyStateProps {
  searchQuery: string;
  onAddStock: () => void;
}

export const WatchlistEmptyState: React.FC<WatchlistEmptyStateProps> = ({
  searchQuery,
  onAddStock,
}) => (
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
        onClick={onAddStock}
      >
        Add Your First Stock
      </Button>
    )}
  </Box>
);
