import React from 'react';
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  TextField,
  MenuItem,
  Typography,
} from '@mui/material';

export interface TransactionFormData {
  ticker: string;
  type: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  notes: string;
}

export interface AddTransactionDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: () => void;
  transactionForm: TransactionFormData;
  onFormChange: (form: TransactionFormData) => void;
  formatCurrency: (value: number) => string;
}

export const AddTransactionDialog: React.FC<AddTransactionDialogProps> = ({
  open,
  onClose,
  onSubmit,
  transactionForm,
  onFormChange,
  formatCurrency,
}) => {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Add Transaction</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 2 }}>
          <TextField
            label="Ticker Symbol"
            value={transactionForm.ticker}
            onChange={(e) =>
              onFormChange({ ...transactionForm, ticker: e.target.value.toUpperCase() })
            }
            fullWidth
          />
          <TextField
            select
            label="Type"
            value={transactionForm.type}
            onChange={(e) =>
              onFormChange({ ...transactionForm, type: e.target.value as 'BUY' | 'SELL' })
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
              onFormChange({ ...transactionForm, quantity: Number(e.target.value) })
            }
            fullWidth
          />
          <TextField
            label="Price per Share"
            type="number"
            value={transactionForm.price}
            onChange={(e) =>
              onFormChange({ ...transactionForm, price: Number(e.target.value) })
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
              onFormChange({ ...transactionForm, notes: e.target.value })
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
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={onSubmit}
          variant="contained"
          disabled={!transactionForm.ticker || transactionForm.quantity === 0 || transactionForm.price === 0}
        >
          Add Transaction
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export interface DeleteConfirmDialogProps {
  open: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

export const DeleteConfirmDialog: React.FC<DeleteConfirmDialogProps> = ({
  open,
  onCancel,
  onConfirm,
}) => {
  return (
    <Dialog
      open={open}
      onClose={onCancel}
      aria-labelledby="delete-confirm-title"
      aria-describedby="delete-confirm-description"
    >
      <DialogTitle id="delete-confirm-title">Delete Position</DialogTitle>
      <DialogContent>
        <DialogContentText id="delete-confirm-description">
          Are you sure you want to delete this position? This action cannot be undone.
        </DialogContentText>
      </DialogContent>
      <DialogActions>
        <Button onClick={onCancel} autoFocus>
          Cancel
        </Button>
        <Button onClick={onConfirm} color="error" variant="contained">
          Delete
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default { AddTransactionDialog, DeleteConfirmDialog };
