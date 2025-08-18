import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Dialog,
  DialogContent,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Typography,
  Box,
  Chip,
  InputAdornment,
  CircularProgress,
  IconButton,
  Paper,
  Divider,
} from '@mui/material';
import {
  Search as SearchIcon,
  Close as CloseIcon,
  TrendingUp,
  TrendingDown,
  Remove,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../../hooks/redux';
import { searchStocks, clearSearchResults } from '../../store/slices/stockSlice';

// Simple debounce implementation
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;
  return (...args: Parameters<T>) => {
    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

interface SearchModalProps {
  open: boolean;
  onClose: () => void;
}

const SearchModal: React.FC<SearchModalProps> = ({ open, onClose }) => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const { searchResults } = useAppSelector((state) => state.stock);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [popularStocks] = useState([
    { ticker: 'AAPL', name: 'Apple Inc.', change: 2.45 },
    { ticker: 'MSFT', name: 'Microsoft Corporation', change: -0.82 },
    { ticker: 'GOOGL', name: 'Alphabet Inc.', change: 1.23 },
    { ticker: 'AMZN', name: 'Amazon.com Inc.', change: 3.21 },
    { ticker: 'NVDA', name: 'NVIDIA Corporation', change: 5.67 },
    { ticker: 'TSLA', name: 'Tesla Inc.', change: -2.34 },
    { ticker: 'META', name: 'Meta Platforms Inc.', change: 0.98 },
    { ticker: 'BRK.B', name: 'Berkshire Hathaway', change: 0.45 },
  ]);

  // Load recent searches from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('recentSearches');
    if (saved) {
      setRecentSearches(JSON.parse(saved));
    }
  }, []);

  // Debounced search function
  const debouncedSearch = useCallback(
    debounce(async (query: string) => {
      if (query.length >= 2) {
        setIsSearching(true);
        try {
          await dispatch(searchStocks(query)).unwrap();
        } catch (error) {
          console.error('Search failed:', error);
        } finally {
          setIsSearching(false);
        }
      }
    }, 300),
    [dispatch]
  );

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const query = event.target.value;
    setSearchQuery(query);
    
    if (query.length === 0) {
      dispatch(clearSearchResults());
    } else {
      debouncedSearch(query);
    }
  };

  const handleStockClick = (ticker: string) => {
    // Save to recent searches
    const updated = [ticker, ...recentSearches.filter(t => t !== ticker)].slice(0, 5);
    setRecentSearches(updated);
    localStorage.setItem('recentSearches', JSON.stringify(updated));
    
    // Navigate to stock analysis page
    navigate(`/analysis/${ticker}`);
    onClose();
    setSearchQuery('');
    dispatch(clearSearchResults());
  };

  const handleClose = () => {
    onClose();
    setSearchQuery('');
    dispatch(clearSearchResults());
  };

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Escape') {
      handleClose();
    } else if (event.key === 'Enter' && searchResults.length > 0) {
      handleStockClick(searchResults[0].ticker);
    }
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp sx={{ color: 'success.main' }} />;
    if (change < 0) return <TrendingDown sx={{ color: 'error.main' }} />;
    return <Remove sx={{ color: 'text.secondary' }} />;
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          position: 'fixed',
          top: '10%',
          m: 0,
          maxHeight: '80vh',
        },
      }}
    >
      <DialogContent sx={{ p: 0 }}>
        <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <TextField
            fullWidth
            autoFocus
            placeholder="Search stocks, ETFs, or enter a ticker..."
            value={searchQuery}
            onChange={handleSearchChange}
            onKeyDown={handleKeyDown}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon />
                </InputAdornment>
              ),
              endAdornment: (
                <InputAdornment position="end">
                  {isSearching ? (
                    <CircularProgress size={20} />
                  ) : (
                    <IconButton size="small" onClick={handleClose}>
                      <CloseIcon />
                    </IconButton>
                  )}
                </InputAdornment>
              ),
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                '& fieldset': {
                  border: 'none',
                },
              },
            }}
          />
        </Box>

        <Box sx={{ maxHeight: '60vh', overflow: 'auto' }}>
          {searchQuery.length > 0 && searchResults.length > 0 ? (
            <List>
              {searchResults.map((result) => (
                <ListItem
                  key={result.ticker}
                  button
                  onClick={() => handleStockClick(result.ticker)}
                  sx={{
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  }}
                >
                  <ListItemAvatar>
                    <Avatar sx={{ bgcolor: 'primary.main' }}>
                      {result.ticker.charAt(0)}
                    </Avatar>
                  </ListItemAvatar>
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {result.ticker}
                        </Typography>
                        <Chip label={result.exchange} size="small" />
                      </Box>
                    }
                    secondary={result.name}
                  />
                </ListItem>
              ))}
            </List>
          ) : searchQuery.length > 0 && !isSearching ? (
            <Box sx={{ p: 4, textAlign: 'center' }}>
              <Typography color="text.secondary">
                No results found for "{searchQuery}"
              </Typography>
            </Box>
          ) : (
            <>
              {recentSearches.length > 0 && (
                <>
                  <Box sx={{ px: 2, pt: 2 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Recent Searches
                    </Typography>
                  </Box>
                  <List dense>
                    {recentSearches.map((ticker) => (
                      <ListItem
                        key={ticker}
                        button
                        onClick={() => handleStockClick(ticker)}
                      >
                        <ListItemText primary={ticker} />
                      </ListItem>
                    ))}
                  </List>
                  <Divider />
                </>
              )}

              <Box sx={{ px: 2, pt: 2 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Popular Stocks
                </Typography>
              </Box>
              <List>
                {popularStocks.map((stock) => (
                  <ListItem
                    key={stock.ticker}
                    button
                    onClick={() => handleStockClick(stock.ticker)}
                    sx={{
                      '&:hover': {
                        backgroundColor: 'action.hover',
                      },
                    }}
                  >
                    <ListItemAvatar>
                      {getChangeIcon(stock.change)}
                    </ListItemAvatar>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Typography variant="subtitle1" fontWeight="bold">
                            {stock.ticker}
                          </Typography>
                          <Typography
                            variant="subtitle1"
                            color={stock.change > 0 ? 'success.main' : stock.change < 0 ? 'error.main' : 'text.secondary'}
                          >
                            {stock.change > 0 ? '+' : ''}{stock.change}%
                          </Typography>
                        </Box>
                      }
                      secondary={stock.name}
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
        </Box>

        {searchQuery.length === 0 && (
          <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider', bgcolor: 'background.default' }}>
            <Typography variant="caption" color="text.secondary">
              Press ESC to close • Use arrow keys to navigate • Press Enter to select
            </Typography>
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default SearchModal;