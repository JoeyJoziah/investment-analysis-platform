import React from 'react';
import {
  Grid,
  TextField,
  MenuItem,
  InputAdornment,
  Paper,
  Slider,
  FormControl,
  InputLabel,
  Select,
  Typography,
  Box,
} from '@mui/material';
import { Search as SearchIcon } from '@mui/icons-material';

export interface RecommendationFilters {
  recommendation: string;
  sector: string;
  risk: string;
  timeHorizon: string;
  minConfidence: number;
  sortBy: string;
}

export interface RecommendationsFilterProps {
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  filters: RecommendationFilters;
  onFiltersChange: (filters: RecommendationFilters) => void;
  uniqueSectors: string[];
}

const RecommendationsFilter: React.FC<RecommendationsFilterProps> = ({
  searchQuery,
  onSearchQueryChange,
  filters,
  onFiltersChange,
  uniqueSectors,
}) => (
  <Paper sx={{ p: 2, mb: 3 }}>
    <Grid container spacing={2} alignItems="center">
      <Grid item xs={12} md={3}>
        <TextField
          fullWidth
          placeholder="Search ticker or company..."
          value={searchQuery}
          onChange={(e) => onSearchQueryChange(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
      </Grid>
      <Grid item xs={12} sm={6} md={2}>
        <FormControl fullWidth size="small">
          <InputLabel>Recommendation</InputLabel>
          <Select
            value={filters.recommendation}
            label="Recommendation"
            onChange={(e) => onFiltersChange({ ...filters, recommendation: e.target.value })}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="STRONG_BUY">Strong Buy</MenuItem>
            <MenuItem value="BUY">Buy</MenuItem>
            <MenuItem value="HOLD">Hold</MenuItem>
            <MenuItem value="SELL">Sell</MenuItem>
            <MenuItem value="STRONG_SELL">Strong Sell</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} sm={6} md={2}>
        <FormControl fullWidth size="small">
          <InputLabel>Sector</InputLabel>
          <Select
            value={filters.sector}
            label="Sector"
            onChange={(e) => onFiltersChange({ ...filters, sector: e.target.value })}
          >
            <MenuItem value="all">All Sectors</MenuItem>
            {uniqueSectors.map((sector) => (
              <MenuItem key={sector} value={sector}>
                {sector}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} sm={6} md={1.5}>
        <FormControl fullWidth size="small">
          <InputLabel>Risk</InputLabel>
          <Select
            value={filters.risk}
            label="Risk"
            onChange={(e) => onFiltersChange({ ...filters, risk: e.target.value })}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="LOW">Low</MenuItem>
            <MenuItem value="MEDIUM">Medium</MenuItem>
            <MenuItem value="HIGH">High</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} sm={6} md={1.5}>
        <FormControl fullWidth size="small">
          <InputLabel>Time Horizon</InputLabel>
          <Select
            value={filters.timeHorizon}
            label="Time Horizon"
            onChange={(e) => onFiltersChange({ ...filters, timeHorizon: e.target.value })}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="SHORT">Short Term</MenuItem>
            <MenuItem value="MEDIUM">Medium Term</MenuItem>
            <MenuItem value="LONG">Long Term</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12} sm={6} md={2}>
        <FormControl fullWidth size="small">
          <InputLabel>Sort By</InputLabel>
          <Select
            value={filters.sortBy}
            label="Sort By"
            onChange={(e) => onFiltersChange({ ...filters, sortBy: e.target.value })}
          >
            <MenuItem value="confidence">Confidence</MenuItem>
            <MenuItem value="expectedReturn">Expected Return</MenuItem>
            <MenuItem value="ticker">Ticker</MenuItem>
          </Select>
        </FormControl>
      </Grid>
      <Grid item xs={12}>
        <Box sx={{ px: 2 }}>
          <Typography variant="body2" gutterBottom>
            Minimum Confidence: {filters.minConfidence}%
          </Typography>
          <Slider
            value={filters.minConfidence}
            onChange={(_, value) => onFiltersChange({ ...filters, minConfidence: value as number })}
            valueLabelDisplay="auto"
            min={0}
            max={100}
            marks={[
              { value: 0, label: '0%' },
              { value: 50, label: '50%' },
              { value: 100, label: '100%' },
            ]}
          />
        </Box>
      </Grid>
    </Grid>
  </Paper>
);

export default RecommendationsFilter;
