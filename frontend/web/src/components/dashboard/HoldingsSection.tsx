import React, { memo, useState, useCallback } from 'react';
import {
  Box,
  Typography,
  Button,
  Skeleton,
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { Position } from '../../types';
import HoldingsTable from './HoldingsTable';
import type { SortColumn, SortDirection } from './HoldingsTable';

interface HoldingsSectionProps {
  positions: Position[];
  isLoading?: boolean;
  maxRows?: number;
  onAddPosition?: () => void;
}

/**
 * HoldingsSection - Portfolio holdings orchestrator.
 *
 * Manages sort state, show-all toggle, and delegates rendering to
 * HoldingsTable and HoldingsActions sub-components. Handles loading
 * and empty states inline.
 */
const HoldingsSection: React.FC<HoldingsSectionProps> = ({
  positions = [],
  isLoading = false,
  maxRows = 10,
  onAddPosition,
}) => {
  const navigate = useNavigate();

  const [sortBy, setSortBy] = useState<SortColumn>('marketValue');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');
  const [showAll, setShowAll] = useState(false);

  const handleSort = useCallback(
    (column: SortColumn) => {
      const isAsc = sortBy === column && sortDirection === 'asc';
      setSortDirection(isAsc ? 'desc' : 'asc');
      setSortBy(column);
    },
    [sortBy, sortDirection]
  );

  const handleRowClick = useCallback(
    (ticker: string) => {
      navigate(`/stocks/${ticker}`);
    },
    [navigate]
  );

  const handleToggleShowAll = useCallback(() => {
    setShowAll((prev) => !prev);
  }, []);

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Skeleton variant="text" width={120} height={28} />
          <Skeleton variant="rectangular" width={100} height={32} sx={{ borderRadius: 1 }} />
        </Box>
        <HoldingsTable
          positions={[]}
          sortBy={sortBy}
          sortDirection={sortDirection}
          showAll={false}
          maxRows={maxRows}
          onSort={handleSort}
          onRowClick={handleRowClick}
          isLoading
        />
      </Box>
    );
  }

  // Empty state
  if (positions.length === 0) {
    return (
      <Box>
        <Typography variant="h6" component="h2" gutterBottom>
          Holdings
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            py: 6,
            color: 'text.secondary',
          }}
        >
          <Typography variant="body1" gutterBottom>
            No positions in your portfolio
          </Typography>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={onAddPosition}
            sx={{ mt: 2 }}
          >
            Add Position
          </Button>
        </Box>
      </Box>
    );
  }

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography variant="h6" component="h2">
          Holdings
        </Typography>
        <Button
          variant="outlined"
          size="small"
          startIcon={<AddIcon />}
          onClick={onAddPosition}
        >
          Add Position
        </Button>
      </Box>

      {/* Table */}
      <HoldingsTable
        positions={positions}
        sortBy={sortBy}
        sortDirection={sortDirection}
        showAll={showAll}
        maxRows={maxRows}
        onSort={handleSort}
        onRowClick={handleRowClick}
      />

      {/* Show more/less button */}
      {positions.length > maxRows && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
          <Button
            size="small"
            onClick={handleToggleShowAll}
            aria-expanded={showAll}
            aria-controls="holdings-table"
          >
            {showAll
              ? 'Show Less'
              : `Show All (${positions.length - maxRows} more)`}
          </Button>
        </Box>
      )}
    </Box>
  );
};

export default memo(HoldingsSection);
