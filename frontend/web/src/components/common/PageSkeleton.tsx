import React from 'react';
import { Box, Skeleton, Grid, Paper } from '@mui/material';

interface PageSkeletonProps {
  type?: 'dashboard' | 'portfolio' | 'analysis' | 'list' | 'default';
}

/**
 * PageSkeleton - Skeleton loader for lazy-loaded pages
 *
 * Provides visual feedback during page loading with layout-specific skeletons
 * that match the target page structure for smoother perceived transitions.
 */
const PageSkeleton: React.FC<PageSkeletonProps> = ({ type = 'default' }) => {
  // Dashboard skeleton with metric cards and chart areas
  if (type === 'dashboard') {
    return (
      <Box sx={{ p: 3 }}>
        {/* Header skeleton */}
        <Box sx={{ mb: 3 }}>
          <Skeleton variant="text" width={200} height={40} />
          <Skeleton variant="text" width={300} height={24} />
        </Box>

        {/* Metric cards row */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Paper sx={{ p: 2 }}>
                <Skeleton variant="text" width="60%" height={20} />
                <Skeleton variant="text" width="80%" height={36} sx={{ my: 1 }} />
                <Skeleton variant="text" width="40%" height={20} />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Main content area */}
        <Grid container spacing={3}>
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3 }}>
              <Skeleton variant="text" width={150} height={28} sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" height={300} />
            </Paper>
          </Grid>
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, height: '100%' }}>
              <Skeleton variant="text" width={180} height={28} sx={{ mb: 2 }} />
              {[1, 2, 3].map((i) => (
                <Box key={i} sx={{ mb: 2 }}>
                  <Skeleton variant="rectangular" height={80} sx={{ borderRadius: 1 }} />
                </Box>
              ))}
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  }

  // Portfolio skeleton with table structure
  if (type === 'portfolio') {
    return (
      <Box sx={{ p: 3 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Skeleton variant="text" width={150} height={40} />
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Skeleton variant="rectangular" width={100} height={36} sx={{ borderRadius: 1 }} />
            <Skeleton variant="rectangular" width={140} height={36} sx={{ borderRadius: 1 }} />
          </Box>
        </Box>

        {/* Summary cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          {[1, 2, 3, 4].map((i) => (
            <Grid item xs={12} sm={6} md={3} key={i}>
              <Paper sx={{ p: 2 }}>
                <Skeleton variant="text" width="50%" height={18} />
                <Skeleton variant="text" width="70%" height={32} sx={{ my: 1 }} />
                <Skeleton variant="text" width="40%" height={18} />
              </Paper>
            </Grid>
          ))}
        </Grid>

        {/* Table area */}
        <Paper sx={{ p: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
            ))}
          </Box>
          {/* Table rows */}
          {[1, 2, 3, 4, 5, 6].map((row) => (
            <Box
              key={row}
              sx={{
                display: 'flex',
                gap: 2,
                py: 2,
                borderBottom: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Skeleton variant="text" width="10%" height={24} />
              <Skeleton variant="text" width="20%" height={24} />
              <Skeleton variant="text" width="10%" height={24} />
              <Skeleton variant="text" width="10%" height={24} />
              <Skeleton variant="text" width="10%" height={24} />
              <Skeleton variant="text" width="15%" height={24} />
              <Skeleton variant="text" width="15%" height={24} />
            </Box>
          ))}
        </Paper>
      </Box>
    );
  }

  // Analysis skeleton with chart and side panels
  if (type === 'analysis') {
    return (
      <Box sx={{ p: 3 }}>
        {/* Header with search */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Box>
            <Skeleton variant="text" width={120} height={40} />
            <Skeleton variant="text" width={200} height={24} />
          </Box>
          <Skeleton variant="rectangular" width={300} height={40} sx={{ borderRadius: 1 }} />
        </Box>

        <Grid container spacing={3}>
          {/* Main chart area */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                <Skeleton variant="text" width={150} height={32} />
                <Box sx={{ display: 'flex', gap: 1 }}>
                  {[1, 2, 3, 4, 5].map((i) => (
                    <Skeleton key={i} variant="rectangular" width={40} height={32} sx={{ borderRadius: 1 }} />
                  ))}
                </Box>
              </Box>
              <Skeleton variant="rectangular" height={400} />
            </Paper>
          </Grid>

          {/* Side panel */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 3, mb: 3 }}>
              <Skeleton variant="text" width={120} height={28} sx={{ mb: 2 }} />
              {[1, 2, 3, 4, 5].map((i) => (
                <Box key={i} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Skeleton variant="text" width="40%" height={20} />
                  <Skeleton variant="text" width="30%" height={20} />
                </Box>
              ))}
            </Paper>
            <Paper sx={{ p: 3 }}>
              <Skeleton variant="text" width={100} height={28} sx={{ mb: 2 }} />
              <Skeleton variant="rectangular" height={200} />
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  }

  // List skeleton for watchlist, alerts, reports
  if (type === 'list') {
    return (
      <Box sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
          <Skeleton variant="text" width={150} height={40} />
          <Skeleton variant="rectangular" width={120} height={36} sx={{ borderRadius: 1 }} />
        </Box>

        <Paper sx={{ p: 3 }}>
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <Box
              key={i}
              sx={{
                display: 'flex',
                alignItems: 'center',
                gap: 2,
                py: 2,
                borderBottom: '1px solid',
                borderColor: 'divider',
              }}
            >
              <Skeleton variant="circular" width={40} height={40} />
              <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="30%" height={24} />
                <Skeleton variant="text" width="50%" height={18} />
              </Box>
              <Skeleton variant="text" width={80} height={24} />
              <Skeleton variant="rectangular" width={60} height={28} sx={{ borderRadius: 1 }} />
            </Box>
          ))}
        </Paper>
      </Box>
    );
  }

  // Default skeleton
  return (
    <Box sx={{ p: 3 }}>
      <Skeleton variant="text" width={200} height={40} sx={{ mb: 2 }} />
      <Skeleton variant="text" width={400} height={24} sx={{ mb: 3 }} />
      <Paper sx={{ p: 3 }}>
        <Skeleton variant="rectangular" height={400} />
      </Paper>
    </Box>
  );
};

export default PageSkeleton;
