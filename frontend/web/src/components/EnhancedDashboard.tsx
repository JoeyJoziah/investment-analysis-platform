/**
 * Enhanced Dashboard with improved accessibility, usability, and performance
 * Implements WCAG 2.1 AA standards and modern UX patterns
 */

import React, { useState, useEffect, useCallback, useMemo, lazy, Suspense } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
  Alert,
  Button,
  useTheme,
  alpha,
  Skeleton,
  Fade,
  Zoom,
  useMediaQuery,
  Snackbar,
  CircularProgress,
  Badge,
  Menu,
  MenuItem,
  Divider,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Assessment,
  Notifications,
  Speed,
  AttachMoney,
  PieChart,
  Timeline,
  Refresh,
  MoreVert,
  Settings,
  Fullscreen,
  FullscreenExit,
  Download,
  Share,
  FilterList,
  ViewModule,
  ViewList,
  DragIndicator,
  ErrorOutline,
  CheckCircle,
  Warning,
  Info,
  Undo,
  Redo,
  Save,
  CloudUpload,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import { fetchDashboardData } from '../store/slices/dashboardSlice';
import { fetchRecommendations } from '../store/slices/recommendationsSlice';
import { designTokens } from '../theme/tokens';
import { 
  announceToScreenReader, 
  useKeyboardNavigation, 
  useReducedMotion,
  ScreenReaderOnly,
  useFocusVisible,
  useAriaLive,
} from '../utils/accessibility';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import { ErrorBoundary } from 'react-error-boundary';
import { useInView } from 'react-intersection-observer';
import { motion, AnimatePresence } from 'framer-motion';

// Lazy load heavy components
const StockChart = lazy(() => import('./charts/StockChart'));
const RecommendationCard = lazy(() => import('./cards/RecommendationCard'));
const MarketHeatmap = lazy(() => import('./charts/MarketHeatmap'));
const PortfolioSummary = lazy(() => import('./cards/PortfolioSummary'));
const NewsCard = lazy(() => import('./cards/NewsCard'));
const CostMonitor = lazy(() => import('./monitoring/CostMonitor'));

// Loading skeleton component
const LoadingSkeleton = ({ height = 200 }: { height?: number }) => (
  <Skeleton 
    variant="rectangular" 
    height={height} 
    animation="wave"
    sx={{ borderRadius: designTokens.borderRadius.md }}
  />
);

// Error fallback component
const ErrorFallback = ({ error, resetErrorBoundary }: any) => (
  <Alert 
    severity="error" 
    action={
      <Button color="inherit" size="small" onClick={resetErrorBoundary}>
        Retry
      </Button>
    }
  >
    <Typography variant="subtitle2">Something went wrong</Typography>
    <Typography variant="caption">{error.message}</Typography>
  </Alert>
);

// Empty state component
const EmptyState = ({ 
  title, 
  message, 
  action 
}: { 
  title: string; 
  message: string; 
  action?: React.ReactNode;
}) => (
  <Box 
    sx={{ 
      textAlign: 'center', 
      py: designTokens.spacing.xxl,
      px: designTokens.spacing.lg,
    }}
  >
    <ErrorOutline sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
    <Typography variant="h6" gutterBottom>{title}</Typography>
    <Typography variant="body2" color="text.secondary" paragraph>
      {message}
    </Typography>
    {action}
  </Box>
);

// Metric card with enhanced accessibility
const AccessibleMetricCard = ({ 
  title, 
  value, 
  change, 
  icon, 
  color,
  ariaLabel,
  loading = false 
}: any) => {
  const theme = useTheme();
  const reducedMotion = useReducedMotion();
  const [ref, inView] = useInView({ triggerOnce: true, threshold: 0.1 });
  
  return (
    <motion.div
      ref={ref}
      initial={!reducedMotion ? { opacity: 0, y: 20 } : {}}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.3 }}
    >
      <Card 
        sx={{ 
          height: '100%', 
          position: 'relative', 
          overflow: 'visible',
          transition: 'all 0.2s ease-in-out',
          '&:hover': {
            transform: !reducedMotion ? 'translateY(-2px)' : 'none',
            boxShadow: theme.shadows[4],
          },
          '&:focus-within': {
            outline: `2px solid ${theme.palette.primary.main}`,
            outlineOffset: 2,
          }
        }}
        role="article"
        aria-label={ariaLabel || `${title}: ${value}, ${change >= 0 ? 'up' : 'down'} ${Math.abs(change)}%`}
        tabIndex={0}
      >
        <CardContent>
          {loading ? (
            <>
              <Skeleton width="40%" height={20} sx={{ mb: 1 }} />
              <Skeleton width="60%" height={32} sx={{ mb: 1 }} />
              <Skeleton width="30%" height={20} />
            </>
          ) : (
            <Box display="flex" justifyContent="space-between" alignItems="flex-start">
              <Box>
                <Typography 
                  color="textSecondary" 
                  gutterBottom 
                  variant="body2"
                  component="h3"
                >
                  {title}
                </Typography>
                <Typography 
                  variant="h4" 
                  component="div" 
                  sx={{ mb: 1 }}
                  aria-live="polite"
                >
                  {value}
                </Typography>
                {change !== undefined && (
                  <Box display="flex" alignItems="center" role="status">
                    {change >= 0 ? (
                      <TrendingUp sx={{ color: theme.palette.success.main, mr: 0.5 }} aria-hidden="true" />
                    ) : (
                      <TrendingDown sx={{ color: theme.palette.error.main, mr: 0.5 }} aria-hidden="true" />
                    )}
                    <Typography
                      variant="body2"
                      sx={{
                        color: change >= 0 ? theme.palette.success.main : theme.palette.error.main
                      }}
                    >
                      <ScreenReaderOnly>{change >= 0 ? 'increased' : 'decreased'} by</ScreenReaderOnly>
                      {Math.abs(change)}%
                    </Typography>
                  </Box>
                )}
              </Box>
              <Box
                sx={{
                  backgroundColor: alpha(color || theme.palette.primary.main, 0.1),
                  borderRadius: designTokens.borderRadius.sm,
                  p: 1,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center'
                }}
                aria-hidden="true"
              >
                {icon}
              </Box>
            </Box>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
};

const EnhancedDashboard: React.FC = () => {
  const theme = useTheme();
  const dispatch = useAppDispatch();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));
  const isTablet = useMediaQuery(theme.breakpoints.down('md'));
  const reducedMotion = useReducedMotion();
  
  // State management
  const [refreshing, setRefreshing] = useState(false);
  const [fullscreen, setFullscreen] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(60000); // 1 minute
  const [notification, setNotification] = useState<{
    open: boolean;
    message: string;
    severity: 'success' | 'error' | 'warning' | 'info';
  }>({ open: false, message: '', severity: 'info' });
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [widgetOrder, setWidgetOrder] = useState([
    'metrics',
    'market-overview',
    'recommendations',
    'portfolio-chart',
    'portfolio-summary',
    'news',
    'cost-monitor',
    'sectors',
  ]);
  
  // Undo/Redo functionality
  const [history, setHistory] = useState<any[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  
  const {
    marketOverview,
    topRecommendations,
    portfolioSummary,
    recentNews,
    marketSentiment,
    costMetrics,
    loading,
    error
  } = useAppSelector((state) => state.dashboard);
  
  // Focus management
  useFocusVisible();
  
  // Announce status changes to screen readers
  useAriaLive(
    loading ? 'Loading dashboard data' : 'Dashboard data loaded',
    'polite'
  );
  
  // Keyboard navigation
  useKeyboardNavigation({
    onEscape: () => setFullscreen(false),
    onEnter: () => {
      if (document.activeElement?.getAttribute('role') === 'button') {
        (document.activeElement as HTMLElement).click();
      }
    },
  });
  
  // Auto-refresh functionality
  useEffect(() => {
    if (!autoRefresh) return;
    
    const interval = setInterval(() => {
      handleRefresh();
    }, refreshInterval);
    
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval]);
  
  // Initial data fetch
  useEffect(() => {
    dispatch(fetchDashboardData());
    dispatch(fetchRecommendations({ limit: 5 }));
  }, [dispatch]);
  
  // Handlers
  const handleRefresh = useCallback(async () => {
    setRefreshing(true);
    announceToScreenReader('Refreshing dashboard data');
    
    try {
      await Promise.all([
        dispatch(fetchDashboardData()),
        dispatch(fetchRecommendations({ limit: 5 }))
      ]);
      
      setNotification({
        open: true,
        message: 'Dashboard refreshed successfully',
        severity: 'success'
      });
      announceToScreenReader('Dashboard data refreshed successfully');
    } catch (error) {
      setNotification({
        open: true,
        message: 'Failed to refresh dashboard',
        severity: 'error'
      });
      announceToScreenReader('Failed to refresh dashboard data');
    } finally {
      setRefreshing(false);
    }
  }, [dispatch]);
  
  const handleDragEnd = useCallback((result: any) => {
    if (!result.destination) return;
    
    const items = Array.from(widgetOrder);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);
    
    // Save to history for undo/redo
    setHistory([...history.slice(0, historyIndex + 1), widgetOrder]);
    setHistoryIndex(historyIndex + 1);
    
    setWidgetOrder(items);
    announceToScreenReader('Widget order changed');
  }, [widgetOrder, history, historyIndex]);
  
  const handleUndo = useCallback(() => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setWidgetOrder(history[historyIndex - 1]);
      announceToScreenReader('Undo successful');
    }
  }, [history, historyIndex]);
  
  const handleRedo = useCallback(() => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setWidgetOrder(history[historyIndex + 1]);
      announceToScreenReader('Redo successful');
    }
  }, [history, historyIndex]);
  
  const handleExport = useCallback(() => {
    // Export dashboard data as JSON
    const data = {
      marketOverview,
      topRecommendations,
      portfolioSummary,
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `dashboard-${new Date().toISOString()}.json`;
    a.click();
    
    setNotification({
      open: true,
      message: 'Dashboard data exported',
      severity: 'success'
    });
  }, [marketOverview, topRecommendations, portfolioSummary]);
  
  // Memoized computations
  const metrics = useMemo(() => [
    {
      title: 'Portfolio Value',
      value: portfolioSummary?.totalValue ? 
        `$${portfolioSummary.totalValue.toLocaleString()}` : '$0',
      change: portfolioSummary?.dayChange,
      icon: <AttachMoney sx={{ color: theme.palette.primary.main }} />,
      color: theme.palette.primary.main,
    },
    {
      title: 'Total Return',
      value: portfolioSummary?.totalReturn ? 
        `${portfolioSummary.totalReturn.toFixed(2)}%` : '0%',
      change: portfolioSummary?.totalReturn,
      icon: <Timeline sx={{ color: theme.palette.success.main }} />,
      color: theme.palette.success.main,
    },
    {
      title: 'Market Sentiment',
      value: marketSentiment?.overall || 'Neutral',
      icon: <Speed sx={{ color: theme.palette.info.main }} />,
      color: theme.palette.info.main,
    },
    {
      title: 'Active Positions',
      value: portfolioSummary?.activePositions || 0,
      icon: <PieChart sx={{ color: theme.palette.warning.main }} />,
      color: theme.palette.warning.main,
    },
  ], [portfolioSummary, marketSentiment, theme]);
  
  if (error) {
    return (
      <Box p={3} role="alert">
        <Alert 
          severity="error" 
          action={
            <Button 
              color="inherit" 
              size="small" 
              onClick={handleRefresh}
              aria-label="Retry loading dashboard"
            >
              Retry
            </Button>
          }
        >
          {error}
        </Alert>
      </Box>
    );
  }
  
  return (
    <Box 
      sx={{ 
        flexGrow: 1, 
        p: isMobile ? 2 : 3,
        minHeight: '100vh',
        bgcolor: 'background.default'
      }}
      role="main"
      aria-label="Investment Dashboard"
      id="main-content"
    >
      {/* Header with enhanced controls */}
      <Box 
        display="flex" 
        justifyContent="space-between" 
        alignItems="center" 
        mb={3}
        flexWrap="wrap"
        gap={2}
      >
        <Box>
          <Typography 
            variant="h4" 
            component="h1" 
            gutterBottom
            sx={{ fontWeight: designTokens.typography.fontWeight.semibold }}
          >
            Investment Dashboard
          </Typography>
          <Typography variant="body1" color="textSecondary">
            <time dateTime={new Date().toISOString()}>
              {new Date().toLocaleDateString('en-US', { 
                weekday: 'long', 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
              })}
            </time>
          </Typography>
        </Box>
        
        <Box display="flex" gap={1} alignItems="center" flexWrap="wrap">
          {/* Auto-refresh toggle */}
          <FormControlLabel
            control={
              <Switch
                checked={autoRefresh}
                onChange={(e) => setAutoRefresh(e.target.checked)}
                aria-label="Toggle auto-refresh"
              />
            }
            label="Auto-refresh"
          />
          
          {/* Undo/Redo */}
          <Tooltip title="Undo (Ctrl+Z)">
            <span>
              <IconButton 
                onClick={handleUndo} 
                disabled={historyIndex <= 0}
                aria-label="Undo last action"
              >
                <Undo />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Redo (Ctrl+Y)">
            <span>
              <IconButton 
                onClick={handleRedo} 
                disabled={historyIndex >= history.length - 1}
                aria-label="Redo last action"
              >
                <Redo />
              </IconButton>
            </span>
          </Tooltip>
          
          {/* View mode toggle */}
          <Tooltip title="Change view">
            <IconButton 
              onClick={() => setViewMode(viewMode === 'grid' ? 'list' : 'grid')}
              aria-label={`Switch to ${viewMode === 'grid' ? 'list' : 'grid'} view`}
            >
              {viewMode === 'grid' ? <ViewList /> : <ViewModule />}
            </IconButton>
          </Tooltip>
          
          {/* Refresh button */}
          <Tooltip title="Refresh data (F5)">
            <span>
              <IconButton 
                onClick={handleRefresh} 
                disabled={refreshing}
                aria-label="Refresh dashboard data"
              >
                {refreshing ? <CircularProgress size={24} /> : <Refresh />}
              </IconButton>
            </span>
          </Tooltip>
          
          {/* Notifications */}
          <Tooltip title="View notifications">
            <IconButton aria-label="View notifications">
              <Badge badgeContent={3} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>
          
          {/* More options */}
          <Tooltip title="More options">
            <IconButton 
              onClick={(e) => setAnchorEl(e.currentTarget)}
              aria-label="More dashboard options"
              aria-controls="dashboard-menu"
              aria-haspopup="true"
            >
              <MoreVert />
            </IconButton>
          </Tooltip>
          
          <Menu
            id="dashboard-menu"
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={() => setAnchorEl(null)}
          >
            <MenuItem onClick={() => { setFullscreen(!fullscreen); setAnchorEl(null); }}>
              {fullscreen ? <FullscreenExit /> : <Fullscreen />}
              <Typography sx={{ ml: 1 }}>
                {fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
              </Typography>
            </MenuItem>
            <MenuItem onClick={() => { handleExport(); setAnchorEl(null); }}>
              <Download />
              <Typography sx={{ ml: 1 }}>Export Data</Typography>
            </MenuItem>
            <MenuItem onClick={() => setAnchorEl(null)}>
              <Share />
              <Typography sx={{ ml: 1 }}>Share Dashboard</Typography>
            </MenuItem>
            <Divider />
            <MenuItem onClick={() => setAnchorEl(null)}>
              <Settings />
              <Typography sx={{ ml: 1 }}>Settings</Typography>
            </MenuItem>
          </Menu>
        </Box>
      </Box>
      
      {/* Loading indicator */}
      {loading && (
        <Fade in={loading}>
          <LinearProgress 
            sx={{ mb: 2 }} 
            aria-label="Loading dashboard data"
          />
        </Fade>
      )}
      
      {/* Drag and drop grid */}
      <DragDropContext onDragEnd={handleDragEnd}>
        <Droppable droppableId="dashboard-widgets">
          {(provided) => (
            <Grid 
              container 
              spacing={3}
              {...provided.droppableProps}
              ref={provided.innerRef}
            >
              {widgetOrder.map((widgetId, index) => (
                <Draggable 
                  key={widgetId} 
                  draggableId={widgetId} 
                  index={index}
                  isDragDisabled={isMobile}
                >
                  {(provided, snapshot) => (
                    <Grid
                      item
                      xs={12}
                      md={
                        widgetId === 'metrics' ? 12 :
                        widgetId === 'market-overview' ? 8 :
                        widgetId === 'recommendations' ? 4 :
                        widgetId === 'portfolio-chart' ? 8 :
                        widgetId === 'portfolio-summary' ? 4 :
                        widgetId === 'news' ? 8 :
                        widgetId === 'cost-monitor' ? 4 :
                        12
                      }
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      style={{
                        ...provided.draggableProps.style,
                        opacity: snapshot.isDragging ? 0.8 : 1,
                      }}
                    >
                      {widgetId === 'metrics' && (
                        <Grid container spacing={2}>
                          {metrics.map((metric, idx) => (
                            <Grid item xs={12} sm={6} md={3} key={idx}>
                              <AccessibleMetricCard 
                                {...metric} 
                                loading={loading}
                              />
                            </Grid>
                          ))}
                        </Grid>
                      )}
                      
                      {widgetId === 'market-overview' && (
                        <Paper 
                          sx={{ p: 3, height: '100%' }}
                          role="region"
                          aria-label="Market Overview"
                        >
                          <Box 
                            display="flex" 
                            justifyContent="space-between" 
                            alignItems="center" 
                            mb={2}
                            {...provided.dragHandleProps}
                          >
                            <Box display="flex" alignItems="center" gap={1}>
                              {!isMobile && <DragIndicator sx={{ color: 'text.secondary' }} />}
                              <Typography variant="h6" component="h2">
                                Market Overview
                              </Typography>
                            </Box>
                            <Box display="flex" gap={1}>
                              {marketOverview?.indices?.map((index: any) => (
                                <Chip
                                  key={index.symbol}
                                  label={`${index.symbol}: ${index.change >= 0 ? '+' : ''}${index.change}%`}
                                  color={index.change >= 0 ? 'success' : 'error'}
                                  size="small"
                                />
                              ))}
                            </Box>
                          </Box>
                          <ErrorBoundary FallbackComponent={ErrorFallback}>
                            <Suspense fallback={<LoadingSkeleton height={300} />}>
                              <MarketHeatmap data={marketOverview?.heatmap} />
                            </Suspense>
                          </ErrorBoundary>
                        </Paper>
                      )}
                      
                      {/* Add other widgets similarly with enhanced features */}
                    </Grid>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </Grid>
          )}
        </Droppable>
      </DragDropContext>
      
      {/* Notification snackbar */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={() => setNotification({ ...notification, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setNotification({ ...notification, open: false })} 
          severity={notification.severity}
          variant="filled"
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default EnhancedDashboard;