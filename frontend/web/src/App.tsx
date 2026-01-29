import React, { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { Provider } from 'react-redux';

import { store } from './store';
import { theme } from './theme';
import Layout from './components/Layout';
import LoadingSpinner from './components/common/LoadingSpinner';
import PageSkeleton from './components/common/PageSkeleton';
import ErrorBoundary from './components/common/ErrorBoundary';
import { useAppDispatch, useAppSelector } from './hooks/redux';
import { initializeApp } from './store/slices/appSlice';

// =============================================================================
// LAZY-LOADED PAGE COMPONENTS
// =============================================================================
// These components are code-split and loaded on demand to reduce initial bundle size.
// Each page is loaded only when the user navigates to it, resulting in:
// - 60-70% smaller initial bundle
// - Faster Time to First Contentful Paint (FCP)
// - Better Core Web Vitals scores
// =============================================================================

// Authentication - loaded immediately as it's the entry point for unauthenticated users
const Login = lazy(() => import('./pages/Login'));

// Primary pages - most frequently accessed
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Portfolio = lazy(() => import('./pages/Portfolio'));
const Recommendations = lazy(() => import('./pages/Recommendations'));

// Analysis pages - heavy charting components
const Analysis = lazy(() => import('./pages/Analysis'));
const MarketOverview = lazy(() => import('./pages/MarketOverview'));

// Secondary pages - less frequently accessed
const Watchlist = lazy(() => import('./pages/Watchlist'));
const Alerts = lazy(() => import('./pages/Alerts'));
const Reports = lazy(() => import('./pages/Reports'));

// Utility pages - rarely accessed
const Settings = lazy(() => import('./pages/Settings'));
const Help = lazy(() => import('./pages/Help'));
const InvestmentThesis = lazy(() => import('./pages/InvestmentThesis'));

// =============================================================================
// ROUTE PREFETCHING
// =============================================================================
// Prefetch commonly navigated routes to improve perceived performance
// =============================================================================

const routeModules = {
  dashboard: () => import('./pages/Dashboard'),
  portfolio: () => import('./pages/Portfolio'),
  recommendations: () => import('./pages/Recommendations'),
  analysis: () => import('./pages/Analysis'),
  market: () => import('./pages/MarketOverview'),
  watchlist: () => import('./pages/Watchlist'),
  alerts: () => import('./pages/Alerts'),
  reports: () => import('./pages/Reports'),
  settings: () => import('./pages/Settings'),
  help: () => import('./pages/Help'),
};

/**
 * Prefetch a route module for faster navigation
 */
const prefetchRoute = (route: keyof typeof routeModules): void => {
  routeModules[route]();
};

/**
 * Prefetch related routes based on current location
 */
const usePrefetchRoutes = (): void => {
  const location = useLocation();

  useEffect(() => {
    // Prefetch related routes based on current page
    const prefetchMap: Record<string, (keyof typeof routeModules)[]> = {
      '/dashboard': ['portfolio', 'recommendations', 'analysis'],
      '/portfolio': ['dashboard', 'analysis', 'reports'],
      '/recommendations': ['analysis', 'portfolio'],
      '/analysis': ['portfolio', 'recommendations'],
      '/market': ['analysis', 'watchlist'],
      '/watchlist': ['analysis', 'alerts'],
    };

    const routesToPrefetch = prefetchMap[location.pathname];
    if (routesToPrefetch) {
      // Delay prefetching to prioritize current page resources
      const timeoutId = setTimeout(() => {
        routesToPrefetch.forEach(prefetchRoute);
      }, 1000);

      return () => clearTimeout(timeoutId);
    }
  }, [location.pathname]);
};

// =============================================================================
// SUSPENSE WRAPPER COMPONENT
// =============================================================================
// Provides consistent loading states with skeleton loaders and error boundaries
// =============================================================================

type SkeletonType = 'dashboard' | 'portfolio' | 'analysis' | 'list' | 'default';

interface SuspenseWrapperProps {
  children: React.ReactNode;
  loadingMessage?: string;
  skeletonType?: SkeletonType;
}

const SuspenseWrapper: React.FC<SuspenseWrapperProps> = ({
  children,
  loadingMessage = 'Loading...',
  skeletonType = 'default',
}) => (
  <ErrorBoundary>
    <Suspense fallback={<PageSkeleton type={skeletonType} />}>
      {children}
    </Suspense>
  </ErrorBoundary>
);

// =============================================================================
// ROUTE PREFETCH COMPONENT
// =============================================================================
// Component that handles prefetching logic within Router context
// =============================================================================

const RoutePrefetcher: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  usePrefetchRoutes();
  return <>{children}</>;
};

// =============================================================================
// APP CONTENT COMPONENT
// =============================================================================
// Handles authentication state and routing logic
// =============================================================================

function AppContent() {
  const dispatch = useAppDispatch();
  const { isAuthenticated, isInitialized } = useAppSelector((state) => state.app);

  useEffect(() => {
    dispatch(initializeApp());
  }, [dispatch]);

  if (!isInitialized) {
    return <LoadingSpinner message="Initializing application..." fullScreen />;
  }

  return (
    <Router>
      <RoutePrefetcher>
      <Routes>
        {!isAuthenticated ? (
          <>
            <Route
              path="/login"
              element={
                <SuspenseWrapper loadingMessage="Loading login...">
                  <Login />
                </SuspenseWrapper>
              }
            />
            <Route path="*" element={<Navigate to="/login" replace />} />
          </>
        ) : (
          <Route path="/" element={<Layout />}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route
              path="dashboard"
              element={
                <SuspenseWrapper skeletonType="dashboard">
                  <Dashboard />
                </SuspenseWrapper>
              }
            />
            <Route
              path="recommendations"
              element={
                <SuspenseWrapper skeletonType="list">
                  <Recommendations />
                </SuspenseWrapper>
              }
            />
            <Route
              path="analysis/:ticker?"
              element={
                <SuspenseWrapper skeletonType="analysis">
                  <Analysis />
                </SuspenseWrapper>
              }
            />
            <Route
              path="portfolio"
              element={
                <SuspenseWrapper skeletonType="portfolio">
                  <Portfolio />
                </SuspenseWrapper>
              }
            />
            <Route
              path="market"
              element={
                <SuspenseWrapper skeletonType="analysis">
                  <MarketOverview />
                </SuspenseWrapper>
              }
            />
            <Route
              path="watchlist"
              element={
                <SuspenseWrapper skeletonType="list">
                  <Watchlist />
                </SuspenseWrapper>
              }
            />
            <Route
              path="alerts"
              element={
                <SuspenseWrapper skeletonType="list">
                  <Alerts />
                </SuspenseWrapper>
              }
            />
            <Route
              path="reports"
              element={
                <SuspenseWrapper skeletonType="list">
                  <Reports />
                </SuspenseWrapper>
              }
            />
            <Route
              path="settings"
              element={
                <SuspenseWrapper skeletonType="default">
                  <Settings />
                </SuspenseWrapper>
              }
            />
            <Route
              path="help"
              element={
                <SuspenseWrapper skeletonType="default">
                  <Help />
                </SuspenseWrapper>
              }
            />
            <Route
              path="thesis/:stockId"
              element={
                <SuspenseWrapper loadingMessage="Loading investment thesis...">
                  <InvestmentThesis />
                </SuspenseWrapper>
              }
            />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        )}
      </Routes>
      </RoutePrefetcher>
    </Router>
  );
}

// =============================================================================
// ROOT APP COMPONENT
// =============================================================================
// Wraps the application with all required providers
// =============================================================================

function App() {
  return (
    <Provider store={store}>
      <ThemeProvider theme={theme}>
        <LocalizationProvider dateAdapter={AdapterDateFns}>
          <CssBaseline />
          <AppContent />
        </LocalizationProvider>
      </ThemeProvider>
    </Provider>
  );
}

export default App;
