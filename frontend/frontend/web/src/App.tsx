import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, CssBaseline } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { Provider } from 'react-redux';

import { store } from './store';
import { theme } from './theme';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Recommendations from './pages/Recommendations';
import Analysis from './pages/Analysis';
import Portfolio from './pages/Portfolio';
import MarketOverview from './pages/MarketOverview';
import Settings from './pages/Settings';
import Login from './pages/Login';
import Watchlist from './pages/Watchlist';
import Alerts from './pages/Alerts';
import Reports from './pages/Reports';
import Help from './pages/Help';
import ErrorBoundary from './components/common/ErrorBoundary';
import { useAppDispatch, useAppSelector } from './hooks/redux';
import { useGlobalErrorHandler } from './hooks/useErrorHandler';
import { initializeApp } from './store/slices/appSlice';

function AppContent() {
  const dispatch = useAppDispatch();
  const { isAuthenticated, isInitialized } = useAppSelector((state) => state.app);
  const { handleError } = useGlobalErrorHandler();

  useEffect(() => {
    const initApp = async () => {
      try {
        await dispatch(initializeApp()).unwrap();
      } catch (error) {
        handleError(error, 'Failed to initialize application');
      }
    };
    
    initApp();
  }, [dispatch, handleError]);

  if (!isInitialized) {
    return <div>Loading...</div>;
  }

  return (
    <ErrorBoundary 
      showDetails={process.env.NODE_ENV === 'development'}
      onError={(error, errorInfo) => {
        handleError(error, 'Application error occurred');
        console.error('App Error Boundary:', error, errorInfo);
      }}
    >
      <Router>
        <Routes>
          {!isAuthenticated ? (
            <>
              <Route path="/login" element={<Login />} />
              <Route path="*" element={<Navigate to="/login" replace />} />
            </>
          ) : (
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route 
                path="dashboard" 
                element={
                  <ErrorBoundary>
                    <Dashboard />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="recommendations" 
                element={
                  <ErrorBoundary>
                    <Recommendations />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="analysis/:ticker?" 
                element={
                  <ErrorBoundary>
                    <Analysis />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="portfolio" 
                element={
                  <ErrorBoundary>
                    <Portfolio />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="market" 
                element={
                  <ErrorBoundary>
                    <MarketOverview />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="watchlist" 
                element={
                  <ErrorBoundary>
                    <Watchlist />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="alerts" 
                element={
                  <ErrorBoundary>
                    <Alerts />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="reports" 
                element={
                  <ErrorBoundary>
                    <Reports />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="settings" 
                element={
                  <ErrorBoundary>
                    <Settings />
                  </ErrorBoundary>
                } 
              />
              <Route 
                path="help" 
                element={
                  <ErrorBoundary>
                    <Help />
                  </ErrorBoundary>
                } 
              />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Route>
          )}
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

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