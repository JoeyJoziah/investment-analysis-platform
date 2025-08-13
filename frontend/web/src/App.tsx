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
import { useAppDispatch, useAppSelector } from './hooks/redux';
import { initializeApp } from './store/slices/appSlice';

function AppContent() {
  const dispatch = useAppDispatch();
  const { isAuthenticated, isInitialized } = useAppSelector((state) => state.app);

  useEffect(() => {
    dispatch(initializeApp());
  }, [dispatch]);

  if (!isInitialized) {
    return <div>Loading...</div>;
  }

  return (
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
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="recommendations" element={<Recommendations />} />
            <Route path="analysis/:ticker?" element={<Analysis />} />
            <Route path="portfolio" element={<Portfolio />} />
            <Route path="market" element={<MarketOverview />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Route>
        )}
      </Routes>
    </Router>
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