/**
 * Market Overview page - Thin orchestrator that composes:
 *  - MarketSummary (index cards + market breadth)
 *  - MarketMovers / MarketNewsList (movers tables + news)
 *  - SectorPanel / HeatmapPanel / EconomicCalendarPanel (charts + calendar)
 */

import React, { useEffect, useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Tabs,
  Tab,
  LinearProgress,
  Button,
} from '@mui/material';
import { Refresh } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchMarketOverview,
  fetchSectorPerformance,
  fetchMarketNews,
  fetchHeatmapData,
  fetchEconomicCalendar,
} from '../store/slices/marketSlice';
import { addNotification } from '../store/slices/appSlice';
import MarketSummary from '../components/market/MarketSummary';
import { MarketMovers, MarketNewsList } from '../components/market/MarketTickers';
import { SectorPanel, HeatmapPanel, EconomicCalendarPanel } from '../components/market/MarketCharts';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const MarketOverview: React.FC = () => {
  const dispatch = useAppDispatch();
  const market = useAppSelector((state) => state.market);

  // Defensive defaults: async thunks may overwrite arrays with undefined
  const indices = market.indices ?? [];
  const topGainers = market.topGainers ?? [];
  const topLosers = market.topLosers ?? [];
  const mostActive = market.mostActive ?? [];
  const sectorPerformance = market.sectorPerformance ?? [];
  const marketNews = market.marketNews ?? [];
  const marketBreadth = market.marketBreadth ?? null;
  const heatmapData = market.heatmapData ?? [];
  const economicCalendar = market.economicCalendar ?? [];
  const isLoading = market.isLoading;

  const [tabValue, setTabValue] = useState(0);

  useEffect(() => {
    dispatch(fetchMarketOverview());
    dispatch(fetchSectorPerformance());
    dispatch(fetchMarketNews({ limit: 10 }));
    dispatch(fetchHeatmapData({}));
    dispatch(fetchEconomicCalendar());
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchMarketOverview());
    dispatch(fetchSectorPerformance());
    dispatch(fetchMarketNews({ limit: 10 }));
    dispatch(
      addNotification({
        type: 'info',
        message: 'Market data refreshed',
      })
    );
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatLargeNumber = (value: number) => {
    if (value >= 1e12) return `${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(0);
  };

  if (isLoading) {
    return <LinearProgress />;
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h4" fontWeight="bold">
          Market Overview
        </Typography>
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={handleRefresh}
        >
          Refresh
        </Button>
      </Box>

      <MarketSummary
        indices={indices}
        marketBreadth={marketBreadth}
        formatPercent={formatPercent}
        formatLargeNumber={formatLargeNumber}
      />

      {/* Main Content Tabs */}
      <Paper>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="Movers" />
          <Tab label="Sectors" />
          <Tab label="Heat Map" />
          <Tab label="News" />
          <Tab label="Economic Calendar" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <MarketMovers
            topGainers={topGainers}
            topLosers={topLosers}
            mostActive={mostActive}
            formatPercent={formatPercent}
            formatLargeNumber={formatLargeNumber}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <SectorPanel
            sectorPerformance={sectorPerformance}
            formatPercent={formatPercent}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <HeatmapPanel heatmapData={heatmapData} />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <MarketNewsList marketNews={marketNews} />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <EconomicCalendarPanel economicCalendar={economicCalendar} />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default MarketOverview;
