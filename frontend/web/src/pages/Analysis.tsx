import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  Box,
  Tabs,
  Tab,
} from '@mui/material';
import {
  Assessment,
  Timeline,
  ShowChart,
  CandlestickChart,
  Analytics,
  Article,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchStockData,
  fetchStockChart,
  fetchOptionsChain,
  fetchSimilarStocks,
  selectStock,
} from '../store/slices/stockSlice';
import { addToWatchlist, removeFromWatchlist } from '../store/slices/portfolioSlice';
import { addNotification } from '../store/slices/appSlice';
import { AnalysisHeader, EmptyTickerView, LoadingErrorView } from '../components/analysis/AnalysisFilters';
import { ChartTabContent, TechnicalTabContent } from '../components/analysis/AnalysisCharts';
import {
  FundamentalTabContent,
  NewsTabContent,
  OptionsTabContent,
  SimilarStocksTabContent,
} from '../components/analysis/AnalysisTable';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`analysis-tabpanel-${index}`}
      aria-labelledby={`analysis-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const Analysis: React.FC = () => {
  const { ticker } = useParams<{ ticker: string }>();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();

  const {
    quote,
    chartData,
    technicalIndicators,
    fundamentalData,
    news,
    optionsChain,
    similarStocks,
    isLoading,
    error,
  } = useAppSelector((state) => state.stock);

  const { watchlist } = useAppSelector((state) => state.portfolio);

  const [tabValue, setTabValue] = useState(0);
  const [chartInterval, setChartInterval] = useState('1d');
  const [chartType, setChartType] = useState<'line' | 'candle'>('candle');
  const [searchTicker, setSearchTicker] = useState('');

  useEffect(() => {
    if (ticker) {
      dispatch(selectStock(ticker));
      dispatch(fetchStockData(ticker));
      dispatch(fetchStockChart({ ticker, interval: chartInterval }));
      dispatch(fetchSimilarStocks(ticker));
      dispatch(fetchOptionsChain(ticker));
    }
  }, [dispatch, ticker, chartInterval]);

  const handleRefresh = () => {
    if (ticker) {
      dispatch(fetchStockData(ticker));
      dispatch(fetchStockChart({ ticker, interval: chartInterval }));
      dispatch(
        addNotification({
          type: 'info',
          message: `${ticker} data refreshed`,
        })
      );
    }
  };

  const isInWatchlist = watchlist?.items?.some(
    (item) => item.symbol.toUpperCase() === ticker?.toUpperCase()
  ) ?? false;

  const handleWatchlistToggle = async () => {
    if (ticker) {
      if (isInWatchlist) {
        try {
          await dispatch(removeFromWatchlist(ticker)).unwrap();
          dispatch(
            addNotification({
              type: 'info',
              message: `${ticker} removed from watchlist`,
            })
          );
        } catch {
          dispatch(
            addNotification({
              type: 'error',
              message: `Failed to remove ${ticker} from watchlist`,
            })
          );
        }
      } else {
        try {
          await dispatch(addToWatchlist({ symbol: ticker })).unwrap();
          dispatch(
            addNotification({
              type: 'success',
              message: `${ticker} added to watchlist`,
            })
          );
        } catch {
          dispatch(
            addNotification({
              type: 'error',
              message: `Failed to add ${ticker} to watchlist`,
            })
          );
        }
      }
    }
  };

  const handleSearch = () => {
    if (searchTicker) {
      navigate(`/analysis/${searchTicker.toUpperCase()}`);
      setSearchTicker('');
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
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

  if (!ticker) {
    return (
      <EmptyTickerView
        searchTicker={searchTicker}
        onSearch={handleSearch}
        onSearchTickerChange={setSearchTicker}
      />
    );
  }

  const loadingErrorView = (
    <LoadingErrorView isLoading={isLoading} error={error} hasQuote={!!quote} />
  );
  if (isLoading || error || !quote) {
    return loadingErrorView;
  }

  const radarData = technicalIndicators?.signals ? [
    { signal: 'Technical', value: technicalIndicators.signals.trend === 'bullish' ? 80 : technicalIndicators.signals.trend === 'bearish' ? 20 : 50 },
    { signal: 'Momentum', value: technicalIndicators.signals.momentum === 'strong' ? 90 : technicalIndicators.signals.momentum === 'weak' ? 30 : 60 },
    { signal: 'Volatility', value: technicalIndicators.signals.volatility === 'high' ? 80 : technicalIndicators.signals.volatility === 'low' ? 20 : 50 },
    { signal: 'RSI', value: technicalIndicators.rsi ?? 50 },
    { signal: 'MACD', value: (technicalIndicators.macd?.histogram ?? 0) > 0 ? 70 : 30 },
    { signal: 'Volume', value: quote.volume > quote.avgVolume ? 80 : 40 },
  ] : [];

  return (
    <Container maxWidth="xl">
      <AnalysisHeader
        quote={quote}
        isInWatchlist={isInWatchlist}
        searchTicker={searchTicker}
        onWatchlistToggle={handleWatchlistToggle}
        onSearch={handleSearch}
        onRefresh={handleRefresh}
        onSearchTickerChange={setSearchTicker}
        formatCurrency={formatCurrency}
        formatPercent={formatPercent}
        formatLargeNumber={formatLargeNumber}
      />

      {/* Main Content */}
      <Paper>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)} aria-label="Stock analysis tabs">
          <Tab label="Chart" icon={<ShowChart />} id="analysis-tab-0" aria-controls="analysis-tabpanel-0" />
          <Tab label="Technical" icon={<Timeline />} id="analysis-tab-1" aria-controls="analysis-tabpanel-1" />
          <Tab label="Fundamental" icon={<Assessment />} id="analysis-tab-2" aria-controls="analysis-tabpanel-2" />
          <Tab label="News" icon={<Article />} id="analysis-tab-3" aria-controls="analysis-tabpanel-3" />
          <Tab label="Options" icon={<CandlestickChart />} id="analysis-tab-4" aria-controls="analysis-tabpanel-4" />
          <Tab label="Similar" icon={<Analytics />} id="analysis-tab-5" aria-controls="analysis-tabpanel-5" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <ChartTabContent
            chartData={chartData}
            chartType={chartType}
            chartInterval={chartInterval}
            onChartIntervalChange={setChartInterval}
            onChartTypeChange={setChartType}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <TechnicalTabContent
            technicalIndicators={technicalIndicators}
            radarData={radarData}
            formatCurrency={formatCurrency}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <FundamentalTabContent
            fundamentalData={fundamentalData}
            quote={quote}
            formatCurrency={formatCurrency}
            formatPercent={formatPercent}
            formatLargeNumber={formatLargeNumber}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={3}>
          <NewsTabContent news={news} />
        </TabPanel>

        <TabPanel value={tabValue} index={4}>
          <OptionsTabContent
            optionsChain={optionsChain}
            ticker={ticker}
            formatCurrency={formatCurrency}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={5}>
          <SimilarStocksTabContent
            similarStocks={similarStocks}
            onNavigate={(t) => navigate(`/analysis/${t}`)}
            formatPercent={formatPercent}
          />
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Analysis;
