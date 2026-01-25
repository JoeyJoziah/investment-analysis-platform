import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Tabs,
  Tab,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Divider,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  List,
  ListItem,
  ListItemText,
  Alert,
  TextField,
  MenuItem,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  BookmarkBorder,
  Bookmark,
  Share,
  Refresh,
  Info,
  Assessment,
  Timeline,
  ShowChart,
  CandlestickChart,
  Analytics,
  Article,
  Speed,
  Security,
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
import StockChart from '../components/charts/StockChart';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from 'recharts';
import { format } from 'date-fns';

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
    selectedTicker,
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

  // Check if ticker is in watchlist
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
        } catch (error) {
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
        } catch (error) {
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
      <Container maxWidth="xl">
        <Box sx={{ textAlign: 'center', py: 8 }}>
          <Typography variant="h5" gutterBottom>
            Enter a ticker symbol to analyze
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mt: 3 }}>
            <TextField
              label="Ticker Symbol"
              value={searchTicker}
              onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            />
            <Button variant="contained" onClick={handleSearch}>
              Analyze
            </Button>
          </Box>
        </Box>
      </Container>
    );
  }

  if (isLoading) {
    return <LinearProgress />;
  }

  if (error) {
    return (
      <Container maxWidth="xl">
        <Alert severity="error" sx={{ mt: 3 }}>
          {error}
        </Alert>
      </Container>
    );
  }

  if (!quote) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h6" sx={{ mt: 3 }}>
          Loading stock data...
        </Typography>
      </Container>
    );
  }

  const radarData = technicalIndicators ? [
    { signal: 'Technical', value: technicalIndicators.signals.trend === 'bullish' ? 80 : technicalIndicators.signals.trend === 'bearish' ? 20 : 50 },
    { signal: 'Momentum', value: technicalIndicators.signals.momentum === 'strong' ? 90 : technicalIndicators.signals.momentum === 'weak' ? 30 : 60 },
    { signal: 'Volatility', value: technicalIndicators.signals.volatility === 'high' ? 80 : technicalIndicators.signals.volatility === 'low' ? 20 : 50 },
    { signal: 'RSI', value: technicalIndicators.rsi },
    { signal: 'MACD', value: technicalIndicators.macd.histogram > 0 ? 70 : 30 },
    { signal: 'Volume', value: quote.volume > quote.avgVolume ? 80 : 40 },
  ] : [];

  return (
    <Container maxWidth="xl">
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 2 }}>
          <Box>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Typography variant="h4" fontWeight="bold">
                {quote.ticker}
              </Typography>
              <IconButton onClick={handleWatchlistToggle}>
                {isInWatchlist ? (
                  <Bookmark color="primary" />
                ) : (
                  <BookmarkBorder />
                )}
              </IconButton>
              <IconButton>
                <Share />
              </IconButton>
            </Box>
            <Typography variant="h6" color="text.secondary">
              {quote.companyName}
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 2 }}>
            <TextField
              size="small"
              label="Compare with"
              value={searchTicker}
              onChange={(e) => setSearchTicker(e.target.value.toUpperCase())}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              sx={{ width: 150 }}
            />
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleRefresh}
            >
              Refresh
            </Button>
          </Box>
        </Box>

        {/* Price Info */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={8}>
            <Paper sx={{ p: 2 }}>
              <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 2, mb: 2 }}>
                <Typography variant="h3" fontWeight="bold">
                  {formatCurrency(quote.price)}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {quote.change >= 0 ? (
                    <TrendingUp color="success" />
                  ) : (
                    <TrendingDown color="error" />
                  )}
                  <Typography
                    variant="h5"
                    color={quote.change >= 0 ? 'success.main' : 'error.main'}
                  >
                    {quote.change >= 0 ? '+' : ''}{quote.change.toFixed(2)} ({formatPercent(quote.changePercent)})
                  </Typography>
                </Box>
              </Box>
              <Grid container spacing={2}>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Open
                  </Typography>
                  <Typography variant="body2">{formatCurrency(quote.open)}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Day Range
                  </Typography>
                  <Typography variant="body2">
                    {formatCurrency(quote.low)} - {formatCurrency(quote.high)}
                  </Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Volume
                  </Typography>
                  <Typography variant="body2">{formatLargeNumber(quote.volume)}</Typography>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Typography variant="caption" color="text.secondary">
                    Avg Volume
                  </Typography>
                  <Typography variant="body2">{formatLargeNumber(quote.avgVolume)}</Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
          <Grid item xs={12} md={4}>
            <Paper sx={{ p: 2, height: '100%' }}>
              <Typography variant="h6" gutterBottom>
                Key Statistics
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Market Cap
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {formatLargeNumber(quote.marketCap)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    P/E Ratio
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {quote.peRatio?.toFixed(2) || '-'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    52W Range
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    ${quote.week52Low.toFixed(2)} - ${quote.week52High.toFixed(2)}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Dividend Yield
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {quote.dividendYield ? `${(quote.dividendYield * 100).toFixed(2)}%` : '-'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    Beta
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {quote.beta?.toFixed(2) || '-'}
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    EPS
                  </Typography>
                  <Typography variant="body2" fontWeight="bold">
                    {quote.eps ? formatCurrency(quote.eps) : '-'}
                  </Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </Box>

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

        {/* Chart Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between' }}>
            <ToggleButtonGroup
              value={chartInterval}
              exclusive
              onChange={(_, newInterval) => newInterval && setChartInterval(newInterval)}
              size="small"
            >
              <ToggleButton value="1d">1D</ToggleButton>
              <ToggleButton value="1w">1W</ToggleButton>
              <ToggleButton value="1m">1M</ToggleButton>
              <ToggleButton value="3m">3M</ToggleButton>
              <ToggleButton value="6m">6M</ToggleButton>
              <ToggleButton value="1y">1Y</ToggleButton>
              <ToggleButton value="5y">5Y</ToggleButton>
              <ToggleButton value="max">MAX</ToggleButton>
            </ToggleButtonGroup>
            <ToggleButtonGroup
              value={chartType}
              exclusive
              onChange={(_, newType) => newType && setChartType(newType)}
              size="small"
            >
              <ToggleButton value="line">Line</ToggleButton>
              <ToggleButton value="candle">Candlestick</ToggleButton>
            </ToggleButtonGroup>
          </Box>
          <Box sx={{ height: 500 }}>
            {chartData && <StockChart data={chartData.data} chartType={chartType === 'candle' ? 'area' : chartType} />}
          </Box>
        </TabPanel>

        {/* Technical Tab */}
        <TabPanel value={tabValue} index={1}>
          {technicalIndicators && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Technical Indicators
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>RSI (14)</TableCell>
                        <TableCell align="right">
                          <Chip
                            label={technicalIndicators.rsi.toFixed(2)}
                            color={
                              technicalIndicators.rsi > 70
                                ? 'error'
                                : technicalIndicators.rsi < 30
                                ? 'success'
                                : 'default'
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>MACD</TableCell>
                        <TableCell align="right">
                          {technicalIndicators.macd.macd.toFixed(2)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>MACD Signal</TableCell>
                        <TableCell align="right">
                          {technicalIndicators.macd.signal.toFixed(2)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>MACD Histogram</TableCell>
                        <TableCell align="right">
                          <Typography
                            color={technicalIndicators.macd.histogram > 0 ? 'success.main' : 'error.main'}
                          >
                            {technicalIndicators.macd.histogram.toFixed(2)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>SMA 20</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.sma.sma20)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>SMA 50</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.sma.sma50)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>SMA 200</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.sma.sma200)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Bollinger Upper</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.bollingerBands.upper)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Bollinger Middle</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.bollingerBands.middle)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Bollinger Lower</TableCell>
                        <TableCell align="right">
                          {formatCurrency(technicalIndicators.bollingerBands.lower)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>ATR</TableCell>
                        <TableCell align="right">
                          {technicalIndicators.atr.toFixed(2)}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>ADX</TableCell>
                        <TableCell align="right">
                          {technicalIndicators.adx.toFixed(2)}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="h6" gutterBottom>
                  Analysis Signals
                </Typography>
                <Box sx={{ mb: 3 }}>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="signal" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar name="Signal Strength" dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                    </RadarChart>
                  </ResponsiveContainer>
                </Box>
                <Paper sx={{ p: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Trading Signals
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Trend</Typography>
                      <Chip
                        label={technicalIndicators.signals.trend.toUpperCase()}
                        color={
                          technicalIndicators.signals.trend === 'bullish'
                            ? 'success'
                            : technicalIndicators.signals.trend === 'bearish'
                            ? 'error'
                            : 'default'
                        }
                        size="small"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Momentum</Typography>
                      <Chip
                        label={technicalIndicators.signals.momentum.toUpperCase()}
                        color={
                          technicalIndicators.signals.momentum === 'strong'
                            ? 'success'
                            : technicalIndicators.signals.momentum === 'weak'
                            ? 'error'
                            : 'warning'
                        }
                        size="small"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography>Volatility</Typography>
                      <Chip
                        label={technicalIndicators.signals.volatility.toUpperCase()}
                        color={
                          technicalIndicators.signals.volatility === 'high'
                            ? 'error'
                            : technicalIndicators.signals.volatility === 'low'
                            ? 'success'
                            : 'warning'
                        }
                        size="small"
                      />
                    </Box>
                    <Divider />
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography fontWeight="bold">Recommendation</Typography>
                      <Chip
                        label={technicalIndicators.signals.recommendation.replace('_', ' ').toUpperCase()}
                        color={
                          technicalIndicators.signals.recommendation.includes('buy')
                            ? 'success'
                            : technicalIndicators.signals.recommendation.includes('sell')
                            ? 'error'
                            : 'warning'
                        }
                      />
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            </Grid>
          )}
        </TabPanel>

        {/* Fundamental Tab */}
        <TabPanel value={tabValue} index={2}>
          {fundamentalData && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Financial Performance
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>Revenue</TableCell>
                        <TableCell align="right">{formatLargeNumber(fundamentalData.revenue)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Revenue Growth</TableCell>
                        <TableCell align="right">
                          <Typography color={fundamentalData.revenueGrowth > 0 ? 'success.main' : 'error.main'}>
                            {formatPercent(fundamentalData.revenueGrowth)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Earnings</TableCell>
                        <TableCell align="right">{formatLargeNumber(fundamentalData.earnings)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Earnings Growth</TableCell>
                        <TableCell align="right">
                          <Typography color={fundamentalData.earningsGrowth > 0 ? 'success.main' : 'error.main'}>
                            {formatPercent(fundamentalData.earningsGrowth)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Profit Margin</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.profitMargin)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Operating Margin</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.operatingMargin)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Free Cash Flow</TableCell>
                        <TableCell align="right">{formatLargeNumber(fundamentalData.freeCashFlow)}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Valuation Metrics
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>P/E Ratio</TableCell>
                        <TableCell align="right">{quote.peRatio?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Forward P/E</TableCell>
                        <TableCell align="right">{fundamentalData.forwardPE?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>PEG Ratio</TableCell>
                        <TableCell align="right">{fundamentalData.pegRatio?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Price to Book</TableCell>
                        <TableCell align="right">{fundamentalData.priceToBook?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Price to Sales</TableCell>
                        <TableCell align="right">{fundamentalData.priceToSales?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Book Value</TableCell>
                        <TableCell align="right">{formatCurrency(fundamentalData.bookValue)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Dividend Rate</TableCell>
                        <TableCell align="right">{formatCurrency(fundamentalData.dividendRate)}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              
              <Grid item xs={12} md={4}>
                <Typography variant="h6" gutterBottom>
                  Financial Health
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>ROE</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.roe)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>ROA</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.roa)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Debt to Equity</TableCell>
                        <TableCell align="right">{fundamentalData.debtToEquity?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Current Ratio</TableCell>
                        <TableCell align="right">{fundamentalData.currentRatio?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Quick Ratio</TableCell>
                        <TableCell align="right">{fundamentalData.quickRatio?.toFixed(2) || '-'}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Insider Ownership</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.insiderOwnership)}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Institutional Own.</TableCell>
                        <TableCell align="right">{formatPercent(fundamentalData.institutionalOwnership)}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              
              {fundamentalData.analystRating && (
                <Grid item xs={12}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Analyst Ratings
                    </Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                          <Typography variant="h4" fontWeight="bold">
                            {fundamentalData.analystRating.consensus}
                          </Typography>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Consensus Rating
                            </Typography>
                            <Typography variant="h6">
                              Target: {formatCurrency(fundamentalData.analystRating.targetPrice)}
                            </Typography>
                          </Box>
                        </Box>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Box sx={{ display: 'flex', gap: 2 }}>
                          <Chip
                            label={`Strong Buy: ${fundamentalData.analystRating.strongBuy}`}
                            color="success"
                          />
                          <Chip
                            label={`Buy: ${fundamentalData.analystRating.buy}`}
                            color="success"
                            variant="outlined"
                          />
                          <Chip
                            label={`Hold: ${fundamentalData.analystRating.hold}`}
                            color="warning"
                          />
                          <Chip
                            label={`Sell: ${fundamentalData.analystRating.sell}`}
                            color="error"
                            variant="outlined"
                          />
                          <Chip
                            label={`Strong Sell: ${fundamentalData.analystRating.strongSell}`}
                            color="error"
                          />
                        </Box>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              )}
            </Grid>
          )}
        </TabPanel>

        {/* News Tab */}
        <TabPanel value={tabValue} index={3}>
          <Grid container spacing={2}>
            {news.map((article) => (
              <Grid item xs={12} key={article.id}>
                <Paper sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="h6" gutterBottom>
                        {article.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {article.summary}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                        <Chip
                          label={article.sentiment}
                          size="small"
                          color={
                            article.sentiment === 'positive'
                              ? 'success'
                              : article.sentiment === 'negative'
                              ? 'error'
                              : 'default'
                          }
                        />
                        <Typography variant="caption" color="text.secondary">
                          {article.source} • {format(new Date(article.publishedAt), 'MMM dd, yyyy h:mm a')}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                  <Box sx={{ mt: 1 }}>
                    <Button size="small" href={article.url} target="_blank">
                      Read More
                    </Button>
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Options Tab */}
        <TabPanel value={tabValue} index={4}>
          {optionsChain && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Options Chain - {ticker}
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Chip label="Calls" color="success" sx={{ mr: 1 }} />
                  <Chip label="Puts" color="error" />
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Call Options
                </Typography>
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Strike</TableCell>
                        <TableCell align="right">Bid</TableCell>
                        <TableCell align="right">Ask</TableCell>
                        <TableCell align="right">Volume</TableCell>
                        <TableCell align="right">OI</TableCell>
                        <TableCell align="right">IV</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {optionsChain.calls.slice(0, 10).map((option, index) => (
                        <TableRow key={index}>
                          <TableCell>{formatCurrency(option.strike)}</TableCell>
                          <TableCell align="right">{option.bid.toFixed(2)}</TableCell>
                          <TableCell align="right">{option.ask.toFixed(2)}</TableCell>
                          <TableCell align="right">{option.volume}</TableCell>
                          <TableCell align="right">{option.openInterest}</TableCell>
                          <TableCell align="right">{(option.impliedVolatility * 100).toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" gutterBottom>
                  Put Options
                </Typography>
                <TableContainer component={Paper}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Strike</TableCell>
                        <TableCell align="right">Bid</TableCell>
                        <TableCell align="right">Ask</TableCell>
                        <TableCell align="right">Volume</TableCell>
                        <TableCell align="right">OI</TableCell>
                        <TableCell align="right">IV</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {optionsChain.puts.slice(0, 10).map((option, index) => (
                        <TableRow key={index}>
                          <TableCell>{formatCurrency(option.strike)}</TableCell>
                          <TableCell align="right">{option.bid.toFixed(2)}</TableCell>
                          <TableCell align="right">{option.ask.toFixed(2)}</TableCell>
                          <TableCell align="right">{option.volume}</TableCell>
                          <TableCell align="right">{option.openInterest}</TableCell>
                          <TableCell align="right">{(option.impliedVolatility * 100).toFixed(1)}%</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Grid>
            </Grid>
          )}
        </TabPanel>

        {/* Similar Stocks Tab */}
        <TabPanel value={tabValue} index={5}>
          <Typography variant="h6" gutterBottom>
            Similar Stocks & Competitors
          </Typography>
          <Grid container spacing={2}>
            {similarStocks.map((stock) => (
              <Grid item xs={12} sm={6} md={4} key={stock.ticker}>
                <Card
                  sx={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/analysis/${stock.ticker}`)}
                >
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                      <Box>
                        <Typography variant="h6">{stock.ticker}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {stock.name}
                        </Typography>
                      </Box>
                      <Typography
                        variant="h6"
                        color={stock.changePercent >= 0 ? 'success.main' : 'error.main'}
                      >
                        {formatPercent(stock.changePercent)}
                      </Typography>
                    </Box>
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="caption" color="text.secondary">
                        Correlation: {(stock.correlation * 100).toFixed(0)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={stock.correlation * 100}
                        sx={{ mt: 1 }}
                      />
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default Analysis;