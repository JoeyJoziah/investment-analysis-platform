import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Grid,
  Paper,
  Typography,
  Box,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  Chip,
  LinearProgress,
  IconButton,
  Button,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Assessment,
  Timeline,
  CalendarToday,
  Info,
  Refresh,
  ArrowUpward,
  ArrowDownward,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../hooks/redux';
import {
  fetchMarketOverview,
  fetchSectorPerformance,
  fetchMarketNews,
  fetchHeatmapData,
  fetchEconomicCalendar,
} from '../store/slices/marketSlice';
import { addNotification } from '../store/slices/appSlice';
import MarketHeatmap from '../components/charts/MarketHeatmap';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
  Treemap,
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
    <div hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const MarketOverview: React.FC = () => {
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const {
    indices,
    topGainers,
    topLosers,
    mostActive,
    sectorPerformance,
    marketNews,
    marketBreadth,
    heatmapData,
    economicCalendar,
    isLoading,
  } = useAppSelector((state) => state.market);
  
  const [tabValue, setTabValue] = useState(0);
  const [newsLimit, setNewsLimit] = useState(5);

  useEffect(() => {
    dispatch(fetchMarketOverview());
    dispatch(fetchSectorPerformance());
    dispatch(fetchMarketNews({ limit: 10 }));
    dispatch(fetchHeatmapData());
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

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
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

  const COLORS = ['#00C49F', '#0088FE', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

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

      {/* Market Indices */}
      <Grid container spacing={2} sx={{ mb: 3 }}>
        {indices.map((index) => (
          <Grid item xs={12} sm={6} md={3} key={index.symbol}>
            <Card>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  {index.name}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                  <Typography variant="h5" fontWeight="bold">
                    {index.value.toLocaleString()}
                  </Typography>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    {index.change >= 0 ? (
                      <ArrowUpward sx={{ fontSize: 16, color: 'success.main' }} />
                    ) : (
                      <ArrowDownward sx={{ fontSize: 16, color: 'error.main' }} />
                    )}
                    <Typography
                      variant="body2"
                      color={index.change >= 0 ? 'success.main' : 'error.main'}
                    >
                      {index.change.toFixed(2)} ({formatPercent(index.changePercent)})
                    </Typography>
                  </Box>
                </Box>
                <Box sx={{ mt: 1, display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="caption" color="text.secondary">
                    Vol: {formatLargeNumber(index.volume)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {index.high.toFixed(2)} / {index.low.toFixed(2)}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Market Breadth */}
      {marketBreadth && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Market Breadth
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="success.main">
                    {marketBreadth.advancers}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Advancers
                  </Typography>
                </Box>
                <Typography variant="h6" color="text.secondary">
                  vs
                </Typography>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="error.main">
                    {marketBreadth.decliners}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Decliners
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="text.secondary">
                    {marketBreadth.unchanged}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Unchanged
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', justifyContent: 'space-around' }}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6">
                    {marketBreadth.advanceDeclineRatio.toFixed(2)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    A/D Ratio
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="primary.main">
                    {marketBreadth.newHighs}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    New Highs
                  </Typography>
                </Box>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h6" color="warning.main">
                    {marketBreadth.newLows}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    New Lows
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      )}

      {/* Main Content Tabs */}
      <Paper>
        <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
          <Tab label="Movers" />
          <Tab label="Sectors" />
          <Tab label="Heat Map" />
          <Tab label="News" />
          <Tab label="Economic Calendar" />
        </Tabs>

        {/* Movers Tab */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Top Gainers */}
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingUp color="success" />
                Top Gainers
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Change</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {topGainers.slice(0, 10).map((stock) => (
                      <TableRow
                        key={stock.ticker}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => navigate(`/analysis/${stock.ticker}`)}
                      >
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {stock.ticker}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {stock.companyName.length > 20
                              ? stock.companyName.substring(0, 20) + '...'
                              : stock.companyName}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">${stock.price.toFixed(2)}</TableCell>
                        <TableCell align="right">
                          <Typography color="success.main">
                            {formatPercent(stock.changePercent)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>

            {/* Top Losers */}
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <TrendingDown color="error" />
                Top Losers
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Change</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {topLosers.slice(0, 10).map((stock) => (
                      <TableRow
                        key={stock.ticker}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => navigate(`/analysis/${stock.ticker}`)}
                      >
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {stock.ticker}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {stock.companyName.length > 20
                              ? stock.companyName.substring(0, 20) + '...'
                              : stock.companyName}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">${stock.price.toFixed(2)}</TableCell>
                        <TableCell align="right">
                          <Typography color="error.main">
                            {formatPercent(stock.changePercent)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>

            {/* Most Active */}
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <ShowChart color="primary" />
                Most Active
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Volume</TableCell>
                      <TableCell align="right">Change</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {mostActive.slice(0, 10).map((stock) => (
                      <TableRow
                        key={stock.ticker}
                        hover
                        sx={{ cursor: 'pointer' }}
                        onClick={() => navigate(`/analysis/${stock.ticker}`)}
                      >
                        <TableCell>
                          <Typography variant="body2" fontWeight="bold">
                            {stock.ticker}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {stock.companyName.length > 20
                              ? stock.companyName.substring(0, 20) + '...'
                              : stock.companyName}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">{formatLargeNumber(stock.volume)}</TableCell>
                        <TableCell align="right">
                          <Typography color={stock.changePercent >= 0 ? 'success.main' : 'error.main'}>
                            {formatPercent(stock.changePercent)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Sectors Tab */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={8}>
              <Typography variant="h6" gutterBottom>
                Sector Performance
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={sectorPerformance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="sector" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="changePercent" fill={(entry: any) => entry.changePercent >= 0 ? '#00C49F' : '#FF8042'} />
                </BarChart>
              </ResponsiveContainer>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="h6" gutterBottom>
                Sector Details
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Sector</TableCell>
                      <TableCell align="right">Change</TableCell>
                      <TableCell align="center">Leaders</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {sectorPerformance.map((sector) => (
                      <TableRow key={sector.sector}>
                        <TableCell>{sector.sector}</TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            color={sector.changePercent >= 0 ? 'success.main' : 'error.main'}
                          >
                            {formatPercent(sector.changePercent)}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Box>
                            <Typography variant="caption">
                              {sector.topStock.ticker}
                            </Typography>
                            <Typography
                              variant="caption"
                              display="block"
                              color={sector.topStock.changePercent >= 0 ? 'success.main' : 'error.main'}
                            >
                              {formatPercent(sector.topStock.changePercent)}
                            </Typography>
                          </Box>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Heat Map Tab */}
        <TabPanel value={tabValue} index={2}>
          <Typography variant="h6" gutterBottom>
            Market Heat Map
          </Typography>
          <Box sx={{ height: 600 }}>
            <MarketHeatmap data={heatmapData} />
          </Box>
        </TabPanel>

        {/* News Tab */}
        <TabPanel value={tabValue} index={3}>
          <Typography variant="h6" gutterBottom>
            Market News
          </Typography>
          <Grid container spacing={2}>
            {marketNews.slice(0, newsLimit).map((news) => (
              <Grid item xs={12} key={news.id}>
                <Paper sx={{ p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                        {news.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {news.summary}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                        <Chip
                          label={news.sentiment}
                          size="small"
                          color={
                            news.sentiment === 'positive'
                              ? 'success'
                              : news.sentiment === 'negative'
                              ? 'error'
                              : 'default'
                          }
                        />
                        <Typography variant="caption" color="text.secondary">
                          {news.source} • {format(new Date(news.publishedAt), 'MMM dd, yyyy h:mm a')}
                        </Typography>
                        {news.relatedTickers.length > 0 && (
                          <>
                            <Divider orientation="vertical" flexItem />
                            {news.relatedTickers.map((ticker) => (
                              <Chip
                                key={ticker}
                                label={ticker}
                                size="small"
                                variant="outlined"
                                onClick={() => navigate(`/analysis/${ticker}`)}
                              />
                            ))}
                          </>
                        )}
                      </Box>
                    </Box>
                    {news.image && (
                      <Box
                        component="img"
                        src={news.image}
                        alt={news.title}
                        sx={{ width: 120, height: 80, objectFit: 'cover', ml: 2, borderRadius: 1 }}
                      />
                    )}
                  </Box>
                  <Box sx={{ mt: 1 }}>
                    <Button size="small" href={news.url} target="_blank">
                      Read More
                    </Button>
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>
          {newsLimit < marketNews.length && (
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Button onClick={() => setNewsLimit(newsLimit + 5)}>
                Load More News
              </Button>
            </Box>
          )}
        </TabPanel>

        {/* Economic Calendar Tab */}
        <TabPanel value={tabValue} index={4}>
          <Typography variant="h6" gutterBottom>
            Economic Calendar
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Date/Time</TableCell>
                  <TableCell>Event</TableCell>
                  <TableCell align="center">Importance</TableCell>
                  <TableCell align="right">Actual</TableCell>
                  <TableCell align="right">Forecast</TableCell>
                  <TableCell align="right">Previous</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {economicCalendar.map((event, index) => (
                  <TableRow key={index}>
                    <TableCell>
                      <Typography variant="body2">
                        {format(new Date(event.date), 'MMM dd')}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {event.time}
                      </Typography>
                    </TableCell>
                    <TableCell>{event.event}</TableCell>
                    <TableCell align="center">
                      <Chip
                        label={event.importance}
                        size="small"
                        color={
                          event.importance === 'high'
                            ? 'error'
                            : event.importance === 'medium'
                            ? 'warning'
                            : 'default'
                        }
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Typography
                        variant="body2"
                        fontWeight={event.actual ? 'bold' : 'normal'}
                        color={
                          event.actual && event.forecast
                            ? event.actual > event.forecast
                              ? 'success.main'
                              : event.actual < event.forecast
                              ? 'error.main'
                              : 'text.primary'
                            : 'text.primary'
                        }
                      >
                        {event.actual || '-'}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">{event.forecast || '-'}</TableCell>
                    <TableCell align="right">{event.previous || '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default MarketOverview;