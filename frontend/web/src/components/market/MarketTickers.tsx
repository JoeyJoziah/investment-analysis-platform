/**
 * Market Tickers - Top Gainers, Top Losers, Most Active tables, and Market News
 * Renders the Movers and News tab panel content
 */

import React, { useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  Button,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  NewspaperOutlined,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { format } from 'date-fns';
import { EmptyStateBox } from './MarketSummary';
import type { MarketMover, MarketNews } from '../../store/slices/marketSlice';

interface StockTableProps {
  title: string;
  icon: React.ReactNode;
  stocks: MarketMover[];
  valueColumn: 'price' | 'volume';
  formatValue: (stock: MarketMover) => string;
  formatPercent: (value: number) => string;
  emptyMessage: string;
}

const StockTable: React.FC<StockTableProps> = ({
  title,
  icon,
  stocks,
  valueColumn,
  formatValue,
  formatPercent,
  emptyMessage,
}) => {
  const navigate = useNavigate();

  return (
    <>
      <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {icon}
        {title}
      </Typography>
      {stocks.length > 0 ? (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell align="right">{valueColumn === 'price' ? 'Price' : 'Volume'}</TableCell>
                <TableCell align="right">Change</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {stocks.slice(0, 10).map((stock) => (
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
                  <TableCell align="right">{formatValue(stock)}</TableCell>
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
      ) : (
        <Typography variant="body2" color="text.secondary" sx={{ py: 3, textAlign: 'center' }}>
          {emptyMessage}
        </Typography>
      )}
    </>
  );
};

export interface MarketMoversProps {
  topGainers: MarketMover[];
  topLosers: MarketMover[];
  mostActive: MarketMover[];
  formatPercent: (value: number) => string;
  formatLargeNumber: (value: number) => string;
}

const MarketMovers: React.FC<MarketMoversProps> = ({
  topGainers,
  topLosers,
  mostActive,
  formatPercent,
  formatLargeNumber,
}) => {
  if (topGainers.length === 0 && topLosers.length === 0 && mostActive.length === 0) {
    return (
      <EmptyStateBox
        icon={<TrendingUp sx={{ fontSize: 48 }} />}
        message="No market mover data available"
        submessage="Top gainers, losers, and most active stocks will appear here once connected to a data provider."
        minHeight={300}
      />
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={4}>
        <StockTable
          title="Top Gainers"
          icon={<TrendingUp color="success" />}
          stocks={topGainers}
          valueColumn="price"
          formatValue={(stock) => `$${stock.price.toFixed(2)}`}
          formatPercent={formatPercent}
          emptyMessage="No gainer data available."
        />
      </Grid>
      <Grid item xs={12} md={4}>
        <StockTable
          title="Top Losers"
          icon={<TrendingDown color="error" />}
          stocks={topLosers}
          valueColumn="price"
          formatValue={(stock) => `$${stock.price.toFixed(2)}`}
          formatPercent={formatPercent}
          emptyMessage="No loser data available."
        />
      </Grid>
      <Grid item xs={12} md={4}>
        <StockTable
          title="Most Active"
          icon={<ShowChart color="primary" />}
          stocks={mostActive}
          valueColumn="volume"
          formatValue={(stock) => formatLargeNumber(stock.volume)}
          formatPercent={formatPercent}
          emptyMessage="No active stock data available."
        />
      </Grid>
    </Grid>
  );
};

export interface MarketNewsListProps {
  marketNews: MarketNews[];
}

const MarketNewsList: React.FC<MarketNewsListProps> = ({ marketNews }) => {
  const navigate = useNavigate();
  const [newsLimit, setNewsLimit] = useState(5);

  return (
    <>
      <Typography variant="h6" gutterBottom>
        Market News
      </Typography>
      {marketNews.length > 0 ? (
        <>
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
                          {news.source} {'\u2022'} {format(new Date(news.publishedAt), 'MMM dd, yyyy h:mm a')}
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
        </>
      ) : (
        <EmptyStateBox
          icon={<NewspaperOutlined sx={{ fontSize: 48 }} />}
          message="No market news available"
          submessage="News articles will appear here once connected to a data provider."
          minHeight={300}
        />
      )}
    </>
  );
};

export { MarketMovers, MarketNewsList };
