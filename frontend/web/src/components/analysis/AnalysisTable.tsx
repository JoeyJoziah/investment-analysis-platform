import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent,
  LinearProgress,
  Button,
} from '@mui/material';
import { format } from 'date-fns';
import type { FundamentalData, StockQuote, StockNews, OptionsChain } from '../../store/slices/stockSlice';

export interface FundamentalTabProps {
  fundamentalData: FundamentalData | null;
  quote: StockQuote;
  formatCurrency: (value: number) => string;
  formatPercent: (value: number) => string;
  formatLargeNumber: (value: number) => string;
}

export const FundamentalTabContent: React.FC<FundamentalTabProps> = ({
  fundamentalData,
  quote,
  formatCurrency,
  formatPercent,
  formatLargeNumber,
}) => {
  if (!fundamentalData) return null;

  return (
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
  );
};

export interface NewsTabProps {
  news: StockNews[];
}

export const NewsTabContent: React.FC<NewsTabProps> = ({ news }) => {
  return (
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
                    {article.source} {String.fromCharCode(8226)} {format(new Date(article.publishedAt), 'MMM dd, yyyy h:mm a')}
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
  );
};

export interface OptionsTabProps {
  optionsChain: OptionsChain | null;
  ticker: string;
  formatCurrency: (value: number) => string;
}

export const OptionsTabContent: React.FC<OptionsTabProps> = ({
  optionsChain,
  ticker,
  formatCurrency,
}) => {
  if (!optionsChain) return null;

  return (
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
  );
};

export interface SimilarStocksTabProps {
  similarStocks: Array<{
    ticker: string;
    name: string;
    correlation: number;
    changePercent: number;
  }>;
  onNavigate: (ticker: string) => void;
  formatPercent: (value: number) => string;
}

export const SimilarStocksTabContent: React.FC<SimilarStocksTabProps> = ({
  similarStocks,
  onNavigate,
  formatPercent,
}) => {
  return (
    <>
      <Typography variant="h6" gutterBottom>
        Similar Stocks & Competitors
      </Typography>
      <Grid container spacing={2}>
        {similarStocks.map((stock) => (
          <Grid item xs={12} sm={6} md={4} key={stock.ticker}>
            <Card
              sx={{ cursor: 'pointer' }}
              onClick={() => onNavigate(stock.ticker)}
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
    </>
  );
};

export default {
  FundamentalTabContent,
  NewsTabContent,
  OptionsTabContent,
  SimilarStocksTabContent,
};
