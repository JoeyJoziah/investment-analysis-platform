import React, { useState, useMemo } from 'react';
import {
  Container,
  Typography,
  Box,
  Paper,
  Grid,
  TextField,
  InputAdornment,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Chip,
  Divider,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  Search,
  ExpandMore,
  PlayCircleOutline,
  AccountBalance,
  Recommend,
  BarChart,
  ShowChart,
  SmartToy,
  BookmarkBorder,
  NotificationsActive,
  Assessment,
  Keyboard,
  Email,
  GitHub,
  Info,
  LightbulbOutlined,
} from '@mui/icons-material';

interface FaqItem {
  question: string;
  answer: string;
  tags: string[];
}

interface GuideItem {
  title: string;
  description: string;
  icon: React.ReactNode;
  tags: string[];
}

interface GettingStartedItem {
  title: string;
  description: string;
  icon: React.ReactNode;
}

interface ShortcutItem {
  keys: string;
  description: string;
}

const GETTING_STARTED: GettingStartedItem[] = [
  {
    title: 'Platform Overview',
    description:
      'InvestAI Pro combines real-time market data, AI-powered analysis, and portfolio tracking to help you make informed investment decisions.',
    icon: <PlayCircleOutline fontSize="large" color="primary" />,
  },
  {
    title: 'Setting Up Your Portfolio',
    description:
      'Navigate to Portfolio in the sidebar, click "Add Position", enter a ticker symbol, quantity, and purchase price to start tracking your holdings.',
    icon: <AccountBalance fontSize="large" color="primary" />,
  },
  {
    title: 'Understanding Recommendations',
    description:
      'AI recommendations score stocks from 0-100 based on technical indicators, fundamental data, and sentiment analysis. Scores above 70 indicate strong buy signals.',
    icon: <Recommend fontSize="large" color="primary" />,
  },
  {
    title: 'Using Market Analysis Tools',
    description:
      'Access technical charts, fundamental data, and AI agent analysis from the Analysis page. Select any ticker to view comprehensive insights.',
    icon: <BarChart fontSize="large" color="primary" />,
  },
];

const FAQ_ITEMS: FaqItem[] = [
  {
    question: 'How do I add stocks to my watchlist?',
    answer:
      'Go to the Watchlist page from the sidebar navigation. Click the "Add Stock" button, search for a ticker symbol or company name, and click "Add" to include it in your watchlist. You can also add stocks directly from the search bar (Ctrl+K) by clicking the bookmark icon next to any result.',
    tags: ['watchlist', 'stocks', 'add'],
  },
  {
    question: 'What do the recommendation scores mean?',
    answer:
      'Recommendation scores range from 0 to 100 and represent the AI confidence level for a given stock. Scores 80-100 indicate a Strong Buy, 60-79 a Buy, 40-59 a Hold, 20-39 a Sell, and 0-19 a Strong Sell. Scores are calculated using a weighted combination of technical analysis (40%), fundamental analysis (30%), and sentiment analysis (30%).',
    tags: ['recommendations', 'scores', 'analysis'],
  },
  {
    question: 'How is risk calculated?',
    answer:
      'Risk is assessed using multiple factors: historical volatility (standard deviation of returns), beta relative to the S&P 500, maximum drawdown over the past year, sector concentration in your portfolio, and the VaR (Value at Risk) at a 95% confidence interval. The final risk score is categorized as Low, Medium, High, or Very High.',
    tags: ['risk', 'portfolio', 'analysis'],
  },
  {
    question: 'How do I set up price alerts?',
    answer:
      'Navigate to Settings > Notifications, or go to the Alerts page directly. You can set price-based alerts (e.g., "Notify when AAPL goes above $200") or percentage-based alerts (e.g., "Notify when TSLA drops by 5%"). Alerts can be delivered as in-app notifications, email alerts, or push notifications depending on your notification preferences.',
    tags: ['alerts', 'notifications', 'price'],
  },
  {
    question: 'What analysis types are available?',
    answer:
      'The platform supports four analysis types: (1) Technical Analysis -- moving averages, RSI, MACD, Bollinger Bands, and candlestick patterns. (2) Fundamental Analysis -- P/E ratio, EPS, revenue growth, debt-to-equity, and free cash flow. (3) Sentiment Analysis -- news sentiment, social media trends, and analyst ratings. (4) AI Agent Analysis -- autonomous AI agents that perform deep-dive multi-factor analysis.',
    tags: ['analysis', 'technical', 'fundamental', 'sentiment'],
  },
  {
    question: 'How does the AI agent analysis work?',
    answer:
      'AI agents are autonomous analysis workers that examine stocks from multiple perspectives. When you request an agent analysis, specialized agents for technical, fundamental, and sentiment analysis run concurrently. Each agent produces an independent assessment, and a coordinator agent synthesizes the results into a unified recommendation with confidence scores and reasoning.',
    tags: ['ai', 'agents', 'analysis'],
  },
  {
    question: 'How do I export my portfolio data?',
    answer:
      'Go to the Reports page and select "Export Portfolio". You can export in CSV, PDF, or JSON formats. The export includes all positions, realized and unrealized gains/losses, transaction history, and performance metrics. Scheduled exports can be configured under Settings > Data & Privacy.',
    tags: ['export', 'portfolio', 'reports', 'data'],
  },
  {
    question: 'What is the Investment Thesis feature?',
    answer:
      'The Investment Thesis feature lets you document your reasoning for each position. When adding or editing a position, you can write a thesis explaining why you bought the stock, your price target, time horizon, and key catalysts. The platform tracks your thesis against actual performance to help you refine your investment process over time.',
    tags: ['thesis', 'portfolio', 'investment'],
  },
  {
    question: 'How do I connect external data providers?',
    answer:
      'Navigate to Settings > API Keys. The platform supports Alpha Vantage, Finnhub, Polygon.io, and News API. Enter your API key for each provider you want to use. Free tiers are available for all providers. The platform will automatically use available providers based on rate limits and data freshness.',
    tags: ['api', 'data', 'settings', 'providers'],
  },
  {
    question: 'Can I share my portfolio with others?',
    answer:
      'Yes. From the Portfolio page, click the share icon to generate a read-only link. You can choose to share your full portfolio, specific positions, or just performance metrics. Shared links can be set to expire after a chosen period. Navigate to Settings > Data & Privacy to manage your sharing preferences.',
    tags: ['share', 'portfolio', 'privacy'],
  },
  {
    question: 'How often is market data refreshed?',
    answer:
      'Real-time data refreshes every 15 seconds during market hours by default. You can adjust the refresh interval in Settings > Appearance (30 seconds to 5 minutes). WebSocket connections provide instant updates for watchlist items when available. Historical data is updated daily after market close.',
    tags: ['data', 'refresh', 'real-time', 'market'],
  },
  {
    question: 'What do the WebSocket connection indicators mean?',
    answer:
      'The green dot in the toolbar indicates an active real-time data connection. A yellow dot means the connection is reconnecting. A red dot means the connection is lost and data may be stale. The platform will automatically attempt to reconnect. You can click the indicator to view connection details.',
    tags: ['websocket', 'connection', 'real-time'],
  },
];

const FEATURE_GUIDES: GuideItem[] = [
  {
    title: 'Portfolio Management',
    description:
      'Track holdings, monitor performance, view allocation breakdowns, and manage your investment positions with real-time P&L updates.',
    icon: <AccountBalance color="primary" />,
    tags: ['portfolio', 'holdings', 'positions'],
  },
  {
    title: 'Stock Analysis',
    description:
      'Access both technical and fundamental analysis tools including interactive charts, financial ratios, and historical comparisons.',
    icon: <ShowChart color="primary" />,
    tags: ['analysis', 'technical', 'fundamental', 'charts'],
  },
  {
    title: 'AI-Powered Recommendations',
    description:
      'Receive AI-generated stock recommendations based on multi-factor analysis, with confidence scores and detailed reasoning.',
    icon: <SmartToy color="primary" />,
    tags: ['ai', 'recommendations', 'analysis'],
  },
  {
    title: 'Watchlist Management',
    description:
      'Create and organize watchlists to monitor stocks of interest. Track price movements, volume changes, and key metrics at a glance.',
    icon: <BookmarkBorder color="primary" />,
    tags: ['watchlist', 'stocks', 'monitoring'],
  },
  {
    title: 'Alerts System',
    description:
      'Configure price alerts, percentage change notifications, and volume spike warnings. Receive alerts via in-app notifications or email.',
    icon: <NotificationsActive color="primary" />,
    tags: ['alerts', 'notifications', 'price'],
  },
  {
    title: 'Reports & Export',
    description:
      'Generate performance reports, tax summaries, and transaction logs. Export data in CSV, PDF, or JSON formats for external use.',
    icon: <Assessment color="primary" />,
    tags: ['reports', 'export', 'data', 'performance'],
  },
];

const KEYBOARD_SHORTCUTS: ShortcutItem[] = [
  { keys: 'Ctrl + K', description: 'Open global search' },
  { keys: 'Ctrl + /', description: 'Focus search on current page' },
  { keys: 'Ctrl + B', description: 'Toggle sidebar' },
  { keys: 'Ctrl + D', description: 'Go to Dashboard' },
  { keys: 'Ctrl + P', description: 'Go to Portfolio' },
  { keys: 'Ctrl + W', description: 'Go to Watchlist' },
  { keys: 'Ctrl + ,', description: 'Open Settings' },
  { keys: 'Escape', description: 'Close modal / Cancel action' },
  { keys: 'Ctrl + S', description: 'Save current form' },
  { keys: '?', description: 'Show keyboard shortcuts (from any page)' },
];

const APP_VERSION = '1.0.0';

const Help: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedFaq, setExpandedFaq] = useState<string | false>(false);

  const normalizedQuery = searchQuery.toLowerCase().trim();

  const filteredFaq = useMemo(
    () =>
      FAQ_ITEMS.filter(
        (item) =>
          !normalizedQuery ||
          item.question.toLowerCase().includes(normalizedQuery) ||
          item.answer.toLowerCase().includes(normalizedQuery) ||
          item.tags.some((tag) => tag.includes(normalizedQuery))
      ),
    [normalizedQuery]
  );

  const filteredGuides = useMemo(
    () =>
      FEATURE_GUIDES.filter(
        (item) =>
          !normalizedQuery ||
          item.title.toLowerCase().includes(normalizedQuery) ||
          item.description.toLowerCase().includes(normalizedQuery) ||
          item.tags.some((tag) => tag.includes(normalizedQuery))
      ),
    [normalizedQuery]
  );

  const filteredGettingStarted = useMemo(
    () =>
      GETTING_STARTED.filter(
        (item) =>
          !normalizedQuery ||
          item.title.toLowerCase().includes(normalizedQuery) ||
          item.description.toLowerCase().includes(normalizedQuery)
      ),
    [normalizedQuery]
  );

  const handleFaqChange =
    (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
      setExpandedFaq(isExpanded ? panel : false);
    };

  const hasResults =
    filteredFaq.length > 0 ||
    filteredGuides.length > 0 ||
    filteredGettingStarted.length > 0;

  return (
    <Container maxWidth="lg">
      <Box sx={{ mb: 3 }}>
        <Typography variant="h4" fontWeight="bold">
          Help & Support
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Find answers, learn features, and get support
        </Typography>
      </Box>

      {/* Search Bar */}
      <Paper sx={{ p: 2, mb: 4 }}>
        <TextField
          fullWidth
          placeholder="Search help articles, FAQs, and guides..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search />
              </InputAdornment>
            ),
          }}
          aria-label="Search help content"
        />
        {normalizedQuery && !hasResults && (
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{ mt: 2, textAlign: 'center' }}
          >
            No results found for "{searchQuery}". Try a different search term.
          </Typography>
        )}
      </Paper>

      {/* Getting Started */}
      {filteredGettingStarted.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <LightbulbOutlined color="primary" />
            <Typography variant="h5" fontWeight="bold">
              Getting Started
            </Typography>
          </Box>
          <Grid container spacing={2}>
            {filteredGettingStarted.map((item) => (
              <Grid item xs={12} sm={6} key={item.title}>
                <Card
                  variant="outlined"
                  sx={{
                    height: '100%',
                    transition: 'box-shadow 0.2s',
                    '&:hover': { boxShadow: 3 },
                  }}
                >
                  <CardContent>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1.5,
                        mb: 1.5,
                      }}
                    >
                      {item.icon}
                      <Typography variant="h6">{item.title}</Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {item.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* FAQ Section */}
      {filteredFaq.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" fontWeight="bold" sx={{ mb: 2 }}>
            Frequently Asked Questions
          </Typography>
          {filteredFaq.map((item, index) => (
            <Accordion
              key={index}
              expanded={expandedFaq === `faq-${index}`}
              onChange={handleFaqChange(`faq-${index}`)}
              disableGutters
            >
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography fontWeight="medium">{item.question}</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
                  {item.answer}
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                  {item.tags.map((tag) => (
                    <Chip
                      key={tag}
                      label={tag}
                      size="small"
                      variant="outlined"
                      onClick={() => setSearchQuery(tag)}
                      sx={{ cursor: 'pointer' }}
                    />
                  ))}
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
        </Box>
      )}

      {/* Feature Guides */}
      {filteredGuides.length > 0 && (
        <Box sx={{ mb: 4 }}>
          <Typography variant="h5" fontWeight="bold" sx={{ mb: 2 }}>
            Feature Guides
          </Typography>
          <Grid container spacing={2}>
            {filteredGuides.map((item) => (
              <Grid item xs={12} sm={6} md={4} key={item.title}>
                <Card
                  variant="outlined"
                  sx={{
                    height: '100%',
                    transition: 'box-shadow 0.2s',
                    '&:hover': { boxShadow: 3 },
                  }}
                >
                  <CardContent>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        mb: 1,
                      }}
                    >
                      {item.icon}
                      <Typography variant="subtitle1" fontWeight="bold">
                        {item.title}
                      </Typography>
                    </Box>
                    <Typography variant="body2" color="text.secondary">
                      {item.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      )}

      {/* Keyboard Shortcuts */}
      {!normalizedQuery && (
        <Box sx={{ mb: 4 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Keyboard color="primary" />
            <Typography variant="h5" fontWeight="bold">
              Keyboard Shortcuts
            </Typography>
          </Box>
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>Shortcut</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {KEYBOARD_SHORTCUTS.map((shortcut) => (
                  <TableRow key={shortcut.keys}>
                    <TableCell>
                      <Chip
                        label={shortcut.keys}
                        size="small"
                        sx={{
                          fontFamily: 'monospace',
                          fontWeight: 'bold',
                        }}
                      />
                    </TableCell>
                    <TableCell>{shortcut.description}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}

      {/* Contact Support */}
      {!normalizedQuery && (
        <Paper variant="outlined" sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" fontWeight="bold" sx={{ mb: 2 }}>
            Contact Support
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Email color="primary" />
                <Typography variant="subtitle1" fontWeight="bold">
                  Email Support
                </Typography>
              </Box>
              <Link href="mailto:support@investai.com" underline="hover">
                support@investai.com
              </Link>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <GitHub color="primary" />
                <Typography variant="subtitle1" fontWeight="bold">
                  GitHub Issues
                </Typography>
              </Box>
              <Link
                href="https://github.com/investai-pro/platform/issues"
                target="_blank"
                rel="noopener noreferrer"
                underline="hover"
              >
                Report a Bug or Request a Feature
              </Link>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Info color="primary" />
                <Typography variant="subtitle1" fontWeight="bold">
                  Version Info
                </Typography>
              </Box>
              <Typography variant="body2" color="text.secondary">
                InvestAI Pro v{APP_VERSION}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                React {React.version}
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Container>
  );
};

export default Help;
