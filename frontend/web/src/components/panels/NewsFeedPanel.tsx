import React, { memo, useCallback, useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Skeleton,
  Chip,
  useTheme,
  alpha,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  OpenInNew as OpenInNewIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  SentimentSatisfied as PositiveIcon,
  SentimentDissatisfied as NegativeIcon,
  SentimentNeutral as NeutralIcon,
} from '@mui/icons-material';
import { NewsArticle } from '../../types';
import { formatDistanceToNow, parseISO } from 'date-fns';

interface NewsFeedPanelProps {
  news: NewsArticle[];
  isLoading?: boolean;
  maxItems?: number;
}

/**
 * NewsFeedPanel - Market news and alerts feed
 *
 * Features:
 * - Sentiment indicators
 * - Related tickers
 * - Time ago formatting
 * - Expandable summaries
 *
 * WCAG 2.1 AA Compliant
 */
const NewsFeedPanel: React.FC<NewsFeedPanelProps> = ({
  news = [],
  isLoading = false,
  maxItems = 5,
}) => {
  const theme = useTheme();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const handleExpand = useCallback((id: string) => {
    setExpandedId((prev) => (prev === id ? null : id));
  }, []);

  const handleOpenArticle = useCallback((url: string, e: React.MouseEvent) => {
    e.stopPropagation();
    window.open(url, '_blank', 'noopener,noreferrer');
  }, []);

  // Get sentiment icon and color
  const getSentimentDisplay = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return {
          icon: <PositiveIcon sx={{ fontSize: 16 }} aria-hidden="true" />,
          color: theme.palette.success.main,
          label: 'Positive',
        };
      case 'negative':
        return {
          icon: <NegativeIcon sx={{ fontSize: 16 }} aria-hidden="true" />,
          color: theme.palette.error.main,
          label: 'Negative',
        };
      default:
        return {
          icon: <NeutralIcon sx={{ fontSize: 16 }} aria-hidden="true" />,
          color: theme.palette.grey[500],
          label: 'Neutral',
        };
    }
  };

  // Format time ago
  const formatTimeAgo = (dateString: string): string => {
    try {
      return formatDistanceToNow(parseISO(dateString), { addSuffix: true });
    } catch {
      return dateString;
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <Box>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 2,
          }}
        >
          <Skeleton variant="text" width={180} height={28} />
          <Skeleton variant="rectangular" width={80} height={32} sx={{ borderRadius: 1 }} />
        </Box>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {[...Array(maxItems)].map((_, i) => (
            <Box key={i}>
              <Skeleton variant="text" width="80%" height={24} />
              <Skeleton variant="text" width="60%" height={18} />
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 1 }} />
                <Skeleton variant="rectangular" width={60} height={24} sx={{ borderRadius: 1 }} />
              </Box>
            </Box>
          ))}
        </Box>
      </Box>
    );
  }

  // Empty state
  if (news.length === 0) {
    return (
      <Box>
        <Typography variant="h6" component="h2" gutterBottom>
          Market News
        </Typography>
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            py: 6,
            color: 'text.secondary',
          }}
        >
          <Typography variant="body1">No news available</Typography>
          <Typography variant="caption">
            Check back later for market updates
          </Typography>
        </Box>
      </Box>
    );
  }

  const displayedNews = news.slice(0, maxItems);

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 2,
        }}
      >
        <Typography variant="h6" component="h2">
          Market News & Insights
        </Typography>
        <Button size="small">View All</Button>
      </Box>

      {/* News list */}
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
        }}
        role="feed"
        aria-label="Market news feed"
      >
        {displayedNews.map((article) => {
          const sentiment = getSentimentDisplay(article.sentiment);
          const isExpanded = expandedId === article.id;

          return (
            <Box
              key={article.id}
              role="article"
              sx={{
                p: 2,
                borderRadius: 1,
                border: `1px solid ${theme.palette.divider}`,
                transition: 'all 0.2s ease-out',
                cursor: 'pointer',
                '&:hover': {
                  backgroundColor: alpha(theme.palette.primary.main, 0.02),
                  borderColor: alpha(theme.palette.primary.main, 0.2),
                },
                '&:focus-visible': {
                  outline: `2px solid ${theme.palette.primary.main}`,
                  outlineOffset: 2,
                },
              }}
              tabIndex={0}
              onClick={() => handleExpand(article.id)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  handleExpand(article.id);
                }
              }}
              aria-expanded={isExpanded}
            >
              {/* Title row */}
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'flex-start',
                  gap: 1,
                }}
              >
                <Typography
                  variant="subtitle2"
                  sx={{
                    fontWeight: 600,
                    lineHeight: 1.4,
                    display: '-webkit-box',
                    WebkitLineClamp: isExpanded ? 'none' : 2,
                    WebkitBoxOrient: 'vertical',
                    overflow: isExpanded ? 'visible' : 'hidden',
                  }}
                >
                  {article.title}
                </Typography>
                <IconButton
                  size="small"
                  aria-label={isExpanded ? 'Collapse' : 'Expand'}
                  sx={{ flexShrink: 0 }}
                >
                  {isExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                </IconButton>
              </Box>

              {/* Meta row */}
              <Box
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1.5,
                  mt: 1,
                  flexWrap: 'wrap',
                }}
              >
                {/* Source */}
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ fontWeight: 500 }}
                >
                  {article.source}
                </Typography>

                {/* Time ago */}
                <Typography variant="caption" color="text.secondary">
                  {formatTimeAgo(article.publishedAt)}
                </Typography>

                {/* Sentiment */}
                <Chip
                  icon={sentiment.icon}
                  label={sentiment.label}
                  size="small"
                  sx={{
                    height: 22,
                    fontSize: '0.65rem',
                    backgroundColor: alpha(sentiment.color, 0.1),
                    color: sentiment.color,
                    '& .MuiChip-icon': {
                      color: sentiment.color,
                    },
                  }}
                />
              </Box>

              {/* Related tickers */}
              {article.relatedTickers && article.relatedTickers.length > 0 && (
                <Box
                  sx={{
                    display: 'flex',
                    gap: 0.5,
                    mt: 1,
                    flexWrap: 'wrap',
                  }}
                >
                  {article.relatedTickers.slice(0, 5).map((ticker) => (
                    <Chip
                      key={ticker}
                      label={ticker}
                      size="small"
                      variant="outlined"
                      sx={{
                        height: 20,
                        fontSize: '0.65rem',
                        fontWeight: 600,
                        fontFamily: '"SF Mono", Monaco, monospace',
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        // Navigate to stock page
                        window.location.href = `/stocks/${ticker}`;
                      }}
                      aria-label={`View ${ticker} stock`}
                    />
                  ))}
                </Box>
              )}

              {/* Expanded content */}
              <Collapse in={isExpanded}>
                <Box sx={{ mt: 2 }}>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ lineHeight: 1.6 }}
                  >
                    {article.summary}
                  </Typography>

                  <Button
                    size="small"
                    endIcon={<OpenInNewIcon sx={{ fontSize: 14 }} />}
                    onClick={(e) => handleOpenArticle(article.url, e)}
                    sx={{ mt: 1.5 }}
                    aria-label={`Read full article about ${article.title} (opens in new tab)`}
                  >
                    Read Full Article
                  </Button>
                </Box>
              </Collapse>
            </Box>
          );
        })}
      </Box>

      {/* More news indicator */}
      {news.length > maxItems && (
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{
            display: 'block',
            textAlign: 'center',
            mt: 2,
          }}
        >
          +{news.length - maxItems} more articles available
        </Typography>
      )}
    </Box>
  );
};

export default memo(NewsFeedPanel);
