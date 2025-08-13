import React from 'react';
import {
  Card,
  CardContent,
  CardMedia,
  CardActionArea,
  Box,
  Typography,
  Chip,
  Avatar,
  IconButton,
  Tooltip,
  Link,
  useTheme,
  alpha
} from '@mui/material';
import {
  AccessTime,
  TrendingUp,
  TrendingDown,
  TrendingFlat,
  Share,
  BookmarkBorder,
  Bookmark,
  OpenInNew,
  SentimentVeryDissatisfied,
  SentimentDissatisfied,
  SentimentNeutral,
  SentimentSatisfied,
  SentimentVerySatisfied
} from '@mui/icons-material';
import { formatDistanceToNow, parseISO } from 'date-fns';

interface NewsCardProps {
  news: {
    id?: string;
    title: string;
    summary?: string;
    content?: string;
    url?: string;
    source?: string;
    author?: string;
    publishedAt: string;
    image?: string;
    tickers?: string[];
    sentiment?: 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive';
    sentimentScore?: number;
    relevance?: number;
    category?: string;
    impact?: 'low' | 'medium' | 'high';
  };
  variant?: 'default' | 'compact' | 'featured';
  onBookmark?: (newsId: string) => void;
  onShare?: (newsId: string) => void;
  isBookmarked?: boolean;
}

const NewsCard: React.FC<NewsCardProps> = ({
  news,
  variant = 'default',
  onBookmark,
  onShare,
  isBookmarked = false
}) => {
  const theme = useTheme();

  const getSentimentIcon = (sentiment?: string) => {
    switch (sentiment) {
      case 'very_positive':
        return <SentimentVerySatisfied sx={{ color: theme.palette.success.dark }} />;
      case 'positive':
        return <SentimentSatisfied sx={{ color: theme.palette.success.main }} />;
      case 'neutral':
        return <SentimentNeutral sx={{ color: theme.palette.info.main }} />;
      case 'negative':
        return <SentimentDissatisfied sx={{ color: theme.palette.error.main }} />;
      case 'very_negative':
        return <SentimentVeryDissatisfied sx={{ color: theme.palette.error.dark }} />;
      default:
        return <SentimentNeutral sx={{ color: theme.palette.text.secondary }} />;
    }
  };

  const getSentimentColor = (sentiment?: string) => {
    switch (sentiment) {
      case 'very_positive':
        return theme.palette.success.dark;
      case 'positive':
        return theme.palette.success.main;
      case 'neutral':
        return theme.palette.info.main;
      case 'negative':
        return theme.palette.error.main;
      case 'very_negative':
        return theme.palette.error.dark;
      default:
        return theme.palette.text.secondary;
    }
  };

  const getImpactColor = (impact?: string) => {
    switch (impact) {
      case 'high':
        return theme.palette.error.main;
      case 'medium':
        return theme.palette.warning.main;
      case 'low':
        return theme.palette.success.main;
      default:
        return theme.palette.text.secondary;
    }
  };

  const formatTimeAgo = (dateString: string) => {
    try {
      const date = parseISO(dateString);
      return formatDistanceToNow(date, { addSuffix: true });
    } catch {
      return dateString;
    }
  };

  const handleClick = () => {
    if (news.url) {
      window.open(news.url, '_blank', 'noopener,noreferrer');
    }
  };

  const handleBookmark = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onBookmark && news.id) {
      onBookmark(news.id);
    }
  };

  const handleShare = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onShare && news.id) {
      onShare(news.id);
    }
  };

  if (variant === 'compact') {
    return (
      <Card sx={{ 
        display: 'flex',
        cursor: 'pointer',
        transition: 'all 0.3s',
        '&:hover': {
          bgcolor: alpha(theme.palette.primary.main, 0.02),
          transform: 'translateX(4px)'
        }
      }}
      onClick={handleClick}
      >
        <CardContent sx={{ flex: 1, p: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" gap={2}>
            <Box flex={1}>
              <Typography 
                variant="subtitle2" 
                sx={{ 
                  fontWeight: 600,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  mb: 0.5
                }}
              >
                {news.title}
              </Typography>
              <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
                <Box display="flex" alignItems="center" gap={0.5}>
                  <AccessTime sx={{ fontSize: 14, color: theme.palette.text.secondary }} />
                  <Typography variant="caption" color="textSecondary">
                    {formatTimeAgo(news.publishedAt)}
                  </Typography>
                </Box>
                {news.source && (
                  <Typography variant="caption" color="textSecondary">
                    • {news.source}
                  </Typography>
                )}
                {news.sentiment && (
                  <Tooltip title={`Sentiment: ${news.sentiment.replace('_', ' ')}`}>
                    <Box>{getSentimentIcon(news.sentiment)}</Box>
                  </Tooltip>
                )}
              </Box>
            </Box>
            {news.tickers && news.tickers.length > 0 && (
              <Box display="flex" gap={0.5}>
                {news.tickers.slice(0, 2).map(ticker => (
                  <Chip
                    key={ticker}
                    label={ticker}
                    size="small"
                    sx={{ height: 20, fontSize: '0.7rem' }}
                  />
                ))}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (variant === 'featured') {
    return (
      <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        {news.image && (
          <CardMedia
            component="img"
            height="200"
            image={news.image}
            alt={news.title}
            sx={{ objectFit: 'cover' }}
          />
        )}
        <CardContent sx={{ flex: 1 }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box display="flex" gap={1} flexWrap="wrap">
              {news.category && (
                <Chip label={news.category} size="small" variant="outlined" />
              )}
              {news.impact && (
                <Chip
                  label={news.impact.toUpperCase()}
                  size="small"
                  sx={{
                    bgcolor: alpha(getImpactColor(news.impact), 0.1),
                    color: getImpactColor(news.impact),
                    fontWeight: 'bold'
                  }}
                />
              )}
            </Box>
            <Box display="flex" gap={0.5}>
              <IconButton size="small" onClick={handleBookmark}>
                {isBookmarked ? <Bookmark /> : <BookmarkBorder />}
              </IconButton>
              <IconButton size="small" onClick={handleShare}>
                <Share />
              </IconButton>
            </Box>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            {news.title}
          </Typography>

          {news.summary && (
            <Typography 
              variant="body2" 
              color="textSecondary" 
              paragraph
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 3,
                WebkitBoxOrient: 'vertical'
              }}
            >
              {news.summary}
            </Typography>
          )}

          <Box display="flex" justifyContent="space-between" alignItems="center" mt={2}>
            <Box display="flex" alignItems="center" gap={2}>
              <Box display="flex" alignItems="center" gap={0.5}>
                <AccessTime sx={{ fontSize: 16, color: theme.palette.text.secondary }} />
                <Typography variant="caption" color="textSecondary">
                  {formatTimeAgo(news.publishedAt)}
                </Typography>
              </Box>
              {news.source && (
                <Typography variant="caption" color="textSecondary">
                  {news.source}
                </Typography>
              )}
            </Box>
            
            {news.sentiment && (
              <Box display="flex" alignItems="center" gap={1}>
                <Typography variant="caption" color="textSecondary">
                  Sentiment
                </Typography>
                {getSentimentIcon(news.sentiment)}
              </Box>
            )}
          </Box>

          {news.tickers && news.tickers.length > 0 && (
            <Box display="flex" gap={0.5} mt={2} flexWrap="wrap">
              {news.tickers.map(ticker => (
                <Chip
                  key={ticker}
                  label={ticker}
                  size="small"
                  clickable
                  sx={{
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.2)
                    }
                  }}
                />
              ))}
            </Box>
          )}
        </CardContent>
        
        <CardActionArea onClick={handleClick} sx={{ p: 2, pt: 0 }}>
          <Box display="flex" alignItems="center" justifyContent="center" gap={0.5}>
            <Typography variant="button" color="primary">
              Read Full Article
            </Typography>
            <OpenInNew sx={{ fontSize: 16, color: theme.palette.primary.main }} />
          </Box>
        </CardActionArea>
      </Card>
    );
  }

  // Default variant
  return (
    <Card sx={{ 
      display: 'flex',
      cursor: 'pointer',
      transition: 'all 0.3s',
      '&:hover': {
        boxShadow: theme.shadows[4],
        transform: 'translateY(-2px)'
      }
    }}
    onClick={handleClick}
    >
      {news.image && (
        <CardMedia
          component="img"
          sx={{ width: 140, objectFit: 'cover' }}
          image={news.image}
          alt={news.title}
        />
      )}
      <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1 }}>
        <CardContent>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1}>
            <Typography variant="h6" sx={{ 
              fontWeight: 600,
              fontSize: '1rem',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              display: '-webkit-box',
              WebkitLineClamp: 2,
              WebkitBoxOrient: 'vertical',
              flex: 1
            }}>
              {news.title}
            </Typography>
            <Box display="flex" gap={0.5} ml={2}>
              <IconButton size="small" onClick={handleBookmark}>
                {isBookmarked ? 
                  <Bookmark sx={{ fontSize: 18 }} /> : 
                  <BookmarkBorder sx={{ fontSize: 18 }} />
                }
              </IconButton>
              <IconButton size="small" onClick={handleShare}>
                <Share sx={{ fontSize: 18 }} />
              </IconButton>
            </Box>
          </Box>

          {news.summary && (
            <Typography 
              variant="body2" 
              color="textSecondary" 
              sx={{
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                display: '-webkit-box',
                WebkitLineClamp: 2,
                WebkitBoxOrient: 'vertical',
                mb: 2
              }}
            >
              {news.summary}
            </Typography>
          )}

          <Box display="flex" justifyContent="space-between" alignItems="center">
            <Box display="flex" alignItems="center" gap={2}>
              <Box display="flex" alignItems="center" gap={0.5}>
                <AccessTime sx={{ fontSize: 14, color: theme.palette.text.secondary }} />
                <Typography variant="caption" color="textSecondary">
                  {formatTimeAgo(news.publishedAt)}
                </Typography>
              </Box>
              {news.source && (
                <Typography variant="caption" color="textSecondary">
                  {news.source}
                </Typography>
              )}
              {news.author && (
                <Typography variant="caption" color="textSecondary">
                  by {news.author}
                </Typography>
              )}
            </Box>

            <Box display="flex" alignItems="center" gap={1}>
              {news.sentiment && (
                <Tooltip title={`Sentiment: ${news.sentiment.replace('_', ' ')}`}>
                  <Box>{getSentimentIcon(news.sentiment)}</Box>
                </Tooltip>
              )}
              {news.impact && (
                <Chip
                  label={news.impact}
                  size="small"
                  sx={{
                    height: 20,
                    fontSize: '0.7rem',
                    bgcolor: alpha(getImpactColor(news.impact), 0.1),
                    color: getImpactColor(news.impact)
                  }}
                />
              )}
            </Box>
          </Box>

          {news.tickers && news.tickers.length > 0 && (
            <Box display="flex" gap={0.5} mt={1.5} flexWrap="wrap">
              {news.tickers.map(ticker => (
                <Chip
                  key={ticker}
                  label={ticker}
                  size="small"
                  sx={{
                    height: 22,
                    fontSize: '0.75rem',
                    bgcolor: alpha(theme.palette.primary.main, 0.1),
                    '&:hover': {
                      bgcolor: alpha(theme.palette.primary.main, 0.2)
                    }
                  }}
                />
              ))}
            </Box>
          )}
        </CardContent>
      </Box>
    </Card>
  );
};

export default NewsCard;