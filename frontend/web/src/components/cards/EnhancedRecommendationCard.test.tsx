import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import EnhancedRecommendationCard from './EnhancedRecommendationCard';
import { renderWithProviders } from '../../test-utils';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: React.PropsWithChildren<Record<string, unknown>>) => {
      const { initial: _i, animate: _a, transition: _t, ...htmlProps } = props;
      return <div {...htmlProps}>{children}</div>;
    },
  },
  AnimatePresence: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const baseRecommendation = {
  ticker: 'AAPL',
  company_name: 'Apple Inc.',
  action: 'BUY' as const,
  confidence: 85,
  current_price: 175.5,
  target_price: 200,
  potential_return: 13.96,
  risk_level: 'LOW' as const,
  reasoning: 'Strong fundamentals and growth potential.',
  technical_score: 80,
  fundamental_score: 90,
  sentiment_score: 75,
  time_horizon: '6-12 months',
  sector: 'Technology',
  esg_score: 72,
};

describe('EnhancedRecommendationCard', () => {
  describe('loading state', () => {
    it('renders skeleton when loading', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} loading />
      );
      // Skeletons don't have text, just check the card renders without crash
      expect(document.querySelector('.MuiSkeleton-root')).toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('renders error alert', () => {
      renderWithProviders(
        <EnhancedRecommendationCard
          recommendation={baseRecommendation}
          error="Something went wrong"
        />
      );
      expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    });
  });

  describe('compact view', () => {
    it('renders ticker and action in compact mode', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} compact />
      );
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('BUY')).toBeInTheDocument();
      expect(screen.getByText('85% confidence')).toBeInTheDocument();
    });

    it('renders current price in compact mode', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} compact />
      );
      expect(screen.getByText('$175.50')).toBeInTheDocument();
    });

    it('renders potential return in compact mode', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} compact />
      );
      expect(screen.getByText('+13.96%')).toBeInTheDocument();
    });
  });

  describe('full view', () => {
    it('renders ticker and company name', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    it('renders sector chip', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('Technology')).toBeInTheDocument();
    });

    it('renders action chip with BUY label', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      // BUY appears in the action chip and the trade button
      const buyElements = screen.getAllByText('BUY');
      expect(buyElements.length).toBeGreaterThanOrEqual(1);
    });

    it('renders confidence percentage', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('85%')).toBeInTheDocument();
    });

    it('renders current price', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('$175.50')).toBeInTheDocument();
    });

    it('renders target price', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('$200.00')).toBeInTheDocument();
    });

    it('renders potential return', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('+13.96%')).toBeInTheDocument();
    });

    it('renders risk level chip', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('Risk: LOW')).toBeInTheDocument();
    });

    it('renders time horizon chip', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('6-12 months')).toBeInTheDocument();
    });

    it('renders ESG score chip', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('ESG: 72/100')).toBeInTheDocument();
    });

    it('renders analysis score labels', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByText('Technical')).toBeInTheDocument();
      expect(screen.getByText('Fundamental')).toBeInTheDocument();
      expect(screen.getByText('Sentiment')).toBeInTheDocument();
    });

    it('renders View Details button', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByRole('button', { name: /view detailed analysis for aapl/i })).toBeInTheDocument();
    });

    it('renders Show Analysis toggle button', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByRole('button', { name: /show analysis/i })).toBeInTheDocument();
    });

    it('expands reasoning section on toggle click', async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      const toggle = screen.getByRole('button', { name: /show analysis/i });
      await user.click(toggle);
      expect(screen.getByText('Strong fundamentals and growth potential.')).toBeInTheDocument();
    });

    it('renders bookmark and notification buttons', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByRole('button', { name: /add to watchlist/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /enable price alerts/i })).toBeInTheDocument();
    });

    it('renders more options button', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} />
      );
      expect(screen.getByRole('button', { name: /more options/i })).toBeInTheDocument();
    });

    it('renders with selected border', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={baseRecommendation} selected />
      );
      const card = document.getElementById('card-AAPL');
      expect(card).toBeInTheDocument();
    });
  });

  describe('SELL recommendation', () => {
    const sellRecommendation = {
      ...baseRecommendation,
      action: 'SELL' as const,
      potential_return: -5.2,
    };

    it('renders SELL action', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={sellRecommendation} />
      );
      const sellElements = screen.getAllByText('SELL');
      expect(sellElements.length).toBeGreaterThanOrEqual(1);
    });

    it('renders negative return', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={sellRecommendation} />
      );
      expect(screen.getByText('-5.20%')).toBeInTheDocument();
    });
  });

  describe('minimal recommendation', () => {
    const minimalRecommendation = {
      ticker: 'TSLA',
      action: 'HOLD' as const,
      confidence: 50,
      current_price: 250,
    };

    it('renders with minimal data', () => {
      renderWithProviders(
        <EnhancedRecommendationCard recommendation={minimalRecommendation} />
      );
      expect(screen.getByText('TSLA')).toBeInTheDocument();
      expect(screen.getByText('50%')).toBeInTheDocument();
    });
  });
});
