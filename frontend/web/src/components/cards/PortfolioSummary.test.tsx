import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PortfolioSummary from './PortfolioSummary';
import { renderWithProviders, mockPortfolioSummary } from '../../test-utils';

describe('PortfolioSummary', () => {
  describe('without data', () => {
    it('renders empty state when no summary is provided', () => {
      renderWithProviders(<PortfolioSummary />);

      expect(screen.getByText(/no portfolio data available/i)).toBeInTheDocument();
    });
  });

  describe('compact mode', () => {
    it('renders compact view correctly', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} compact={true} />
      );

      expect(screen.getByText('Portfolio Summary')).toBeInTheDocument();
      expect(screen.getByText('Total Value')).toBeInTheDocument();
      expect(screen.getByText('Total Return')).toBeInTheDocument();
      // Check that total value is formatted as currency
      expect(screen.getByText('$150,000')).toBeInTheDocument();
    });

    it('displays correct return percentage with color', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} compact={true} />
      );

      // Positive return should show +
      expect(screen.getByText('+25.00%')).toBeInTheDocument();
    });
  });

  describe('full mode', () => {
    it('renders full view with all sections', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} compact={false} />
      );

      expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
      expect(screen.getByText('Performance')).toBeInTheDocument();
      expect(screen.getByText('Allocation')).toBeInTheDocument();
      expect(screen.getByText('Top Movers')).toBeInTheDocument();
    });

    it('displays total value correctly', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('$150,000')).toBeInTheDocument();
    });

    it('displays performance periods', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('1 Week')).toBeInTheDocument();
      expect(screen.getByText('1 Month')).toBeInTheDocument();
      expect(screen.getByText('1 Year')).toBeInTheDocument();
      expect(screen.getByText('Positions')).toBeInTheDocument();
    });

    it('displays top gainers', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('Gainers')).toBeInTheDocument();
      expect(screen.getByText('AAPL')).toBeInTheDocument();
      expect(screen.getByText('Apple Inc.')).toBeInTheDocument();
    });

    it('displays top losers', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('Losers')).toBeInTheDocument();
      expect(screen.getByText('META')).toBeInTheDocument();
      expect(screen.getByText('Meta Platforms')).toBeInTheDocument();
    });

    it('displays risk metrics when available', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('Risk Metrics')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
      expect(screen.getByText('Beta')).toBeInTheDocument();
      expect(screen.getByText('Std Dev')).toBeInTheDocument();
      expect(screen.getByText('Max DD')).toBeInTheDocument();
      expect(screen.getByText('1.45')).toBeInTheDocument(); // Sharpe ratio value
    });

    it('displays allocation chips', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByText('Technology: 35%')).toBeInTheDocument();
      expect(screen.getByText('Healthcare: 20%')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      renderWithProviders(
        <PortfolioSummary summary={mockPortfolioSummary} />
      );

      expect(screen.getByRole('button', { name: /details/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /rebalance/i })).toBeInTheDocument();
    });
  });

  describe('negative returns', () => {
    it('displays negative return correctly', () => {
      const negativeSummary = {
        ...mockPortfolioSummary,
        totalReturn: -5000,
        totalReturnPercent: -3.5,
      };

      renderWithProviders(
        <PortfolioSummary summary={negativeSummary} compact={true} />
      );

      expect(screen.getByText('-3.50%')).toBeInTheDocument();
    });
  });

  describe('missing optional data', () => {
    it('renders without risk metrics', () => {
      const summaryWithoutRisk = {
        ...mockPortfolioSummary,
        riskMetrics: undefined,
      };

      renderWithProviders(
        <PortfolioSummary summary={summaryWithoutRisk} />
      );

      expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
      expect(screen.queryByText('Risk Metrics')).not.toBeInTheDocument();
    });

    it('renders without top movers', () => {
      const summaryWithoutMovers = {
        ...mockPortfolioSummary,
        topGainers: undefined,
        topLosers: undefined,
      };

      renderWithProviders(
        <PortfolioSummary summary={summaryWithoutMovers} />
      );

      expect(screen.getByText('Portfolio Overview')).toBeInTheDocument();
      expect(screen.queryByText('Gainers')).not.toBeInTheDocument();
      expect(screen.queryByText('Losers')).not.toBeInTheDocument();
    });
  });
});
