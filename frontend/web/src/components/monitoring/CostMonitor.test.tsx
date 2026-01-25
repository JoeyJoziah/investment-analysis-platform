import { describe, it, expect, vi } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CostMonitor from './CostMonitor';
import { renderWithProviders, mockCostMetrics } from '../../test-utils';

describe('CostMonitor', () => {
  describe('without data', () => {
    it('renders empty state when no metrics are provided', () => {
      renderWithProviders(<CostMonitor />);

      expect(screen.getByText(/no cost metrics available/i)).toBeInTheDocument();
    });
  });

  describe('with metrics data', () => {
    it('renders the Cost Monitor title', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('Cost Monitor')).toBeInTheDocument();
    });

    it('displays current month cost and budget', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('Monthly Budget')).toBeInTheDocument();
      expect(screen.getByText(/\$35\.50/)).toBeInTheDocument();
      expect(screen.getByText(/\$50\.00/)).toBeInTheDocument();
    });

    it('displays budget usage percentage', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      // 35.5 / 50 = 71%
      expect(screen.getByText(/71\.0% used/i)).toBeInTheDocument();
    });

    it('displays projected month cost', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('Projected Month')).toBeInTheDocument();
      expect(screen.getByText('$42.00')).toBeInTheDocument();
    });

    it('displays daily average cost', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('Daily Average')).toBeInTheDocument();
      expect(screen.getByText('$1.50')).toBeInTheDocument();
    });

    it('displays API usage section', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('API Usage')).toBeInTheDocument();
      expect(screen.getByText('Alpha Vantage')).toBeInTheDocument();
      expect(screen.getByText('Finnhub')).toBeInTheDocument();
      expect(screen.getByText('Polygon.io')).toBeInTheDocument();
    });

    it('displays API usage counts', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByText('18/25 calls')).toBeInTheDocument();
      expect(screen.getByText('45/60 calls')).toBeInTheDocument();
      expect(screen.getByText('3/5 calls')).toBeInTheDocument();
    });
  });

  describe('budget status indicators', () => {
    it('shows success status when under 80% budget', () => {
      const underBudgetMetrics = {
        ...mockCostMetrics,
        currentMonthCost: 25,
      };

      renderWithProviders(<CostMonitor metrics={underBudgetMetrics} />);

      // Should show check circle icon (success state)
      expect(screen.getByText(/50\.0% used/i)).toBeInTheDocument();
    });

    it('shows warning status when over 80% budget', () => {
      const nearBudgetMetrics = {
        ...mockCostMetrics,
        currentMonthCost: 45,
      };

      renderWithProviders(<CostMonitor metrics={nearBudgetMetrics} />);

      // Should show warning indicator
      expect(screen.getByText(/90\.0% used/i)).toBeInTheDocument();
    });

    it('shows error status when over budget', () => {
      const overBudgetMetrics = {
        ...mockCostMetrics,
        currentMonthCost: 55,
      };

      renderWithProviders(<CostMonitor metrics={overBudgetMetrics} />);

      // Should show error indicator and 100% (capped)
      expect(screen.getByText(/110\.0% used/i)).toBeInTheDocument();
    });
  });

  describe('special modes', () => {
    it('displays savings mode chip when enabled', () => {
      const savingsModeMetrics = {
        ...mockCostMetrics,
        savingsMode: true,
      };

      renderWithProviders(<CostMonitor metrics={savingsModeMetrics} />);

      expect(screen.getByText('Savings Mode')).toBeInTheDocument();
    });

    it('displays emergency mode chip when enabled', () => {
      const emergencyModeMetrics = {
        ...mockCostMetrics,
        emergencyMode: true,
      };

      renderWithProviders(<CostMonitor metrics={emergencyModeMetrics} />);

      expect(screen.getByText('Emergency Mode')).toBeInTheDocument();
    });
  });

  describe('alerts', () => {
    it('displays alert messages when present', () => {
      const metricsWithAlerts = {
        ...mockCostMetrics,
        alerts: [
          { type: 'warning' as const, message: 'Approaching budget limit' },
          { type: 'info' as const, message: 'Cost optimization available' },
        ],
      };

      renderWithProviders(<CostMonitor metrics={metricsWithAlerts} />);

      expect(screen.getByText('Approaching budget limit')).toBeInTheDocument();
      expect(screen.getByText('Cost optimization available')).toBeInTheDocument();
    });
  });

  describe('expandable details', () => {
    it('shows details button', () => {
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      expect(screen.getByRole('button', { name: /show details/i })).toBeInTheDocument();
    });

    it('toggles details on button click', async () => {
      const user = userEvent.setup();
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      const detailsButton = screen.getByRole('button', { name: /show details/i });
      await user.click(detailsButton);

      expect(screen.getByRole('button', { name: /hide details/i })).toBeInTheDocument();
    });

    it('shows cost breakdown when expanded', async () => {
      const user = userEvent.setup();
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      const detailsButton = screen.getByRole('button', { name: /show details/i });
      await user.click(detailsButton);

      expect(screen.getByText('Cost Breakdown')).toBeInTheDocument();
    });

    it('shows last updated timestamp when expanded', async () => {
      const user = userEvent.setup();
      renderWithProviders(<CostMonitor metrics={mockCostMetrics} />);

      const detailsButton = screen.getByRole('button', { name: /show details/i });
      await user.click(detailsButton);

      expect(screen.getByText(/last updated/i)).toBeInTheDocument();
    });
  });

  describe('action buttons', () => {
    it('calls onRefresh when refresh button is clicked', async () => {
      const user = userEvent.setup();
      const mockRefresh = vi.fn();

      renderWithProviders(
        <CostMonitor metrics={mockCostMetrics} onRefresh={mockRefresh} />
      );

      const refreshButton = screen.getByRole('button', { name: /refresh cost metrics/i });
      await user.click(refreshButton);

      expect(mockRefresh).toHaveBeenCalledTimes(1);
    });

    it('calls onSettings when settings button is clicked', async () => {
      const user = userEvent.setup();
      const mockSettings = vi.fn();

      renderWithProviders(
        <CostMonitor metrics={mockCostMetrics} onSettings={mockSettings} />
      );

      const settingsButton = screen.getByRole('button', { name: /open cost monitor settings/i });
      await user.click(settingsButton);

      expect(mockSettings).toHaveBeenCalledTimes(1);
    });
  });

  describe('over budget warning', () => {
    it('shows over budget warning when projected exceeds budget', () => {
      const overBudgetProjectedMetrics = {
        ...mockCostMetrics,
        projectedMonthCost: 60,
      };

      renderWithProviders(<CostMonitor metrics={overBudgetProjectedMetrics} />);

      expect(screen.getByText(/over budget by/i)).toBeInTheDocument();
      // The warning shows the amount over budget
      expect(screen.getByText(/\$10\.00/)).toBeInTheDocument();
    });
  });
});
