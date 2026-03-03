import { describe, it, expect } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import Alerts from './Alerts';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

describe('Alerts', () => {
  const defaultState = mergeWithDefaults({});

  it('renders the page heading and description', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(screen.getByRole('heading', { name: /alerts/i })).toBeInTheDocument();
    expect(
      screen.getByText(/monitor price movements, volume spikes, and portfolio drift/i)
    ).toBeInTheDocument();
  });

  it('renders summary cards', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(screen.getByText('Total Active')).toBeInTheDocument();
    expect(screen.getByText('Triggered Today')).toBeInTheDocument();
    expect(screen.getByText('Expiring Soon')).toBeInTheDocument();
  });

  it('renders the Active Alerts and Alert History tabs', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(screen.getByRole('tab', { name: /active alerts/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /alert history/i })).toBeInTheDocument();
  });

  it('displays seed alerts in the active tab', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('GOOGL')).toBeInTheDocument();
  });

  it('renders the Create Alert button', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(
      screen.getByRole('button', { name: /create alert/i })
    ).toBeInTheDocument();
  });

  it('opens the create dialog when Create Alert is clicked', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    fireEvent.click(screen.getByRole('button', { name: /create alert/i }));

    // Dialog should appear with heading and the ticker field
    expect(screen.getByRole('dialog')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /create alert/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/ticker symbol/i)).toBeInTheDocument();
  });

  it('renders Cancel and Create buttons in dialog', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    fireEvent.click(screen.getByRole('button', { name: /create alert/i }));

    expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^create$/i })).toBeInTheDocument();
  });

  it('switches to the history tab', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    fireEvent.click(screen.getByRole('tab', { name: /alert history/i }));

    // History tab should show triggered alerts
    expect(screen.getByText('TSLA')).toBeInTheDocument();
    expect(screen.getByText('NVDA')).toBeInTheDocument();
  });

  it('renders the search field', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    expect(
      screen.getByPlaceholderText(/search alerts/i)
    ).toBeInTheDocument();
  });

  it('filters alerts by search query', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    const searchInput = screen.getByPlaceholderText(/search alerts/i);
    fireEvent.change(searchInput, { target: { value: 'AAPL' } });

    // AAPL should remain visible
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    // GOOGL should be filtered out
    expect(screen.queryByText('GOOGL')).not.toBeInTheDocument();
  });

  it('shows correct active alert count', () => {
    renderWithProviders(<Alerts />, { preloadedState: defaultState });

    // 4 active alerts in the seed data (AAPL, GOOGL, AMZN, Portfolio)
    const summaryCards = screen.getAllByText('4');
    expect(summaryCards.length).toBeGreaterThanOrEqual(1);
  });
});
