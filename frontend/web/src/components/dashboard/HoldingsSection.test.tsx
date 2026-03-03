import { describe, it, expect, vi } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import HoldingsSection from './HoldingsSection';
import { renderWithProviders, mockPosition } from '../../test-utils';
import { Position } from '../../types';

// Mock the Sparkline chart to avoid canvas rendering issues in tests
vi.mock('../charts/Sparkline', () => ({
  default: () => <div data-testid="sparkline">Sparkline</div>,
}));

const makePosition = (overrides: Partial<Position> = {}): Position => ({
  ...mockPosition,
  ...overrides,
});

const positions: Position[] = [
  makePosition({ id: '1', ticker: 'AAPL', companyName: 'Apple Inc.', marketValue: 17500, dayGainPercent: 0.72 }),
  makePosition({ id: '2', ticker: 'MSFT', companyName: 'Microsoft Corp.', marketValue: 25000, dayGainPercent: -1.05 }),
  makePosition({ id: '3', ticker: 'GOOGL', companyName: 'Alphabet Inc.', marketValue: 12000, dayGainPercent: 0.15 }),
];

describe('HoldingsSection', () => {
  it('renders the Holdings heading and positions', () => {
    renderWithProviders(<HoldingsSection positions={positions} />);

    expect(screen.getByRole('heading', { name: /holdings/i })).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(screen.getByText('MSFT')).toBeInTheDocument();
    expect(screen.getByText('GOOGL')).toBeInTheDocument();
  });

  it('renders empty state when positions array is empty', () => {
    renderWithProviders(<HoldingsSection positions={[]} />);

    expect(screen.getByText(/no positions in your portfolio/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /add position/i })).toBeInTheDocument();
  });

  it('renders loading skeletons when isLoading is true', () => {
    const { container } = renderWithProviders(
      <HoldingsSection positions={[]} isLoading />
    );

    // MUI Skeleton elements should be present
    const skeletons = container.querySelectorAll('.MuiSkeleton-root');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('renders Add Position button when positions exist', () => {
    renderWithProviders(<HoldingsSection positions={positions} />);

    const addButtons = screen.getAllByRole('button', { name: /add position/i });
    expect(addButtons.length).toBeGreaterThanOrEqual(1);
  });

  it('calls onAddPosition when Add Position button is clicked', () => {
    const onAdd = vi.fn();

    renderWithProviders(
      <HoldingsSection positions={positions} onAddPosition={onAdd} />
    );

    const addButton = screen.getByRole('button', { name: /add position/i });
    fireEvent.click(addButton);
    expect(onAdd).toHaveBeenCalledTimes(1);
  });

  it('renders the table with sortable column headers', () => {
    renderWithProviders(<HoldingsSection positions={positions} />);

    expect(screen.getByText('Symbol')).toBeInTheDocument();
    expect(screen.getByText('Price')).toBeInTheDocument();
    expect(screen.getByText('Value')).toBeInTheDocument();
    expect(screen.getByText('Day P&L')).toBeInTheDocument();
  });

  it('renders Show All button when positions exceed maxRows', () => {
    renderWithProviders(
      <HoldingsSection positions={positions} maxRows={1} />
    );

    expect(screen.getByRole('button', { name: /show all/i })).toBeInTheDocument();
  });

  it('does not render Show All when positions fit within maxRows', () => {
    renderWithProviders(
      <HoldingsSection positions={positions} maxRows={10} />
    );

    expect(screen.queryByRole('button', { name: /show all/i })).not.toBeInTheDocument();
  });

  it('toggles Show All / Show Less', () => {
    renderWithProviders(
      <HoldingsSection positions={positions} maxRows={1} />
    );

    const toggleBtn = screen.getByRole('button', { name: /show all/i });
    fireEvent.click(toggleBtn);
    expect(screen.getByRole('button', { name: /show less/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /show less/i }));
    expect(screen.getByRole('button', { name: /show all/i })).toBeInTheDocument();
  });

  it('has accessible table with proper aria-label', () => {
    renderWithProviders(<HoldingsSection positions={positions} />);

    const table = screen.getByRole('table');
    expect(table).toHaveAttribute(
      'aria-label',
      expect.stringContaining('3 positions')
    );
  });
});
