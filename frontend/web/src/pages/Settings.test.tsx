import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Settings from './Settings';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    put: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

describe('Settings', () => {
  const defaultState = mergeWithDefaults({
    app: {
      isInitialized: true,
      isAuthenticated: true,
      user: { id: '1', email: 'test@example.com', name: 'Test User' } as never,
      themeMode: 'light' as const,
      sidebarOpen: true,
      searchOpen: false,
      notifications: [],
      webSocketConnected: false,
    },
  });

  it('renders the settings page title', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    expect(screen.getByRole('heading', { name: /settings/i })).toBeInTheDocument();
  });

  it('renders the subtitle', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    expect(screen.getByText(/manage your account settings and preferences/i)).toBeInTheDocument();
  });

  it('renders all setting tabs', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    expect(screen.getByRole('tab', { name: /profile/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /appearance/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /notifications/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /api keys/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /security/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /data & privacy/i })).toBeInTheDocument();
  });

  it('shows profile tab by default', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    expect(screen.getByText('Profile Information')).toBeInTheDocument();
  });

  it('switches to appearance tab when clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    await user.click(screen.getByRole('tab', { name: /appearance/i }));

    expect(screen.getByText('Appearance Settings')).toBeInTheDocument();
  });

  it('switches to notifications tab when clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    await user.click(screen.getByRole('tab', { name: /notifications/i }));

    expect(screen.getByText('Notification Settings')).toBeInTheDocument();
  });

  it('switches to API keys tab when clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    await user.click(screen.getByRole('tab', { name: /api keys/i }));

    expect(screen.getByText('API Configuration')).toBeInTheDocument();
  });

  it('switches to security tab when clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    await user.click(screen.getByRole('tab', { name: /security/i }));

    expect(screen.getByText('Security Settings')).toBeInTheDocument();
  });

  it('switches to data and privacy tab when clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    await user.click(screen.getByRole('tab', { name: /data & privacy/i }));

    expect(screen.getByText('Data & Privacy Settings')).toBeInTheDocument();
  });

  it('renders save profile button on profile tab', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    expect(screen.getByRole('button', { name: /save profile/i })).toBeInTheDocument();
  });

  it('renders user name field with default value', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    const nameField = screen.getByLabelText(/full name/i);
    expect(nameField).toBeInTheDocument();
    expect(nameField).toHaveValue('Test User');
  });

  it('renders user email field with default value', () => {
    renderWithProviders(<Settings />, { preloadedState: defaultState });

    const emailField = screen.getByLabelText(/email/i);
    expect(emailField).toBeInTheDocument();
    expect(emailField).toHaveValue('test@example.com');
  });
});
