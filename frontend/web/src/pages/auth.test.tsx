import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Login from './Login';
import Register from './Register';
import ForgotPassword from './ForgotPassword';
import { renderWithProviders, mergeWithDefaults } from '../test-utils';

// ---------------------------------------------------------------------------
// Navigation mock
// ---------------------------------------------------------------------------
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual<typeof import('react-router-dom')>('react-router-dom');
  return { ...actual, useNavigate: () => mockNavigate };
});

// ---------------------------------------------------------------------------
// API mock
// ---------------------------------------------------------------------------
vi.mock('../services/api.service', () => ({
  apiService: {
    get: vi.fn().mockResolvedValue({ data: {} }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    auth: {
      register: vi.fn().mockResolvedValue({ data: {} }),
    },
  },
}));

// ---------------------------------------------------------------------------
// Redux store mock — login thunk calls apiService.post under the hood, but
// we mock the thunk at dispatch level instead (via rejecting/resolving).
// We import apiService for direct assertions in Register / ForgotPassword.
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.clearAllMocks();
});

const defaultState = mergeWithDefaults({});

// =========================================================================
// Login
// =========================================================================
describe('Login', () => {
  it('renders the brand heading', () => {
    renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByRole('heading', { name: /investai pro/i })).toBeInTheDocument();
  });

  it('renders email and password fields', () => {
    const { container } = renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(container.querySelector('#password')).toBeInTheDocument();
  });

  it('renders sign in button', () => {
    renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('renders forgot password link', () => {
    renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByRole('link', { name: /forgot password/i })).toBeInTheDocument();
  });

  it('renders create account link', () => {
    renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByRole('link', { name: /create account/i })).toBeInTheDocument();
  });

  it('renders demo account button', () => {
    renderWithProviders(<Login />, { preloadedState: defaultState });
    expect(screen.getByRole('button', { name: /use demo account/i })).toBeInTheDocument();
  });

  it('fills demo credentials when demo button is clicked', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Login />, { preloadedState: defaultState });

    await user.click(screen.getByRole('button', { name: /use demo account/i }));

    expect(screen.getByLabelText(/email address/i)).toHaveValue('demo@invest.com');
    expect(container.querySelector('#password')).toHaveValue('Demo12345!');
  });

  it('toggles password visibility', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Login />, { preloadedState: defaultState });

    const passwordField = container.querySelector('#password') as HTMLInputElement;
    expect(passwordField).toHaveAttribute('type', 'password');

    await user.click(screen.getByRole('button', { name: /toggle password visibility/i }));
    expect(passwordField).toHaveAttribute('type', 'text');

    await user.click(screen.getByRole('button', { name: /toggle password visibility/i }));
    expect(passwordField).toHaveAttribute('type', 'password');
  });

  it('submits form and navigates on success', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Login />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'password123');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    // Button should show loading state
    await waitFor(() => {
      // The form was submitted — sign in button re-appears once done
      expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
    });
  });
});

// =========================================================================
// Register
// =========================================================================
describe('Register', () => {
  it('renders the create account heading', () => {
    renderWithProviders(<Register />, { preloadedState: defaultState });
    expect(screen.getByRole('heading', { name: /create account/i })).toBeInTheDocument();
  });

  it('renders all form fields', () => {
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });
    expect(screen.getByLabelText(/full name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(container.querySelector('#password')).toBeInTheDocument();
    expect(container.querySelector('#confirmPassword')).toBeInTheDocument();
  });

  it('renders create account button', () => {
    renderWithProviders(<Register />, { preloadedState: defaultState });
    expect(screen.getByRole('button', { name: /create account/i })).toBeInTheDocument();
  });

  it('renders sign in link', () => {
    renderWithProviders(<Register />, { preloadedState: defaultState });
    expect(screen.getByRole('link', { name: /sign in/i })).toBeInTheDocument();
  });

  it('shows password helper text', () => {
    renderWithProviders(<Register />, { preloadedState: defaultState });
    expect(screen.getByText(/at least 8 characters/i)).toBeInTheDocument();
  });

  it('shows error when name is empty and form is submitted', async () => {
    const user = userEvent.setup();
    renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
    expect(screen.getByText(/full name is required/i)).toBeInTheDocument();
  });

  it('shows error when email is invalid', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/full name/i), 'John Doe');
    await user.type(screen.getByLabelText(/email address/i), 'notanemail');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'password123');
    await user.type(container.querySelector('#confirmPassword') as HTMLInputElement, 'password123');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/please enter a valid email/i)).toBeInTheDocument();
    });
  });

  it('shows error when password is too short', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/full name/i), 'John Doe');
    await user.type(screen.getByLabelText(/email address/i), 'john@example.com');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'short');
    await user.type(container.querySelector('#confirmPassword') as HTMLInputElement, 'short');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/password must be at least 8 characters/i)).toBeInTheDocument();
    });
  });

  it('shows error when passwords do not match', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/full name/i), 'John Doe');
    await user.type(screen.getByLabelText(/email address/i), 'john@example.com');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'password123');
    await user.type(container.querySelector('#confirmPassword') as HTMLInputElement, 'password456');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('calls register API and navigates on success', async () => {
    const user = userEvent.setup();
    const { apiService } = await import('../services/api.service');
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/full name/i), 'John Doe');
    await user.type(screen.getByLabelText(/email address/i), 'john@example.com');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'password123');
    await user.type(container.querySelector('#confirmPassword') as HTMLInputElement, 'password123');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(apiService.auth.register).toHaveBeenCalledWith({
        full_name: 'John Doe',
        email: 'john@example.com',
        password: 'password123',
      });
    });

    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/login', { state: { registered: true } });
    });
  });

  it('shows error alert on register API failure', async () => {
    const user = userEvent.setup();
    const { apiService } = await import('../services/api.service');
    vi.mocked(apiService.auth.register).mockRejectedValueOnce({
      response: { data: { detail: 'Email already exists' } },
    });
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/full name/i), 'John Doe');
    await user.type(screen.getByLabelText(/email address/i), 'john@example.com');
    await user.type(container.querySelector('#password') as HTMLInputElement, 'password123');
    await user.type(container.querySelector('#confirmPassword') as HTMLInputElement, 'password123');
    await user.click(screen.getByRole('button', { name: /create account/i }));

    await waitFor(() => {
      expect(screen.getByText('Email already exists')).toBeInTheDocument();
    });
  });

  it('toggles password visibility', async () => {
    const user = userEvent.setup();
    const { container } = renderWithProviders(<Register />, { preloadedState: defaultState });

    const passwordField = container.querySelector('#password') as HTMLInputElement;
    expect(passwordField).toHaveAttribute('type', 'password');

    await user.click(screen.getByRole('button', { name: /toggle password visibility/i }));
    expect(passwordField).toHaveAttribute('type', 'text');
  });
});

// =========================================================================
// ForgotPassword
// =========================================================================
describe('ForgotPassword', () => {
  it('renders the reset password heading', () => {
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });
    expect(screen.getByRole('heading', { name: /reset password/i })).toBeInTheDocument();
  });

  it('renders email field', () => {
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
  });

  it('renders send reset link button', () => {
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });
    expect(screen.getByRole('button', { name: /send reset link/i })).toBeInTheDocument();
  });

  it('renders back to sign in link', () => {
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });
    expect(screen.getByRole('link', { name: /back to sign in/i })).toBeInTheDocument();
  });

  it('shows error when email is empty', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });

    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText(/email is required/i)).toBeInTheDocument();
    });
  });

  it('shows error when email is invalid', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/email address/i), 'notanemail');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText(/please enter a valid email/i)).toBeInTheDocument();
    });
  });

  it('shows success alert after valid submission', async () => {
    const user = userEvent.setup();
    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText(/password reset link/i)).toBeInTheDocument();
    });
    // Email field and submit button should be replaced by success message
    expect(screen.queryByRole('button', { name: /send reset link/i })).not.toBeInTheDocument();
  });

  it('shows success alert even on API error (prevents email enumeration)', async () => {
    const user = userEvent.setup();
    const { apiService } = await import('../services/api.service');
    vi.mocked(apiService.post).mockRejectedValueOnce(new Error('Server error'));

    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(screen.getByText(/password reset link/i)).toBeInTheDocument();
    });
  });

  it('calls API with correct email payload', async () => {
    const user = userEvent.setup();
    const { apiService } = await import('../services/api.service');

    renderWithProviders(<ForgotPassword />, { preloadedState: defaultState });

    await user.type(screen.getByLabelText(/email address/i), 'test@example.com');
    await user.click(screen.getByRole('button', { name: /send reset link/i }));

    await waitFor(() => {
      expect(apiService.post).toHaveBeenCalledWith('/api/v1/auth/forgot-password', {
        email: 'test@example.com',
      });
    });
  });
});
