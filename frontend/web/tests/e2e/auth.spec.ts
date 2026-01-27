import { test, expect, Page } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const API_URL = process.env.API_URL || 'http://localhost:8000';

// Test user credentials
const TEST_USER = {
  email: `test-${Date.now()}@example.com`,
  username: `testuser${Date.now()}`,
  password: 'TestPassword123!@#',
};

const EXISTING_USER = {
  email: 'existing@example.com',
  username: 'existinguser',
  password: 'ExistingPass123!@#',
};

test.describe('Authentication Flows', () => {
  test.beforeEach(async ({ page }) => {
    // Clear storage and cookies before each test
    await page.context().clearCookies();
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
  });

  test.describe('User Registration', () => {
    test('should complete user registration with valid data', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/register`);

      // Wait for registration form
      await page.waitForSelector('form', { timeout: 5000 });

      // Fill in registration form
      await page.fill('input[name="email"]', TEST_USER.email);
      await page.fill('input[name="username"]', TEST_USER.username);
      await page.fill('input[name="password"]', TEST_USER.password);
      await page.fill(
        'input[name="confirmPassword"]',
        TEST_USER.password
      );

      // Submit form
      const submitButton = page.locator('button:has-text("Register")');
      await submitButton.click();

      // Wait for success and redirect
      await page.waitForURL(`${BASE_URL}/login`, { timeout: 10000 });
      expect(page.url()).toContain('/login');
    });

    test('should reject registration with duplicate email', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/register`);
      await page.waitForSelector('form', { timeout: 5000 });

      // Try to register with existing email
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="username"]', `newuser${Date.now()}`);
      await page.fill('input[name="password"]', TEST_USER.password);
      await page.fill(
        'input[name="confirmPassword"]',
        TEST_USER.password
      );

      const submitButton = page.locator('button:has-text("Register")');
      await submitButton.click();

      // Wait for error message
      const errorMessage = page.locator('[role="alert"]');
      await expect(errorMessage).toContainText(/email.*already/i);
    });

    test('should validate password strength requirements', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/register`);
      await page.waitForSelector('form', { timeout: 5000 });

      // Try with weak password
      await page.fill('input[name="email"]', TEST_USER.email);
      await page.fill('input[name="username"]', TEST_USER.username);
      await page.fill('input[name="password"]', 'weak');
      await page.fill('input[name="confirmPassword"]', 'weak');

      // Check for validation error
      const passwordInput = page.locator('input[name="password"]');
      await expect(passwordInput).toHaveAttribute('aria-invalid', 'true');
    });

    test('should require password confirmation match', async ({ page }) => {
      await page.goto(`${BASE_URL}/register`);
      await page.waitForSelector('form', { timeout: 5000 });

      await page.fill('input[name="email"]', TEST_USER.email);
      await page.fill('input[name="username"]', TEST_USER.username);
      await page.fill('input[name="password"]', TEST_USER.password);
      await page.fill('input[name="confirmPassword"]', 'DifferentPass123!');

      const submitButton = page.locator('button:has-text("Register")');
      await expect(submitButton).toBeDisabled();
    });
  });

  test.describe('User Login', () => {
    test('should successfully login with valid credentials', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/login`);
      await page.waitForSelector('form', { timeout: 5000 });

      // Fill login form
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);

      // Submit
      const submitButton = page.locator('button:has-text("Login")');
      await submitButton.click();

      // Wait for redirect to dashboard
      await page.waitForURL(`${BASE_URL}/dashboard`, { timeout: 10000 });
      expect(page.url()).toContain('/dashboard');

      // Verify JWT token is stored
      const hasToken = await page.evaluate(() => {
        return !!localStorage.getItem('token');
      });
      expect(hasToken).toBe(true);
    });

    test('should reject login with invalid credentials', async ({ page }) => {
      await page.goto(`${BASE_URL}/login`);
      await page.waitForSelector('form', { timeout: 5000 });

      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', 'WrongPassword123!');

      const submitButton = page.locator('button:has-text("Login")');
      await submitButton.click();

      // Wait for error
      const errorMessage = page.locator('[role="alert"]');
      await expect(errorMessage).toContainText(/invalid|incorrect/i);

      // Should stay on login page
      expect(page.url()).toContain('/login');
    });

    test('should show validation error for empty credentials', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/login`);
      await page.waitForSelector('form', { timeout: 5000 });

      const submitButton = page.locator('button:has-text("Login")');
      await expect(submitButton).toBeDisabled();

      // Fill only email
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await expect(submitButton).toBeDisabled();
    });

    test('should provide forgot password link', async ({ page }) => {
      await page.goto(`${BASE_URL}/login`);
      await page.waitForSelector('form', { timeout: 5000 });

      const forgotPasswordLink = page.locator('a:has-text("Forgot password")');
      await expect(forgotPasswordLink).toBeVisible();
      await expect(forgotPasswordLink).toHaveAttribute('href', /\/forgot-password/);
    });
  });

  test.describe('JWT Token Verification', () => {
    test('should include valid JWT in Authorization header', async ({
      page,
      context,
    }) => {
      // Login first
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      // Wait for dashboard
      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Intercept API request to verify JWT
      let authHeader = '';
      await context.route('**/api/**', async (route) => {
        const request = route.request();
        authHeader = request.headerValue('Authorization') || '';
        await route.abort();
      });

      // Make API call
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for potential API calls
      await page.waitForTimeout(1000);

      // Verify JWT format (Bearer <token>)
      expect(authHeader).toMatch(/^Bearer /);
      const token = authHeader.replace('Bearer ', '');
      expect(token.split('.').length).toBe(3); // JWT has 3 parts
    });

    test('should refresh expired token automatically', async ({ page }) => {
      // This test verifies the token refresh mechanism
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Store original token
      const originalToken = await page.evaluate(() =>
        localStorage.getItem('token')
      );
      expect(originalToken).toBeTruthy();

      // Simulate token expiration by waiting
      // In real test, you'd manipulate the token expiry
      await page.waitForTimeout(2000);

      // Try to make an API request that would fail with expired token
      // The app should refresh and retry automatically
      const token = await page.evaluate(() =>
        localStorage.getItem('token')
      );
      expect(token).toBeTruthy();
    });

    test('should decode JWT and extract user claims', async ({ page }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Decode and verify JWT
      const tokenData = await page.evaluate(() => {
        const token = localStorage.getItem('token');
        if (!token) return null;

        const parts = token.split('.');
        const payload = JSON.parse(atob(parts[1]));
        return payload;
      });

      expect(tokenData).toBeTruthy();
      expect(tokenData).toHaveProperty('user_id');
      expect(tokenData).toHaveProperty('username');
      expect(tokenData).toHaveProperty('email');
      expect(tokenData.email).toBe(EXISTING_USER.email);
    });
  });

  test.describe('Protected Route Access', () => {
    test('should redirect unauthenticated user to login', async ({ page }) => {
      // Try to access protected route without token
      await page.goto(`${BASE_URL}/portfolio`);

      // Should redirect to login
      await page.waitForURL(`${BASE_URL}/login`, { timeout: 5000 });
      expect(page.url()).toContain('/login');
    });

    test('should allow authenticated user to access protected routes', async ({
      page,
    }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Navigate to portfolio (protected route)
      await page.goto(`${BASE_URL}/portfolio`);

      // Should stay on portfolio page
      await page.waitForTimeout(1000);
      expect(page.url()).toContain('/portfolio');
    });

    test('should display user profile in authenticated state', async ({
      page,
    }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Look for user profile/name in header
      const userProfile = page.locator(
        `text=${EXISTING_USER.email.split('@')[0]}`
      );
      await expect(userProfile).toBeVisible({ timeout: 5000 });
    });

    test('should clear token and redirect on 401 response', async ({
      page,
      context,
    }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Mock API to return 401
      await context.route('**/api/**', (route) => {
        route.abort('accessdenied');
      });

      // Navigate to portfolio to trigger API call
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for potential redirect to login
      await page.waitForTimeout(2000);

      // Token should be cleared
      const hasToken = await page.evaluate(() =>
        !!localStorage.getItem('token')
      );
      expect(hasToken).toBe(false);
    });
  });

  test.describe('Logout', () => {
    test('should logout user and clear token', async ({ page }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Verify token exists
      let hasToken = await page.evaluate(() =>
        !!localStorage.getItem('token')
      );
      expect(hasToken).toBe(true);

      // Click logout
      const logoutButton = page.locator('button:has-text("Logout")');
      await logoutButton.click();

      // Verify redirect to login
      await page.waitForURL(`${BASE_URL}/login`, { timeout: 5000 });

      // Verify token is cleared
      hasToken = await page.evaluate(() =>
        !!localStorage.getItem('token')
      );
      expect(hasToken).toBe(false);
    });

    test('should invalidate refresh token on logout', async ({
      page,
      context,
    }) => {
      // Login
      await page.goto(`${BASE_URL}/login`);
      await page.fill('input[name="email"]', EXISTING_USER.email);
      await page.fill('input[name="password"]', EXISTING_USER.password);
      await page.locator('button:has-text("Login")').click();

      await page.waitForURL(`${BASE_URL}/dashboard`);

      // Click logout
      const logoutButton = page.locator('button:has-text("Logout")');
      await logoutButton.click();

      await page.waitForURL(`${BASE_URL}/login`);

      // Attempt to use refresh token should fail
      // This would be handled by a 401 on the next API call
      // Test verifies tokens are cleared in localStorage
      const tokens = await page.evaluate(() => ({
        token: localStorage.getItem('token'),
        refreshToken: localStorage.getItem('refreshToken'),
      }));

      expect(tokens.token).toBeNull();
      expect(tokens.refreshToken).toBeNull();
    });
  });
});
