import { Page, expect, request as pwRequest } from '@playwright/test';

export const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
export const API_URL = process.env.API_URL || 'http://localhost:8000';

// F8-15-022: no committed credential literal. The password must come from
// the environment; failing fast here beats a suite that silently runs
// unauthenticated. CI sets E2E_USER_PASSWORD in the e2e-tests job.
function requiredPassword(): string {
  const value = process.env.E2E_USER_PASSWORD;
  if (!value) {
    throw new Error(
      'E2E_USER_PASSWORD is required (no committed default). ' +
        'Set E2E_USER_EMAIL/E2E_USER_PASSWORD before running the E2E suite.',
    );
  }
  return value;
}

export const E2E_USER = {
  email: process.env.E2E_USER_EMAIL || 'portfolio-test@example.com',
  get password(): string {
    return requiredPassword();
  },
};

/**
 * Ensure the E2E user exists (idempotent). Registers via the API; a 400
 * "already registered" is success. Anything else is a real failure.
 */
export async function ensureTestUser(): Promise<void> {
  const api = await pwRequest.newContext({ baseURL: API_URL });
  try {
    const res = await api.post('/api/v1/auth/register', {
      data: { email: E2E_USER.email, password: E2E_USER.password },
    });
    if (!res.ok() && res.status() !== 400) {
      throw new Error(
        `E2E user registration failed: HTTP ${res.status()} ${await res.text()}`,
      );
    }
  } finally {
    await api.dispose();
  }
}

/**
 * Login via UI and land on an authenticated route.
 *
 * F8-15-013: a missing login form THROWS (no silent degrade), the site
 * root is not accepted as a post-login URL, and a real auth signal
 * (access_token in localStorage) is asserted before returning.
 */
export async function loginAsTestUser(page: Page): Promise<void> {
  await ensureTestUser();
  await page.goto(`${BASE_URL}/login`);
  const email = page.locator('input[name="email"], input[type="email"]').first();
  await email.waitFor({ state: 'visible', timeout: 10000 });
  await email.fill(E2E_USER.email);
  await page
    .locator('input[name="password"], input[type="password"]')
    .first()
    .fill(E2E_USER.password);
  await page
    .locator('button:has-text("Sign In"), button:has-text("Login"), button[type="submit"]')
    .first()
    .click();
  await page.waitForURL(/dashboard|portfolio|watchlist/, { timeout: 30000 });
  const hasToken = await page.evaluate(() => !!localStorage.getItem('access_token'));
  if (!hasToken) {
    throw new Error('login navigated but produced no access_token — auth is broken');
  }
}

/** Navigate to a protected route after login. */
export async function gotoAuthed(page: Page, path: string): Promise<void> {
  await loginAsTestUser(page);
  await page.goto(`${BASE_URL}${path.startsWith('/') ? path : `/${path}`}`);
  await page.waitForLoadState('domcontentloaded');
}

/** Soft assert that the page is not a raw 404 shell. */
export async function expectAppShell(page: Page): Promise<void> {
  await expect(page.locator('body')).toBeVisible();
  // Layout should render something interactive (nav/main)
  const shell = page.locator('nav, main, [role="main"], header').first();
  await expect(shell).toBeVisible({ timeout: 15000 });
}
