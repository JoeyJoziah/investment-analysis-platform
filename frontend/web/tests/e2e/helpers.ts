import { Page, expect } from '@playwright/test';

export const BASE_URL = process.env.BASE_URL || 'http://localhost:3000';
export const API_URL = process.env.API_URL || 'http://localhost:8000';

export const E2E_USER = {
  email: process.env.E2E_USER_EMAIL || 'portfolio-test@example.com',
  password: process.env.E2E_USER_PASSWORD || 'PortfolioTest123!',
};

/** Login via UI and land on dashboard (or stay if already authenticated). */
export async function loginAsTestUser(page: Page): Promise<void> {
  await page.goto(`${BASE_URL}/login`);
  const email = page.locator('input[name="email"], input[type="email"]').first();
  if (await email.isVisible({ timeout: 5000 }).catch(() => false)) {
    await email.fill(E2E_USER.email);
    await page
      .locator('input[name="password"], input[type="password"]')
      .first()
      .fill(E2E_USER.password);
    await page
      .locator('button:has-text("Sign In"), button:has-text("Login"), button[type="submit"]')
      .first()
      .click();
  }
  await page.waitForURL(/dashboard|portfolio|watchlist|\/$/, { timeout: 30000 });
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
