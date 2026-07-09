import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Dashboard flows', () => {
  test('loads dashboard after authentication', async ({ page }) => {
    await gotoAuthed(page, '/dashboard');
    await expectAppShell(page);
    await expect(page).toHaveURL(/dashboard/);
  });

  test('dashboard shows primary widgets or empty state', async ({ page }) => {
    await gotoAuthed(page, '/dashboard');
    const content = page
      .locator(
        'h1, h2, [data-testid="dashboard"], text=/portfolio|market|recommend|overview/i'
      )
      .first();
    await expect(content).toBeVisible({ timeout: 15000 });
  });
});
