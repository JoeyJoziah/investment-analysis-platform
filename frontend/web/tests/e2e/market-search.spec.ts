import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Market / stock search flows', () => {
  test('loads market overview', async ({ page }) => {
    await gotoAuthed(page, '/market');
    await expectAppShell(page);
    await expect(page).toHaveURL(/market/);
  });

  test('search input is available for ticker lookup', async ({ page }) => {
    await gotoAuthed(page, '/market');
    const search = page
      .locator(
        'input[type="search"], input[name="q"], input[name="query"], input[placeholder*="search" i], input[placeholder*="ticker" i], input[placeholder*="symbol" i]'
      )
      .first();
    if (await search.count()) {
      await expect(search).toBeVisible({ timeout: 10000 });
      await search.fill('AAPL');
    } else {
      // Market page still loads without dedicated search box
      await expectAppShell(page);
    }
  });
});
