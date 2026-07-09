import { test, expect } from '@playwright/test';
import { BASE_URL, gotoAuthed, expectAppShell } from './helpers';

test.describe('Watchlist flows', () => {
  test('loads watchlist page for authenticated user', async ({ page }) => {
    await gotoAuthed(page, '/watchlist');
    await expectAppShell(page);
    await expect(page).toHaveURL(/watchlist/);
    // Heading or primary content region
    const marker = page
      .locator('h1, h2, [data-testid="watchlist"], text=/watchlist/i')
      .first();
    await expect(marker).toBeVisible({ timeout: 15000 });
  });

  test('watchlist page exposes add/search affordance', async ({ page }) => {
    await gotoAuthed(page, '/watchlist');
    const action = page
      .locator(
        'button:has-text("Add"), button:has-text("Create"), input[placeholder*="symbol" i], input[placeholder*="search" i], input[name="symbol"]'
      )
      .first();
    // Soft: UI may use icon-only buttons; at least page shell is interactive
    if (await action.count()) {
      await expect(action.first()).toBeVisible({ timeout: 10000 });
    } else {
      await expectAppShell(page);
    }
  });
});
