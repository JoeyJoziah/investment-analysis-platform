import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Settings flows', () => {
  test('loads settings page', async ({ page }) => {
    await gotoAuthed(page, '/settings');
    await expectAppShell(page);
    await expect(page).toHaveURL(/settings/);
    const marker = page.locator('h1, h2, text=/settings|preferences|profile/i').first();
    await expect(marker).toBeVisible({ timeout: 15000 });
  });
});
