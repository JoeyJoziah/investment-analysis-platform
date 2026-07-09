import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Recommendations flows', () => {
  test('loads recommendations page', async ({ page }) => {
    await gotoAuthed(page, '/recommendations');
    await expectAppShell(page);
    await expect(page).toHaveURL(/recommendations/);
    const marker = page
      .locator('h1, h2, text=/recommend|buy|hold|sell|signals/i')
      .first();
    await expect(marker).toBeVisible({ timeout: 15000 });
  });
});
