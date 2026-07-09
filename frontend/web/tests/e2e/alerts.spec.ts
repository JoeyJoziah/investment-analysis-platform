import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Alerts flows', () => {
  test('loads alerts page', async ({ page }) => {
    await gotoAuthed(page, '/alerts');
    await expectAppShell(page);
    await expect(page).toHaveURL(/alerts/);
  });
});
