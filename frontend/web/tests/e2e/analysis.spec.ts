import { test, expect } from '@playwright/test';
import { gotoAuthed, expectAppShell } from './helpers';

test.describe('Analysis view flows', () => {
  test('loads analysis page', async ({ page }) => {
    await gotoAuthed(page, '/analysis');
    await expectAppShell(page);
    await expect(page).toHaveURL(/analysis/);
  });

  test('loads analysis for a ticker path', async ({ page }) => {
    await gotoAuthed(page, '/analysis/AAPL');
    await expectAppShell(page);
    await expect(page).toHaveURL(/analysis/i);
  });
});
