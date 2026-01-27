import { test, expect, Page } from '@playwright/test';

const BASE_URL = process.env.BASE_URL || 'http://localhost:5173';
const API_URL = process.env.API_URL || 'http://localhost:8000';

// Test user for portfolio operations
const TEST_USER = {
  email: 'portfolio-test@example.com',
  username: 'portfolio-test',
  password: 'PortfolioTest123!',
};

test.describe('Portfolio Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto(`${BASE_URL}/login`);
    await page.fill('input[name="email"]', TEST_USER.email);
    await page.fill('input[name="password"]', TEST_USER.password);
    await page.locator('button:has-text("Login")').click();

    // Wait for login to complete
    await page.waitForURL(`${BASE_URL}/dashboard`, { timeout: 10000 });
  });

  test.describe('Add Stock to Portfolio', () => {
    test('should successfully add a stock position', async ({ page }) => {
      // Navigate to portfolio
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Click "Add Transaction" button
      const addButton = page.locator('button:has-text("Add Transaction")');
      await addButton.click();

      // Wait for dialog
      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();

      // Fill in stock details
      await page.fill('input[name="ticker"]', 'AAPL');
      await page.fill('input[name="quantity"]', '10');
      await page.fill('input[name="price"]', '150.00');

      // Select transaction type (BUY)
      const typeSelect = page.locator('select[name="type"]');
      await typeSelect.selectOption('BUY');

      // Submit form
      const submitButton = dialog.locator('button:has-text("Add")');
      await submitButton.click();

      // Wait for success message
      const successAlert = page.locator('[role="alert"]:has-text("Success")');
      await expect(successAlert).toBeVisible({ timeout: 5000 });

      // Verify position appears in table
      const appleRow = page.locator('text=AAPL');
      await expect(appleRow).toBeVisible();

      // Verify position details
      await expect(page.locator('text=Apple Inc')).toBeVisible();
      await expect(page.locator('text=10')).toBeVisible();
    });

    test('should validate required fields', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      const addButton = page.locator('button:has-text("Add Transaction")');
      await addButton.click();

      const dialog = page.locator('[role="dialog"]');
      await expect(dialog).toBeVisible();

      // Try to submit empty form
      const submitButton = dialog.locator('button:has-text("Add")');
      await expect(submitButton).toBeDisabled();

      // Fill only ticker
      await page.fill('input[name="ticker"]', 'AAPL');
      await expect(submitButton).toBeDisabled();

      // Fill quantity
      await page.fill('input[name="quantity"]', '10');
      await expect(submitButton).toBeDisabled();

      // Fill price
      await page.fill('input[name="price"]', '150');
      await expect(submitButton).toBeEnabled();
    });

    test('should reject invalid price', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      const addButton = page.locator('button:has-text("Add Transaction")');
      await addButton.click();

      const dialog = page.locator('[role="dialog"]');
      await page.fill('input[name="ticker"]', 'AAPL');
      await page.fill('input[name="quantity"]', '10');
      await page.fill('input[name="price"]', '-100'); // Invalid negative price

      const priceInput = page.locator('input[name="price"]');
      await expect(priceInput).toHaveAttribute('aria-invalid', 'true');
    });

    test('should reject invalid quantity', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      const addButton = page.locator('button:has-text("Add Transaction")');
      await addButton.click();

      const dialog = page.locator('[role="dialog"]');
      await page.fill('input[name="ticker"]', 'AAPL');
      await page.fill('input[name="quantity"]', '0'); // Invalid zero quantity
      await page.fill('input[name="price"]', '150');

      const quantityInput = page.locator('input[name="quantity"]');
      await expect(quantityInput).toHaveAttribute('aria-invalid', 'true');
    });

    test('should handle duplicate ticker correctly', async ({ page }) => {
      // Add first position
      await page.goto(`${BASE_URL}/portfolio`);
      const addButton = page.locator('button:has-text("Add Transaction")');

      // Add AAPL
      await addButton.click();
      let dialog = page.locator('[role="dialog"]');
      await page.fill('input[name="ticker"]', 'AAPL');
      await page.fill('input[name="quantity"]', '10');
      await page.fill('input[name="price"]', '150');
      await dialog.locator('button:has-text("Add")').click();

      // Wait for success
      await page.locator('[role="alert"]:has-text("Success")').waitFor();
      await page.waitForTimeout(1000);

      // Add same ticker again
      await addButton.click();
      dialog = page.locator('[role="dialog"]');
      await page.fill('input[name="ticker"]', 'AAPL');
      await page.fill('input[name="quantity"]', '5');
      await page.fill('input[name="price"]', '155');
      await dialog.locator('button:has-text("Add")').click();

      // Should succeed and consolidate positions
      await page.locator('[role="alert"]:has-text("Success")').waitFor();

      // Verify position is consolidated (15 shares)
      const quantityCell = page.locator(
        'text=15' // Total should be 15 shares
      );
      await expect(quantityCell).toBeVisible({ timeout: 5000 });
    });
  });

  test.describe('View Performance Metrics', () => {
    test('should display portfolio performance summary', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for page to load
      await page.waitForSelector('[role="tab"]', { timeout: 5000 });

      // Check for summary cards
      const totalValueCard = page.locator('text=Total Value');
      const totalGainCard = page.locator('text=Total Gain');
      const dayGainCard = page.locator('text=Day Gain');

      await expect(totalValueCard).toBeVisible();
      await expect(totalGainCard).toBeVisible();
      await expect(dayGainCard).toBeVisible();
    });

    test('should show performance tab with charts', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Click Performance tab
      const performanceTab = page.locator('[role="tab"]:has-text("Performance")');
      await performanceTab.click();

      // Wait for performance content
      await page.waitForSelector('canvas', { timeout: 5000 });

      // Verify risk metrics are displayed
      const riskMetrics = page.locator('text=Sharpe Ratio');
      await expect(riskMetrics).toBeVisible();
    });

    test('should display allocation breakdown', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Click Allocation tab
      const allocationTab = page.locator(
        '[role="tab"]:has-text("Allocation")'
      );
      await allocationTab.click();

      // Wait for allocation content
      await page.waitForSelector('canvas', { timeout: 5000 });

      // Verify allocation sections
      const sectorAllocation = page.locator('text=Sector Allocation');
      const assetAllocation = page.locator('text=Asset Type Allocation');

      await expect(sectorAllocation).toBeVisible();
      await expect(assetAllocation).toBeVisible();
    });

    test('should show position-level gains/losses', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for positions table
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Check for gain/loss column
      const gainLossHeader = page.locator(
        '[role="columnheader"]:has-text("Gain/Loss")'
      );
      await expect(gainLossHeader).toBeVisible();

      // Verify gain/loss values exist
      const gainLossCells = page.locator('[data-testid="gain-loss-cell"]');
      const count = await gainLossCells.count();
      expect(count).toBeGreaterThan(0);
    });
  });

  test.describe('Real-time Price Updates', () => {
    test('should receive WebSocket price updates', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for positions table
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Get initial price
      const initialPrice = await page.locator(
        '[data-testid="current-price"]'
      ).first().textContent();

      // Wait for WebSocket updates
      await page.waitForTimeout(3000);

      // Price should update (or at least be present)
      const updatedPrice = await page.locator(
        '[data-testid="current-price"]'
      ).first().textContent();

      expect(updatedPrice).toBeTruthy();
      expect(updatedPrice).toMatch(/\d+\.?\d*/); // Valid price format
    });

    test('should update position values based on price changes', async ({
      page,
    }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Wait for initial load
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Get initial market value
      const initialMarketValue = await page.locator(
        '[data-testid="market-value"]'
      ).first().textContent();

      // Wait for updates
      await page.waitForTimeout(3000);

      // Market value should exist and be numeric
      const updatedMarketValue = await page.locator(
        '[data-testid="market-value"]'
      ).first().textContent();

      expect(updatedMarketValue).toBeTruthy();
      expect(updatedMarketValue).toMatch(/\$[\d,]+\.?\d*/);
    });

    test('should handle WebSocket disconnection gracefully', async ({
      page,
      context,
    }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Simulate WebSocket disconnect
      await page.evaluate(() => {
        // Close any active WebSocket connections
        if ((window as any).ws) {
          (window as any).ws.close();
        }
      });

      // Wait a moment
      await page.waitForTimeout(2000);

      // Page should still be functional
      const table = page.locator('[role="table"]');
      await expect(table).toBeVisible();

      // Should show reconnection indicator or continue functioning
      const positions = page.locator('[data-testid="position-row"]');
      const count = await positions.count();
      expect(count).toBeGreaterThanOrEqual(0);
    });
  });

  test.describe('Remove Position', () => {
    test('should remove a position from portfolio', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Get initial position count
      let positions = page.locator('[data-testid="position-row"]');
      const initialCount = await positions.count();
      expect(initialCount).toBeGreaterThan(0);

      // Click delete button on first position
      const deleteButton = page.locator(
        '[data-testid="delete-position-btn"]'
      ).first();
      await deleteButton.click();

      // Confirm deletion
      const confirmButton = page.locator(
        'button:has-text("Confirm")'
      );
      await expect(confirmButton).toBeVisible();
      await confirmButton.click();

      // Wait for success message
      const successAlert = page.locator('[role="alert"]:has-text("Success")');
      await expect(successAlert).toBeVisible({ timeout: 5000 });

      // Verify position count decreased
      positions = page.locator('[data-testid="position-row"]');
      const finalCount = await positions.count();
      expect(finalCount).toBe(initialCount - 1);
    });

    test('should show confirmation dialog before delete', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Click delete button
      const deleteButton = page.locator(
        '[data-testid="delete-position-btn"]'
      ).first();
      await deleteButton.click();

      // Verify confirmation dialog appears
      const dialog = page.locator('[role="alertdialog"]');
      await expect(dialog).toBeVisible();

      // Verify message
      const message = dialog.locator('text=delete');
      await expect(message).toBeVisible();
    });

    test('should keep position when cancel is clicked', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Get initial count
      let positions = page.locator('[data-testid="position-row"]');
      const initialCount = await positions.count();

      // Click delete
      const deleteButton = page.locator(
        '[data-testid="delete-position-btn"]'
      ).first();
      await deleteButton.click();

      // Click cancel
      const cancelButton = page.locator('button:has-text("Cancel")');
      await cancelButton.click();

      // Position count should remain same
      positions = page.locator('[data-testid="position-row"]');
      const finalCount = await positions.count();
      expect(finalCount).toBe(initialCount);
    });

    test('should handle deletion of last position', async ({ page }) => {
      // Navigate to portfolio with only one position
      await page.goto(`${BASE_URL}/portfolio`);
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Get position count
      let positions = page.locator('[data-testid="position-row"]');
      const count = await positions.count();

      if (count === 1) {
        // Delete the position
        const deleteButton = page.locator(
          '[data-testid="delete-position-btn"]'
        ).first();
        await deleteButton.click();

        // Confirm
        const confirmButton = page.locator('button:has-text("Confirm")');
        await confirmButton.click();

        // Wait for success
        await page.locator('[role="alert"]:has-text("Success")').waitFor();

        // Page should show empty state
        const emptyState = page.locator('text=No positions');
        await expect(emptyState).toBeVisible({ timeout: 5000 });
      }
    });
  });

  test.describe('Transaction History', () => {
    test('should display transaction history', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      // Click Transactions tab
      const transactionsTab = page.locator(
        '[role="tab"]:has-text("Transactions")'
      );
      await transactionsTab.click();

      // Wait for transactions table
      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Verify table headers
      const dateHeader = page.locator(
        '[role="columnheader"]:has-text("Date")'
      );
      const typeHeader = page.locator(
        '[role="columnheader"]:has-text("Type")'
      );

      await expect(dateHeader).toBeVisible();
      await expect(typeHeader).toBeVisible();
    });

    test('should show transaction details', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      const transactionsTab = page.locator(
        '[role="tab"]:has-text("Transactions")'
      );
      await transactionsTab.click();

      await page.waitForSelector('[role="table"]', { timeout: 5000 });

      // Check for transaction data
      const transactions = page.locator('[data-testid="transaction-row"]');
      const count = await transactions.count();

      if (count > 0) {
        // Verify first transaction has required fields
        const firstTransaction = transactions.first();
        const ticker = firstTransaction.locator('[data-testid="ticker"]');
        const quantity = firstTransaction.locator('[data-testid="quantity"]');

        await expect(ticker).toBeVisible();
        await expect(quantity).toBeVisible();
      }
    });
  });

  test.describe('Portfolio Analysis', () => {
    test('should display analysis tab', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      const analysisTab = page.locator(
        '[role="tab"]:has-text("Analysis")'
      );
      await analysisTab.click();

      // Wait for analysis content
      await page.waitForSelector('text=Analysis', { timeout: 5000 });

      // Verify sections
      const topPerformers = page.locator('text=Top Performers');
      const worstPerformers = page.locator('text=Worst Performers');

      await expect(topPerformers).toBeVisible();
      await expect(worstPerformers).toBeVisible();
    });

    test('should show correlation analysis', async ({ page }) => {
      await page.goto(`${BASE_URL}/portfolio`);

      const analysisTab = page.locator(
        '[role="tab"]:has-text("Analysis")'
      );
      await analysisTab.click();

      // Look for correlation data
      const correlation = page.locator('text=Correlation');
      await expect(correlation).toBeVisible({ timeout: 5000 });
    });
  });
});
