import { defineConfig, devices } from '@playwright/test';

/**
 * Playwright configuration for E2E testing
 * https://playwright.dev/docs/intro
 */
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [
    ['html'],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['json', { outputFile: 'test-results/results.json' }],
  ],

  use: {
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 30000,
    navigationTimeout: 30000,
  },

  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
  ],

  webServer: [
    {
      command: 'npm run dev',
      url: 'http://localhost:3000',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
    {
      command: 'cd ../../ && python -m uvicorn backend.api.main:app --reload --port 8000',
      url: 'http://localhost:8000/api/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
      // F8-15-012: backend.config.settings instantiates at import and
      // hard-requires these; without them uvicorn never binds and the run
      // dies as a webServer timeout instead of a test failure. Values
      // mirror ci.yml's test env; real env vars win when set. The API also
      // needs Postgres/Redis running (see the e2e-tests CI job / README).
      env: {
        DATABASE_URL:
          process.env.DATABASE_URL || 'postgresql://postgres:testpass@localhost:5432/test_db',
        REDIS_URL: process.env.REDIS_URL || 'redis://localhost:6379/0',
        SECRET_KEY: process.env.SECRET_KEY || 'test-secret-key-for-ci',
        JWT_SECRET_KEY: process.env.JWT_SECRET_KEY || 'test-jwt-secret-key-for-ci',
        MASTER_SECRET_KEY: process.env.MASTER_SECRET_KEY || 'test-master-secret-key-for-ci',
        SESSION_SECRET_KEY: process.env.SESSION_SECRET_KEY || 'test-session-secret-key-for-ci',
        ENVIRONMENT: process.env.ENVIRONMENT || 'testing',
        TESTING: process.env.TESTING || 'True',
      },
    },
  ],
});
