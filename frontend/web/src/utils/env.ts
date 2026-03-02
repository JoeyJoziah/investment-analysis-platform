/**
 * Environment variable utility for Vite
 * Uses VITE_ prefix as required by the Vite bundler
 */

// Export commonly used environment variables using Vite's import.meta.env
export const env = {
  API_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  WS_URL: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/ws',
  APP_NAME: import.meta.env.VITE_APP_NAME || 'Investment Analysis Platform',
  APP_VERSION: import.meta.env.VITE_APP_VERSION || '1.0.0',
  ENABLE_WEBSOCKETS: import.meta.env.VITE_ENABLE_WEBSOCKETS === 'true',
  ENABLE_ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  ENABLE_DEBUG: import.meta.env.VITE_ENABLE_DEBUG === 'true',
};

// Export the raw environment for direct access if needed
export const rawEnv = import.meta.env;
