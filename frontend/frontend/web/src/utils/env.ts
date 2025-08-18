/**
 * Environment variable utility for Vite migration
 * Provides backward compatibility for REACT_APP_ prefixed variables
 */

// Helper function to get environment variables with fallback support
export const getEnvVar = (key: string): string | undefined => {
  // Try Vite format first (import.meta.env.VITE_*)
  if (import.meta.env && import.meta.env[`VITE_${key}`]) {
    return import.meta.env[`VITE_${key}`];
  }
  
  // Fallback to REACT_APP_ format for backward compatibility
  if (import.meta.env && import.meta.env[`REACT_APP_${key}`]) {
    return import.meta.env[`REACT_APP_${key}`];
  }
  
  // Also check process.env for compatibility (though Vite doesn't use it)
  if (typeof process !== 'undefined' && process.env && process.env[`REACT_APP_${key}`]) {
    return process.env[`REACT_APP_${key}`];
  }
  
  return undefined;
};

// Export commonly used environment variables
export const env = {
  API_URL: getEnvVar('API_URL') || 'http://localhost:8000',
  WS_URL: getEnvVar('WS_URL') || 'ws://localhost:8000/api/ws',
  APP_NAME: getEnvVar('APP_NAME') || 'Investment Analysis Platform',
  APP_VERSION: getEnvVar('APP_VERSION') || '1.0.0',
  ENABLE_WEBSOCKETS: getEnvVar('ENABLE_WEBSOCKETS') === 'true',
  ENABLE_ANALYTICS: getEnvVar('ENABLE_ANALYTICS') === 'true',
  ENABLE_DEBUG: getEnvVar('ENABLE_DEBUG') === 'true',
};

// Export the raw environment for direct access if needed
export const rawEnv = import.meta.env;