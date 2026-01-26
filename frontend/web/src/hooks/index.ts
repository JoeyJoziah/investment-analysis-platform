/**
 * Custom React Hooks
 *
 * Application-wide hooks for the investment analysis platform.
 */

export { useAppDispatch, useAppSelector } from './redux';
export { default as usePerformance } from './usePerformance';
export { default as useRealTimePrices } from './useRealTimePrices';

// Re-export types
export type { default as UseRealTimePricesOptions } from './useRealTimePrices';
