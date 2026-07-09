/**
 * Optional frontend Sentry bootstrap (#102).
 *
 * Activates only when VITE_SENTRY_DSN is set. Uses dynamic import so the app
 * builds without a hard @sentry/* dependency; install @sentry/react (or
 * @sentry/browser) and set the DSN to enable production error tracking.
 */

const dsn = (import.meta.env.VITE_SENTRY_DSN as string | undefined)?.trim();

export async function initFrontendSentry(): Promise<boolean> {
  if (!dsn || dsn.startsWith('optional_')) {
    return false;
  }

  try {
    // Dynamic string import keeps Sentry an optional peer dependency
    const load = (name: string) =>
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (new Function('n', 'return import(n)') as (n: string) => Promise<any>)(name);

    const sentryMod = await load('@sentry/react').catch(() =>
      load('@sentry/browser')
    );

    const init = sentryMod.init as (opts: Record<string, unknown>) => void;
    init({
      dsn,
      environment:
        (import.meta.env.VITE_SENTRY_ENVIRONMENT as string | undefined) ||
        (import.meta.env.MODE as string) ||
        'development',
      tracesSampleRate: Number(
        (import.meta.env.VITE_SENTRY_TRACES_SAMPLE_RATE as string) || 0.1
      ),
    });
    return true;
  } catch {
    // Package not installed or init failed — app continues normally
    if (import.meta.env.DEV) {
      // eslint-disable-next-line no-console
      console.info(
        '[sentry] DSN set but Sentry SDK not available; install @sentry/react to enable'
      );
    }
    return false;
  }
}
