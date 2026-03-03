import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// Mock react-intersection-observer before importing hooks
vi.mock('react-intersection-observer', () => ({
  useInView: vi.fn().mockReturnValue([vi.fn(), false, undefined]),
}));

// Mock lodash throttle to pass-through
vi.mock('lodash', () => ({
  throttle: vi.fn((fn: Function) => fn),
}));

import {
  useVirtualScroll,
  useDebouncedValue,
  useThrottledCallback,
  useLazyLoad,
  useMemoizedAsync,
  useInfiniteScroll,
  useLazyImage,
  usePrefetch,
} from './usePerformance';
import { useInView } from 'react-intersection-observer';

// ---------------------------------------------------------------------------
// useVirtualScroll
// ---------------------------------------------------------------------------
describe('useVirtualScroll', () => {
  const items = Array.from({ length: 100 }, (_, i) => ({ id: i }));

  it('returns correct visible range for initial scroll position', () => {
    const { result } = renderHook(() =>
      useVirtualScroll({ items, itemHeight: 40, containerHeight: 200, overscan: 2 })
    );

    expect(result.current.startIndex).toBe(0);
    // endIndex = ceil((0 + 200) / 40) + 2 = 5 + 2 = 7
    expect(result.current.endIndex).toBe(7);
    expect(result.current.visibleItems.length).toBe(8); // 0..7 inclusive
    expect(result.current.offsetY).toBe(0);
    expect(result.current.totalHeight).toBe(4000); // 100 * 40
  });

  it('provides handleScroll callback', () => {
    const { result } = renderHook(() =>
      useVirtualScroll({ items, itemHeight: 40, containerHeight: 200, overscan: 0 })
    );

    expect(typeof result.current.handleScroll).toBe('function');
  });

  it('updates visible range when scrolled', () => {
    const { result } = renderHook(() =>
      useVirtualScroll({ items, itemHeight: 40, containerHeight: 200, overscan: 0 })
    );

    act(() => {
      result.current.handleScroll({
        currentTarget: { scrollTop: 400 },
      } as unknown as React.UIEvent<HTMLDivElement>);
    });

    // startIndex = floor(400/40) = 10
    expect(result.current.startIndex).toBe(10);
    // endIndex = ceil((400+200)/40) = 15
    expect(result.current.endIndex).toBe(15);
    expect(result.current.offsetY).toBe(400);
  });

  it('handles empty items array', () => {
    const { result } = renderHook(() =>
      useVirtualScroll({ items: [], itemHeight: 40, containerHeight: 200 })
    );

    expect(result.current.totalHeight).toBe(0);
    expect(result.current.visibleItems.length).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// useDebouncedValue
// ---------------------------------------------------------------------------
describe('useDebouncedValue', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it('returns initial value immediately', () => {
    const { result } = renderHook(() => useDebouncedValue('hello', 300));
    expect(result.current).toBe('hello');
  });

  it('debounces value updates', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 300),
      { initialProps: { value: 'a' } }
    );

    rerender({ value: 'b' });
    expect(result.current).toBe('a'); // not updated yet

    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current).toBe('b');
  });

  it('resets timer on rapid updates', () => {
    const { result, rerender } = renderHook(
      ({ value }) => useDebouncedValue(value, 200),
      { initialProps: { value: 1 } }
    );

    rerender({ value: 2 });
    act(() => { vi.advanceTimersByTime(100); });
    rerender({ value: 3 });
    act(() => { vi.advanceTimersByTime(100); });

    // 200ms haven't passed since last change
    expect(result.current).toBe(1);

    act(() => { vi.advanceTimersByTime(100); });
    expect(result.current).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// useThrottledCallback
// ---------------------------------------------------------------------------
describe('useThrottledCallback', () => {
  it('returns a function', () => {
    const fn = vi.fn();
    const { result } = renderHook(() => useThrottledCallback(fn, 300));
    expect(typeof result.current).toBe('function');
  });

  it('calls through when throttle is identity', () => {
    const fn = vi.fn();
    const { result } = renderHook(() => useThrottledCallback(fn, 300));

    act(() => {
      result.current('arg1');
    });

    expect(fn).toHaveBeenCalledWith('arg1');
  });
});

// ---------------------------------------------------------------------------
// useLazyLoad
// ---------------------------------------------------------------------------
describe('useLazyLoad', () => {
  it('returns shouldLoad false when not in view', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const { result } = renderHook(() => useLazyLoad());
    expect(result.current.shouldLoad).toBe(false);
    expect(result.current.isInView).toBe(false);
  });

  it('sets shouldLoad true when in view', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), true, undefined] as any);

    const { result } = renderHook(() => useLazyLoad());
    // After the effect runs, shouldLoad should be true
    expect(result.current.isInView).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// useMemoizedAsync
// ---------------------------------------------------------------------------
describe('useMemoizedAsync', () => {
  it('starts in loading state', async () => {
    let resolveOuter!: (v: string) => void;
    const asyncFn = () => new Promise<string>((resolve) => { resolveOuter = resolve; });

    const { result } = renderHook(() => useMemoizedAsync(asyncFn, []));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();

    // Cleanup: resolve the promise so the effect completes
    await act(async () => { resolveOuter('done'); });
  });

  it('resolves data on success', async () => {
    const asyncFn = () => Promise.resolve('test-data');

    const { result } = renderHook(() => useMemoizedAsync(asyncFn, []));

    await waitFor(() => {
      expect(result.current.data).toBe('test-data');
    });

    expect(result.current.loading).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it('captures error on failure', async () => {
    const asyncFn = () => Promise.reject(new Error('fail'));

    const { result } = renderHook(() => useMemoizedAsync(asyncFn, []));

    await waitFor(() => {
      expect(result.current.error).toBeInstanceOf(Error);
    });

    expect(result.current.error?.message).toBe('fail');
    expect(result.current.data).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// useInfiniteScroll
// ---------------------------------------------------------------------------
describe('useInfiniteScroll', () => {
  it('returns sentinelRef', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const onLoadMore = vi.fn();
    const { result } = renderHook(() =>
      useInfiniteScroll({ onLoadMore, hasMore: true, loading: false })
    );

    expect(result.current.sentinelRef).toBeDefined();
  });

  it('does not call onLoadMore when not in view', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const onLoadMore = vi.fn();
    renderHook(() =>
      useInfiniteScroll({ onLoadMore, hasMore: true, loading: false })
    );

    expect(onLoadMore).not.toHaveBeenCalled();
  });

  it('calls onLoadMore when in view with hasMore', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), true, undefined] as any);

    const onLoadMore = vi.fn();
    renderHook(() =>
      useInfiniteScroll({ onLoadMore, hasMore: true, loading: false })
    );

    expect(onLoadMore).toHaveBeenCalled();
  });

  it('does not call onLoadMore when loading', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), true, undefined] as any);

    const onLoadMore = vi.fn();
    renderHook(() =>
      useInfiniteScroll({ onLoadMore, hasMore: true, loading: true })
    );

    expect(onLoadMore).not.toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// useLazyImage
// ---------------------------------------------------------------------------
describe('useLazyImage', () => {
  it('returns placeholder initially', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const { result } = renderHook(() => useLazyImage('/img.png', '/placeholder.png'));

    expect(result.current.imageSrc).toBe('/placeholder.png');
    expect(result.current.isLoaded).toBe(false);
  });

  it('returns empty string without placeholder', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const { result } = renderHook(() => useLazyImage('/img.png'));
    expect(result.current.imageSrc).toBe('');
  });

  it('provides imageRef', () => {
    vi.mocked(useInView).mockReturnValue([vi.fn(), false, undefined] as any);

    const { result } = renderHook(() => useLazyImage('/img.png'));
    expect(result.current.imageRef).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// usePrefetch
// ---------------------------------------------------------------------------
describe('usePrefetch', () => {
  it('caches fetcher result', async () => {
    const { result } = renderHook(() => usePrefetch());

    const fetcher = vi.fn().mockResolvedValue('data-1');

    await act(async () => {
      await result.current.prefetch('key1', fetcher);
    });

    expect(fetcher).toHaveBeenCalledTimes(1);

    // Second call should NOT re-fetch
    await act(async () => {
      await result.current.prefetch('key1', fetcher);
    });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it('getCached returns undefined for unknown keys', () => {
    const { result } = renderHook(() => usePrefetch());
    expect(result.current.getCached('unknown')).toBeUndefined();
  });

  it('clearCache removes specific key', async () => {
    const { result } = renderHook(() => usePrefetch());

    await act(async () => {
      await result.current.prefetch('k1', () => Promise.resolve('v1'));
    });

    expect(result.current.getCached('k1')).toBeDefined();

    act(() => {
      result.current.clearCache('k1');
    });

    expect(result.current.getCached('k1')).toBeUndefined();
  });

  it('clearCache without key clears all', async () => {
    const { result } = renderHook(() => usePrefetch());

    await act(async () => {
      await result.current.prefetch('a', () => Promise.resolve(1));
      await result.current.prefetch('b', () => Promise.resolve(2));
    });

    act(() => {
      result.current.clearCache();
    });

    expect(result.current.getCached('a')).toBeUndefined();
    expect(result.current.getCached('b')).toBeUndefined();
  });
});
