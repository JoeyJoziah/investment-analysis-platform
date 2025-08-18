/**
 * Performance optimization hooks for the investment platform
 * Includes memoization, virtual scrolling, and lazy loading utilities
 */

import { 
  useCallback, 
  useEffect, 
  useMemo, 
  useRef, 
  useState,
  DependencyList 
} from 'react';
import { useInView } from 'react-intersection-observer';
import { debounce, throttle } from 'lodash';

/**
 * Virtual scrolling hook for large lists
 */
export const useVirtualScroll = <T,>({
  items,
  itemHeight,
  containerHeight,
  overscan = 3,
}: {
  items: T[];
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  
  const visibleRange = useMemo(() => {
    const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const endIndex = Math.min(
      items.length - 1,
      Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
    );
    
    return {
      startIndex,
      endIndex,
      visibleItems: items.slice(startIndex, endIndex + 1),
      offsetY: startIndex * itemHeight,
      totalHeight: items.length * itemHeight,
    };
  }, [scrollTop, items, itemHeight, containerHeight, overscan]);
  
  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);
  
  return {
    ...visibleRange,
    handleScroll,
  };
};

/**
 * Debounced value hook
 */
export const useDebouncedValue = <T,>(value: T, delay: number = 300): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    
    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);
  
  return debouncedValue;
};

/**
 * Throttled callback hook
 */
export const useThrottledCallback = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number = 300,
  deps: DependencyList = []
): T => {
  return useMemo(
    () => throttle(callback, delay),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [...deps, delay]
  ) as T;
};

/**
 * Lazy load hook with intersection observer
 */
export const useLazyLoad = (
  threshold: number = 0.1,
  rootMargin: string = '100px'
) => {
  const [ref, inView, entry] = useInView({
    threshold,
    rootMargin,
    triggerOnce: true,
  });
  
  const [hasLoaded, setHasLoaded] = useState(false);
  
  useEffect(() => {
    if (inView && !hasLoaded) {
      setHasLoaded(true);
    }
  }, [inView, hasLoaded]);
  
  return {
    ref,
    shouldLoad: hasLoaded,
    isInView: inView,
    entry,
  };
};

/**
 * Memoized async data hook
 */
export const useMemoizedAsync = <T,>(
  asyncFunction: () => Promise<T>,
  deps: DependencyList = []
) => {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  
  const memoizedFunction = useCallback(asyncFunction, deps);
  
  useEffect(() => {
    let cancelled = false;
    
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const result = await memoizedFunction();
        if (!cancelled) {
          setData(result);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };
    
    fetchData();
    
    return () => {
      cancelled = true;
    };
  }, [memoizedFunction]);
  
  return { data, loading, error };
};

/**
 * Infinite scroll hook
 */
export const useInfiniteScroll = ({
  onLoadMore,
  hasMore,
  loading,
  threshold = 0.9,
}: {
  onLoadMore: () => void;
  hasMore: boolean;
  loading: boolean;
  threshold?: number;
}) => {
  const [sentinelRef, inView] = useInView({
    threshold,
  });
  
  useEffect(() => {
    if (inView && hasMore && !loading) {
      onLoadMore();
    }
  }, [inView, hasMore, loading, onLoadMore]);
  
  return { sentinelRef };
};

/**
 * Image lazy loading hook
 */
export const useLazyImage = (src: string, placeholder?: string) => {
  const [imageSrc, setImageSrc] = useState(placeholder || '');
  const [imageRef, isInView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });
  
  useEffect(() => {
    if (isInView && src) {
      const img = new Image();
      img.src = src;
      img.onload = () => {
        setImageSrc(src);
      };
    }
  }, [isInView, src]);
  
  return {
    imageSrc,
    imageRef,
    isLoaded: imageSrc === src,
  };
};

/**
 * Web Worker hook for heavy computations
 */
export const useWebWorker = <T, R>(
  workerFunction: (data: T) => R
) => {
  const [result, setResult] = useState<R | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(false);
  const workerRef = useRef<Worker | null>(null);
  
  useEffect(() => {
    return () => {
      if (workerRef.current) {
        workerRef.current.terminate();
      }
    };
  }, []);
  
  const runWorker = useCallback((data: T) => {
    setLoading(true);
    setError(null);
    
    const workerCode = `
      self.onmessage = function(e) {
        const result = (${workerFunction.toString()})(e.data);
        self.postMessage(result);
      }
    `;
    
    const blob = new Blob([workerCode], { type: 'application/javascript' });
    const workerUrl = URL.createObjectURL(blob);
    
    if (workerRef.current) {
      workerRef.current.terminate();
    }
    
    workerRef.current = new Worker(workerUrl);
    
    workerRef.current.onmessage = (e) => {
      setResult(e.data);
      setLoading(false);
      URL.revokeObjectURL(workerUrl);
    };
    
    workerRef.current.onerror = (err) => {
      setError(new Error('Worker error'));
      setLoading(false);
      URL.revokeObjectURL(workerUrl);
    };
    
    workerRef.current.postMessage(data);
  }, [workerFunction]);
  
  return {
    result,
    error,
    loading,
    runWorker,
  };
};

/**
 * Request idle callback hook
 */
export const useIdleCallback = (
  callback: () => void,
  deps: DependencyList = []
) => {
  useEffect(() => {
    if ('requestIdleCallback' in window) {
      const handle = requestIdleCallback(callback);
      return () => cancelIdleCallback(handle);
    } else {
      const timeout = setTimeout(callback, 1);
      return () => clearTimeout(timeout);
    }
  }, deps);
};

/**
 * Prefetch hook for preloading data
 */
export const usePrefetch = () => {
  const cache = useRef(new Map());
  
  const prefetch = useCallback(async (key: string, fetcher: () => Promise<any>) => {
    if (!cache.current.has(key)) {
      cache.current.set(key, fetcher());
    }
    return cache.current.get(key);
  }, []);
  
  const getCached = useCallback((key: string) => {
    return cache.current.get(key);
  }, []);
  
  const clearCache = useCallback((key?: string) => {
    if (key) {
      cache.current.delete(key);
    } else {
      cache.current.clear();
    }
  }, []);
  
  return {
    prefetch,
    getCached,
    clearCache,
  };
};