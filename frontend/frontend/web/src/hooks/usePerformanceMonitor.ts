import { useEffect, useRef, useState } from 'react';
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

interface PerformanceMetrics {
  cls?: number;
  fid?: number;
  fcp?: number;
  lcp?: number;
  ttfb?: number;
  renderTime?: number;
  memoryUsage?: number;
}

interface UsePerformanceMonitorOptions {
  enabled?: boolean;
  reportInterval?: number;
  onMetrics?: (metrics: PerformanceMetrics) => void;
}

export const usePerformanceMonitor = (options: UsePerformanceMonitorOptions = {}) => {
  const {
    enabled = process.env.NODE_ENV === 'development',
    reportInterval = 5000, // Report every 5 seconds
    onMetrics,
  } = options;

  const [metrics, setMetrics] = useState<PerformanceMetrics>({});
  const metricsRef = useRef<PerformanceMetrics>({});
  const intervalRef = useRef<NodeJS.Timeout>();
  const renderStartTime = useRef<number>(performance.now());

  // Measure component render time
  useEffect(() => {
    if (!enabled) return;
    
    const renderTime = performance.now() - renderStartTime.current;
    metricsRef.current.renderTime = renderTime;
    
    setMetrics(prev => ({ ...prev, renderTime }));
  }, [enabled]);

  // Initialize Web Vitals monitoring
  useEffect(() => {
    if (!enabled) return;

    // Core Web Vitals
    getCLS((metric) => {
      metricsRef.current.cls = metric.value;
      setMetrics(prev => ({ ...prev, cls: metric.value }));
    });

    getFID((metric) => {
      metricsRef.current.fid = metric.value;
      setMetrics(prev => ({ ...prev, fid: metric.value }));
    });

    getFCP((metric) => {
      metricsRef.current.fcp = metric.value;
      setMetrics(prev => ({ ...prev, fcp: metric.value }));
    });

    getLCP((metric) => {
      metricsRef.current.lcp = metric.value;
      setMetrics(prev => ({ ...prev, lcp: metric.value }));
    });

    getTTFB((metric) => {
      metricsRef.current.ttfb = metric.value;
      setMetrics(prev => ({ ...prev, ttfb: metric.value }));
    });
  }, [enabled]);

  // Monitor memory usage
  useEffect(() => {
    if (!enabled) return;

    const measureMemory = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        const memoryUsage = memory.usedJSHeapSize / 1024 / 1024; // MB
        
        metricsRef.current.memoryUsage = memoryUsage;
        setMetrics(prev => ({ ...prev, memoryUsage }));
      }
    };

    // Initial measurement
    measureMemory();

    // Set up periodic monitoring
    intervalRef.current = setInterval(measureMemory, reportInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, reportInterval]);

  // Report metrics
  useEffect(() => {
    if (enabled && onMetrics && Object.keys(metrics).length > 0) {
      onMetrics(metrics);
    }
  }, [metrics, onMetrics, enabled]);

  // Log performance warnings
  useEffect(() => {
    if (!enabled) return;

    const warnings = [];
    
    if (metrics.cls && metrics.cls > 0.1) {
      warnings.push(`High CLS detected: ${metrics.cls.toFixed(3)}`);
    }
    
    if (metrics.fid && metrics.fid > 100) {
      warnings.push(`High FID detected: ${metrics.fid.toFixed(0)}ms`);
    }
    
    if (metrics.lcp && metrics.lcp > 2500) {
      warnings.push(`High LCP detected: ${metrics.lcp.toFixed(0)}ms`);
    }
    
    if (metrics.renderTime && metrics.renderTime > 16.67) { // 60fps threshold
      warnings.push(`Slow render detected: ${metrics.renderTime.toFixed(2)}ms`);
    }
    
    if (metrics.memoryUsage && metrics.memoryUsage > 50) {
      warnings.push(`High memory usage: ${metrics.memoryUsage.toFixed(1)}MB`);
    }

    if (warnings.length > 0) {
      console.warn('Performance Issues Detected:', warnings);
    }
  }, [metrics, enabled]);

  const getPerformanceScore = (): number => {
    if (Object.keys(metrics).length === 0) return 0;
    
    let score = 100;
    
    // CLS penalty
    if (metrics.cls) {
      if (metrics.cls > 0.25) score -= 30;
      else if (metrics.cls > 0.1) score -= 15;
    }
    
    // FID penalty
    if (metrics.fid) {
      if (metrics.fid > 300) score -= 25;
      else if (metrics.fid > 100) score -= 15;
    }
    
    // LCP penalty  
    if (metrics.lcp) {
      if (metrics.lcp > 4000) score -= 25;
      else if (metrics.lcp > 2500) score -= 15;
    }
    
    // Render time penalty
    if (metrics.renderTime) {
      if (metrics.renderTime > 50) score -= 20;
      else if (metrics.renderTime > 16.67) score -= 10;
    }

    return Math.max(0, score);
  };

  const reset = () => {
    renderStartTime.current = performance.now();
    metricsRef.current = {};
    setMetrics({});
  };

  return {
    metrics,
    performanceScore: getPerformanceScore(),
    reset,
    isMonitoring: enabled,
  };
};

// Hook for monitoring specific operations
export const useOperationTimer = (operationName: string) => {
  const startTime = useRef<number>();
  const [duration, setDuration] = useState<number>();

  const start = () => {
    startTime.current = performance.now();
    setDuration(undefined);
  };

  const end = () => {
    if (startTime.current) {
      const elapsed = performance.now() - startTime.current;
      setDuration(elapsed);
      
      if (process.env.NODE_ENV === 'development') {
        console.log(`${operationName} took ${elapsed.toFixed(2)}ms`);
        
        if (elapsed > 100) {
          console.warn(`Slow operation detected: ${operationName} (${elapsed.toFixed(2)}ms)`);
        }
      }
      
      startTime.current = undefined;
    }
  };

  return { start, end, duration, isRunning: !!startTime.current };
};

export default usePerformanceMonitor;