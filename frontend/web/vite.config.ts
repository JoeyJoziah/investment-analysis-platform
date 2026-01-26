/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0', // Allow external connections (required for Docker)
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://backend:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://backend:8000',
        ws: true,
      },
    },
  },
  preview: {
    port: 3000,
    host: '0.0.0.0',
  },
  build: {
    // Target 300KB chunks for optimal loading performance
    chunkSizeWarningLimit: 300,
    // Enable source maps for debugging (can be disabled in production)
    sourcemap: false,
    // Minification settings
    minify: 'terser',
    terserOptions: {
      compress: {
        // Remove console.log in production
        drop_console: true,
        drop_debugger: true,
      },
    },
    rollupOptions: {
      output: {
        // =================================================================
        // MANUAL CHUNK CONFIGURATION
        // =================================================================
        // Split vendor chunks for better caching and smaller initial bundle
        // Each chunk is optimized to stay under 300KB when possible
        // =================================================================
        manualChunks: (id) => {
          // React core - small, frequently cached
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/')) {
            return 'react-core';
          }

          // React Router - loaded with initial app
          if (id.includes('node_modules/react-router')) {
            return 'react-router';
          }

          // Emotion (CSS-in-JS) - split from MUI for better caching
          if (id.includes('node_modules/@emotion')) {
            return 'emotion';
          }

          // MUI System/Base - Core utilities
          if (id.includes('node_modules/@mui/system') ||
              id.includes('node_modules/@mui/base') ||
              id.includes('node_modules/@mui/utils') ||
              id.includes('node_modules/@mui/private-theming')) {
            return 'mui-system';
          }

          // MUI Core Components - Material UI base components
          if (id.includes('node_modules/@mui/material')) {
            return 'mui-core';
          }

          // MUI Icons - large bundle, lazy load when possible
          if (id.includes('node_modules/@mui/icons-material')) {
            return 'mui-icons';
          }

          // MUI X Components - Data grid, date pickers
          if (id.includes('node_modules/@mui/x-')) {
            return 'mui-x';
          }

          // Redux Toolkit - State management
          if (id.includes('node_modules/@reduxjs/toolkit') ||
              id.includes('node_modules/react-redux') ||
              id.includes('node_modules/redux') ||
              id.includes('node_modules/immer')) {
            return 'redux';
          }

          // D3 modules - Split from recharts for lazy loading
          if (id.includes('node_modules/d3-')) {
            return 'd3';
          }

          // Recharts core - Lightweight charting library
          if (id.includes('node_modules/recharts')) {
            return 'recharts';
          }

          // Plotly - Heavy charting (lazy loaded with Analysis page)
          if (id.includes('node_modules/plotly') ||
              id.includes('node_modules/react-plotly')) {
            return 'plotly';
          }

          // Lightweight Charts - TradingView charts
          if (id.includes('node_modules/lightweight-charts')) {
            return 'lightweight-charts';
          }

          // Chart.js - Another charting option
          if (id.includes('node_modules/chart.js') ||
              id.includes('node_modules/react-chartjs') ||
              id.includes('node_modules/chartjs-')) {
            return 'chartjs';
          }

          // Date utilities
          if (id.includes('node_modules/date-fns')) {
            return 'date-fns';
          }

          // Framer Motion - Animations
          if (id.includes('node_modules/framer-motion')) {
            return 'framer-motion';
          }

          // Network utilities
          if (id.includes('node_modules/axios') ||
              id.includes('node_modules/socket.io')) {
            return 'network';
          }

          // React utility libraries
          if (id.includes('node_modules/react-transition-group') ||
              id.includes('node_modules/react-is') ||
              id.includes('node_modules/prop-types') ||
              id.includes('node_modules/clsx') ||
              id.includes('node_modules/hoist-non-react-statics')) {
            return 'react-utils';
          }

          // Remaining vendor modules
          if (id.includes('node_modules')) {
            return 'vendor';
          }
        },
        // Consistent chunk naming for better caching
        chunkFileNames: (chunkInfo) => {
          const facadeModuleId = chunkInfo.facadeModuleId
            ? chunkInfo.facadeModuleId.split('/').pop()?.replace('.tsx', '').replace('.ts', '')
            : 'chunk';
          return `assets/${chunkInfo.name || facadeModuleId}-[hash].js`;
        },
        // Entry point naming
        entryFileNames: 'assets/[name]-[hash].js',
        // Asset naming
        assetFileNames: 'assets/[name]-[hash].[ext]',
      },
    },
  },
  // =================================================================
  // OPTIMIZATION SETTINGS
  // =================================================================
  optimizeDeps: {
    // Pre-bundle these dependencies for faster dev server start
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@mui/material',
      '@reduxjs/toolkit',
      'react-redux',
      'axios',
    ],
    // Exclude heavy charting libs from pre-bundling (they're lazy loaded)
    exclude: [
      'plotly.js',
      'react-plotly.js',
    ],
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/setupTests.ts'],
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
    },
  },
})
