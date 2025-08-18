import { configureStore } from '@reduxjs/toolkit';
import appReducer from './slices/appSlice';
import dashboardReducer from './slices/dashboardSlice';
import recommendationsReducer from './slices/recommendationsSlice';
import portfolioReducer from './slices/portfolioSlice';
import marketReducer from './slices/marketSlice';
import stockReducer from './slices/stockSlice';

export const store = configureStore({
  reducer: {
    app: appReducer,
    dashboard: dashboardReducer,
    recommendations: recommendationsReducer,
    portfolio: portfolioReducer,
    market: marketReducer,
    stock: stockReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: [
          'dashboard/fetchData/fulfilled',
          'portfolio/fetchPortfolio/fulfilled',
          'market/fetchOverview/fulfilled',
          'stock/fetchData/fulfilled',
        ],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['meta.arg', 'payload.timestamp'],
        // Ignore these paths in the state
        ignoredPaths: ['items.dates'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;