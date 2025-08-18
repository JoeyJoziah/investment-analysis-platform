import { createTheme, ThemeOptions } from '@mui/material/styles';

declare module '@mui/material/styles' {
  interface Theme {
    status: {
      danger: string;
      success: string;
      warning: string;
    };
  }
  interface ThemeOptions {
    status?: {
      danger?: string;
      success?: string;
      warning?: string;
    };
  }
  interface Palette {
    neutral: Palette['primary'];
  }
  interface PaletteOptions {
    neutral?: PaletteOptions['primary'];
  }
}

const getDesignTokens = (mode: 'light' | 'dark'): ThemeOptions => ({
  palette: {
    mode,
    ...(mode === 'light'
      ? {
          // Light mode colors
          primary: {
            main: '#1976d2',
            light: '#42a5f5',
            dark: '#1565c0',
          },
          secondary: {
            main: '#dc004e',
            light: '#f50057',
            dark: '#c51162',
          },
          background: {
            default: '#f5f5f5',
            paper: '#ffffff',
          },
          neutral: {
            main: '#64748B',
            light: '#94A3B8',
            dark: '#475569',
          },
          success: {
            main: '#4caf50',
            light: '#81c784',
            dark: '#388e3c',
          },
          error: {
            main: '#f44336',
            light: '#e57373',
            dark: '#d32f2f',
          },
          warning: {
            main: '#ff9800',
            light: '#ffb74d',
            dark: '#f57c00',
          },
          info: {
            main: '#2196f3',
            light: '#64b5f6',
            dark: '#1976d2',
          },
        }
      : {
          // Dark mode colors
          primary: {
            main: '#90caf9',
            light: '#e3f2fd',
            dark: '#42a5f5',
          },
          secondary: {
            main: '#f48fb1',
            light: '#ffc1e3',
            dark: '#bf5f82',
          },
          background: {
            default: '#0a0e27',
            paper: '#1a1f3a',
          },
          neutral: {
            main: '#94A3B8',
            light: '#CBD5E1',
            dark: '#64748B',
          },
          success: {
            main: '#66bb6a',
            light: '#a5d6a7',
            dark: '#43a047',
          },
          error: {
            main: '#ef5350',
            light: '#ef9a9a',
            dark: '#e53935',
          },
          warning: {
            main: '#ffa726',
            light: '#ffcc80',
            dark: '#fb8c00',
          },
          info: {
            main: '#29b6f6',
            light: '#81d4fa',
            dark: '#0288d1',
          },
        }),
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
      lineHeight: 1.2,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      lineHeight: 1.3,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      lineHeight: 1.4,
    },
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
      lineHeight: 1.4,
    },
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      lineHeight: 1.5,
    },
    h6: {
      fontSize: '1rem',
      fontWeight: 500,
      lineHeight: 1.6,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: mode === 'light' 
            ? '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)'
            : '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px 0 rgba(0, 0, 0, 0.24)',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
        },
      },
    },
  },
  status: {
    danger: '#ff5252',
    success: '#4caf50',
    warning: '#ff9800',
  },
});

export const createAppTheme = (mode: 'light' | 'dark') => {
  return createTheme(getDesignTokens(mode));
};

export const lightTheme = createAppTheme('light');
export const darkTheme = createAppTheme('dark');
export const theme = lightTheme; // Default theme