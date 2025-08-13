/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL: string
  readonly VITE_WS_URL: string
  readonly VITE_APP_NAME: string
  readonly VITE_APP_VERSION: string
  readonly VITE_ENABLE_WEBSOCKETS: string
  readonly VITE_ENABLE_ANALYTICS: string
  readonly VITE_ENABLE_DEBUG: string
  // Add backward compatibility for REACT_APP_ prefixed variables
  readonly REACT_APP_API_URL: string
  readonly REACT_APP_WS_URL: string
  readonly REACT_APP_NAME: string
  readonly REACT_APP_VERSION: string
  readonly REACT_APP_ENABLE_WEBSOCKETS: string
  readonly REACT_APP_ENABLE_ANALYTICS: string
  readonly REACT_APP_ENABLE_DEBUG: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// For backward compatibility with process.env usage
declare global {
  interface Window {
    process: {
      env: ImportMetaEnv
    }
  }
}