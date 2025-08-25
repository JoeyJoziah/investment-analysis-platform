import { useCallback } from 'react';
import { useAppDispatch } from './redux';
import { addNotification } from '../store/slices/appSlice';

interface ErrorInfo {
  message: string;
  code?: string;
  status?: number;
  context?: string;
}

interface UseErrorHandlerOptions {
  showNotification?: boolean;
  logToConsole?: boolean;
  context?: string;
}

export const useErrorHandler = (options: UseErrorHandlerOptions = {}) => {
  const dispatch = useAppDispatch();
  
  const {
    showNotification = true,
    logToConsole = true,
    context = 'Application',
  } = options;

  const handleError = useCallback((error: unknown, customMessage?: string) => {
    const errorInfo = processError(error, context, customMessage);

    // Log to console if enabled
    if (logToConsole) {
      console.error(`[${errorInfo.context}] Error:`, {
        message: errorInfo.message,
        code: errorInfo.code,
        status: errorInfo.status,
        originalError: error,
      });
    }

    // Show notification if enabled
    if (showNotification) {
      dispatch(addNotification({
        type: 'error',
        message: errorInfo.message,
      }));
    }

    return errorInfo;
  }, [dispatch, showNotification, logToConsole, context]);

  const handleApiError = useCallback((error: any, fallbackMessage = 'An error occurred') => {
    let message = fallbackMessage;
    let status: number | undefined;
    let code: string | undefined;

    if (error.response) {
      // API error with response
      status = error.response.status;
      code = error.response.data?.code;
      
      switch (status) {
        case 401:
          message = 'You are not authorized to perform this action. Please log in again.';
          break;
        case 403:
          message = 'You do not have permission to access this resource.';
          break;
        case 404:
          message = 'The requested resource was not found.';
          break;
        case 429:
          message = 'Rate limit exceeded. Please wait before making more requests.';
          break;
        case 500:
          message = 'Server error occurred. Please try again later.';
          break;
        default:
          message = error.response.data?.message || fallbackMessage;
      }
    } else if (error.request) {
      // Network error
      message = 'Network error occurred. Please check your connection.';
    } else {
      // Other error
      message = error.message || fallbackMessage;
    }

    return handleError({
      message,
      status,
      code,
      context: 'API',
    });
  }, [handleError]);

  const handleAsyncError = useCallback(async (
    asyncFn: () => Promise<any>,
    errorMessage?: string
  ): Promise<any> => {
    try {
      return await asyncFn();
    } catch (error) {
      handleError(error, errorMessage);
      throw error; // Re-throw to allow caller to handle if needed
    }
  }, [handleError]);

  return {
    handleError,
    handleApiError,
    handleAsyncError,
  };
};

// Utility function to process different types of errors
const processError = (error: unknown, context: string, customMessage?: string): ErrorInfo => {
  if (customMessage) {
    return {
      message: customMessage,
      context,
    };
  }

  // Handle different error types
  if (error instanceof Error) {
    return {
      message: error.message,
      context,
    };
  }

  // Handle API errors
  if (typeof error === 'object' && error !== null) {
    const errorObj = error as any;
    
    if (errorObj.response?.data?.message) {
      return {
        message: errorObj.response.data.message,
        code: errorObj.response.data.code,
        status: errorObj.response.status,
        context,
      };
    }

    if (errorObj.message) {
      return {
        message: errorObj.message,
        code: errorObj.code,
        status: errorObj.status,
        context,
      };
    }
  }

  // Handle string errors
  if (typeof error === 'string') {
    return {
      message: error,
      context,
    };
  }

  // Fallback for unknown error types
  return {
    message: 'An unexpected error occurred',
    context,
  };
};

// Hook for form validation errors
export const useFormErrorHandler = () => {
  const { handleError } = useErrorHandler({
    showNotification: false, // Don't show notifications for form errors
    context: 'Form Validation',
  });

  const handleValidationError = useCallback((errors: Record<string, string[]>) => {
    const errorMessages = Object.entries(errors)
      .map(([field, messages]) => `${field}: ${messages.join(', ')}`)
      .join('; ');
    
    return handleError(new Error(errorMessages));
  }, [handleError]);

  return {
    handleValidationError,
  };
};

// Global error boundary error handler
export const useGlobalErrorHandler = () => {
  return useErrorHandler({
    showNotification: true,
    logToConsole: true,
    context: 'Global',
  });
};

export default useErrorHandler;