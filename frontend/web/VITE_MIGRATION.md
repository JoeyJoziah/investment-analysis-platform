# Vite Migration Guide

## Migration from Create React App to Vite

This project has been migrated from Create React App (react-scripts) to Vite to resolve TypeScript 5 compatibility issues and improve build performance.

## What Changed

### 1. Build Tool
- **Before**: Create React App (react-scripts@5.0.1)
- **After**: Vite (^5.0.12)

### 2. Dependencies Updated
- Removed `react-scripts`
- Added `vite`, `@vitejs/plugin-react`, `vitest`, and `@vitest/ui`
- Kept TypeScript 5.3.3 (now fully supported)

### 3. Configuration Files
- **Added**: `vite.config.ts` - Vite configuration
- **Added**: `tsconfig.node.json` - TypeScript config for Vite
- **Added**: `index.html` in root directory (Vite requirement)
- **Added**: `src/vite-env.d.ts` - TypeScript environment declarations
- **Added**: `src/setupTests.ts` - Test setup for Vitest
- **Added**: `src/utils/env.ts` - Environment variable utility
- **Modified**: `tsconfig.json` - Updated for Vite compatibility
- **Modified**: `.env` - Added VITE_ prefixed variables

### 4. Scripts Updated
- `npm start` → Still works, now runs Vite
- `npm run dev` → Alternative to start (Vite convention)
- `npm run build` → Builds with Vite
- `npm test` → Runs tests with Vitest
- `npm run preview` → Preview production build

## Installation Instructions

1. **Clean install dependencies**:
   ```bash
   cd frontend/web
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   # or
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

## Environment Variables

### Migration Required
Vite uses `VITE_` prefix for custom environment variables instead of `REACT_APP_`.

#### Backward Compatibility
The `.env` file has been updated to include both formats for smooth transition:
- `VITE_API_URL` (new format)
- `REACT_APP_API_URL` (backward compatibility)

#### Using Environment Variables in Code

**Option 1: Use the env utility (Recommended)**
```typescript
import { env } from '@/utils/env';

// Automatically handles both VITE_ and REACT_APP_ prefixes
const apiUrl = env.API_URL;
```

**Option 2: Direct access (Vite way)**
```typescript
const apiUrl = import.meta.env.VITE_API_URL;
```

**Option 3: Legacy access (for existing code)**
```typescript
// Will still work during transition
const apiUrl = process.env.REACT_APP_API_URL;
```

## Code Changes Required

### 1. Update Environment Variable References
Search for `process.env.REACT_APP_` in your codebase and update to use either:
- The `env` utility from `src/utils/env.ts`
- Direct `import.meta.env.VITE_` access

### 2. Public Assets
- Move any assets from `public/` that are referenced with `%PUBLIC_URL%` 
- Update references to use absolute paths (e.g., `/favicon.ico` instead of `%PUBLIC_URL%/favicon.ico`)

### 3. Import Aliases
You can now use `@/` as an alias for `src/`:
```typescript
import { MyComponent } from '@/components/MyComponent';
```

## Benefits of Vite

1. **TypeScript 5 Support**: Full support for latest TypeScript features
2. **Faster Development**: Instant server start and HMR (Hot Module Replacement)
3. **Faster Builds**: 10-100x faster than webpack-based tools
4. **Better Error Messages**: More helpful error overlay
5. **Modern Tooling**: ESM-based, future-proof architecture
6. **Active Development**: Unlike CRA, Vite is actively maintained

## Troubleshooting

### Issue: Module not found errors
**Solution**: Clear node_modules and reinstall:
```bash
rm -rf node_modules package-lock.json
npm install
```

### Issue: Environment variables not working
**Solution**: Ensure variables are prefixed with `VITE_` or use the env utility

### Issue: Tests not running
**Solution**: Tests now use Vitest. Run with:
```bash
npm test
```

### Issue: Build errors
**Solution**: Run TypeScript check separately:
```bash
npx tsc --noEmit
npm run build
```

## Rollback Instructions

If you need to rollback to Create React App:
1. Restore the original `package.json` from git
2. Delete Vite configuration files
3. Move `index.html` back to `public/` folder
4. Reinstall dependencies

## Additional Resources

- [Vite Documentation](https://vitejs.dev/)
- [Migrating from CRA](https://vitejs.dev/guide/migration.html)
- [Vitest Documentation](https://vitest.dev/)