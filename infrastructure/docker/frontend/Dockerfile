# Multi-stage build for React/Vite frontend

# Build args
ARG NODE_VERSION=18

# Stage 1: Development (for hot-reloading)
FROM node:${NODE_VERSION}-alpine AS development

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies)
RUN npm install --legacy-peer-deps

# Expose port for development server
EXPOSE 3000

# Development command (overridden by docker-compose)
CMD ["npm", "start"]

# Stage 2: Build
FROM node:${NODE_VERSION}-alpine AS builder

WORKDIR /build

# Copy package files (context is frontend/web when built from docker-compose)
COPY package*.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy source code
COPY . .

# Build arguments for environment variables (VITE_ prefix required by Vite bundler)
ARG VITE_APP_ENV=production
ARG VITE_API_URL
ARG VITE_WS_URL
ARG VITE_APP_VERSION

# Set environment variables for build
ENV VITE_APP_ENV=${VITE_APP_ENV}
ENV VITE_API_URL=${VITE_API_URL}
ENV VITE_WS_URL=${VITE_WS_URL}
ENV VITE_APP_VERSION=${VITE_APP_VERSION}

# Build the application
RUN npm run build

# Stage 3: Production server
FROM nginx:alpine

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built files from builder
COPY --from=builder /build/dist /usr/share/nginx/html

# Create nginx.conf for single-page app with health check
RUN echo 'server { \
    listen 80; \
    server_name _; \
    root /usr/share/nginx/html; \
    index index.html; \
    \
    location /health { \
        access_log off; \
        return 200 "OK"; \
        add_header Content-Type text/plain; \
    } \
    \
    location / { \
        try_files $uri $uri/ /index.html; \
    } \
    \
    # Security headers \
    add_header X-Frame-Options "SAMEORIGIN" always; \
    add_header X-Content-Type-Options "nosniff" always; \
    add_header X-XSS-Protection "1; mode=block" always; \
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always; \
    \
    # Gzip compression \
    gzip on; \
    gzip_vary on; \
    gzip_min_length 1024; \
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json; \
}' > /etc/nginx/conf.d/default.conf

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]