# Development Dockerfile for React frontend with Vite
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json only (without lock file)
COPY package.json ./

# Install dependencies with legacy peer deps
RUN npm install --legacy-peer-deps

# Copy application code
COPY . .

# Expose Vite dev server port
EXPOSE 3000

# Start Vite development server
CMD ["npm", "start"]
