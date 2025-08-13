# SSL Certificate Setup Guide for Production

This guide explains how to obtain and configure SSL certificates for production deployment of the Investment Analysis Platform.

## Overview

SSL/TLS certificates are required for production to:
- Encrypt data in transit (HTTPS)
- Meet security compliance requirements
- Enable secure WebSocket connections
- Build user trust

## Certificate Options

### Option 1: Let's Encrypt (FREE - Recommended)

Let's Encrypt provides free SSL certificates that are trusted by all major browsers.

#### Prerequisites
- A domain name pointing to your server
- Port 80 accessible from the internet
- Root or sudo access on your server

#### Installation Steps

1. **Install Certbot**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install certbot python3-certbot-nginx

   # CentOS/RHEL
   sudo yum install epel-release
   sudo yum install certbot python3-certbot-nginx
   ```

2. **Obtain Certificate (Standalone Method)**
   ```bash
   # Stop any services using port 80
   sudo systemctl stop nginx

   # Get certificate
   sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com

   # For API subdomain
   sudo certbot certonly --standalone -d api.yourdomain.com
   ```

3. **Obtain Certificate (With Nginx)**
   ```bash
   # If Nginx is already configured
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

4. **Certificate Location**
   After successful generation, certificates will be at:
   ```
   /etc/letsencrypt/live/yourdomain.com/fullchain.pem  # Certificate
   /etc/letsencrypt/live/yourdomain.com/privkey.pem    # Private key
   ```

5. **Auto-Renewal Setup**
   ```bash
   # Test renewal
   sudo certbot renew --dry-run

   # Add to crontab for auto-renewal
   sudo crontab -e
   # Add this line:
   0 2 * * * /usr/bin/certbot renew --quiet
   ```

### Option 2: Cloudflare (FREE with Domain)

If using Cloudflare for DNS:

1. **Enable Cloudflare SSL**
   - Log into Cloudflare Dashboard
   - Go to SSL/TLS → Overview
   - Select "Full (strict)" or "Flexible"

2. **Origin Certificate (Optional)**
   - Go to SSL/TLS → Origin Server
   - Create Certificate
   - Download certificate and key

### Option 3: Commercial SSL Certificate

For enterprise deployments:

1. **Purchase from Certificate Authority**
   - DigiCert
   - Comodo/Sectigo
   - GoDaddy
   - RapidSSL

2. **Generate CSR**
   ```bash
   openssl req -new -newkey rsa:2048 -nodes \
     -keyout yourdomain.key \
     -out yourdomain.csr \
     -subj "/C=US/ST=State/L=City/O=Company/CN=yourdomain.com"
   ```

3. **Submit CSR to CA and download certificates**

### Option 4: Self-Signed (Development Only)

For testing purposes only:

```bash
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout server.key \
  -out server.crt \
  -days 365 \
  -subj "/C=US/ST=State/L=City/O=Company/CN=localhost"
```

## Configuration for Investment Analysis App

### 1. Update Environment Variables

Edit `.env.production`:
```env
# SSL Configuration
SSL_CERT_PATH=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/yourdomain.com/privkey.pem
FORCE_HTTPS=True
```

### 2. Docker Compose Configuration

Update `docker-compose.prod.yml`:
```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - /etc/letsencrypt:/etc/letsencrypt:ro
    depends_on:
      - backend
      - frontend
```

### 3. Nginx Configuration

Create `nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    # Redirect HTTP to HTTPS
    server {
        listen 80;
        server_name yourdomain.com api.yourdomain.com;
        return 301 https://$server_name$request_uri;
    }

    # HTTPS configuration for API
    server {
        listen 443 ssl http2;
        server_name api.yourdomain.com;

        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;
        ssl_prefer_server_ciphers on;

        # Security headers
        add_header Strict-Transport-Security "max-age=31536000" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header X-Frame-Options "DENY" always;

        location / {
            proxy_pass http://backend:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket support
        location /ws {
            proxy_pass http://backend:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }

    # HTTPS configuration for Frontend
    server {
        listen 443 ssl http2;
        server_name yourdomain.com www.yourdomain.com;

        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers HIGH:!aNULL:!MD5;

        location / {
            proxy_pass http://frontend:3000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

### 4. Kubernetes Ingress with Cert-Manager

For Kubernetes deployments:

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: investment-app-tls
  namespace: investment-analysis
spec:
  secretName: investment-app-tls-secret
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - yourdomain.com
    - api.yourdomain.com
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: investment-app-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - yourdomain.com
        - api.yourdomain.com
      secretName: investment-app-tls-secret
  rules:
    - host: api.yourdomain.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: backend-service
                port:
                  number: 8000
```

## Cloud Provider SSL Options

### DigitalOcean App Platform
- SSL certificates are automatically provisioned
- No configuration needed

### AWS
- Use AWS Certificate Manager (ACM) - FREE
- Integrates with Load Balancers
```bash
aws acm request-certificate \
  --domain-name yourdomain.com \
  --validation-method DNS \
  --subject-alternative-names "*.yourdomain.com"
```

### Google Cloud
- Use Google-managed certificates
- Configure in Load Balancer

### Azure
- Use App Service Managed Certificates - FREE
- Or Azure Key Vault for custom certificates

## Testing SSL Configuration

1. **Check Certificate**
   ```bash
   openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
   ```

2. **SSL Labs Test**
   - Visit: https://www.ssllabs.com/ssltest/
   - Enter your domain
   - Aim for A+ rating

3. **Check HTTPS Redirect**
   ```bash
   curl -I http://yourdomain.com
   # Should show 301 redirect to https://
   ```

## Troubleshooting

### Common Issues

1. **Certificate Not Trusted**
   - Ensure using fullchain.pem (includes intermediate certificates)
   - Check certificate domain matches

2. **Mixed Content Warnings**
   - Update all internal links to use HTTPS
   - Check API calls use HTTPS

3. **Certificate Renewal Failed**
   - Check port 80 is accessible
   - Verify domain still points to server
   - Check certbot logs: `/var/log/letsencrypt/`

## Security Best Practices

1. **Strong Ciphers Only**
   - Disable TLS 1.0 and 1.1
   - Use modern cipher suites

2. **HSTS Header**
   - Enable HTTP Strict Transport Security
   - Consider HSTS preloading

3. **Certificate Pinning** (Optional)
   - For mobile apps
   - Increases security but complicates updates

4. **Monitor Expiration**
   - Set up alerts 30 days before expiry
   - Use monitoring tools

## Cost Summary

- **Let's Encrypt**: FREE
- **Cloudflare**: FREE (with domain)
- **AWS ACM**: FREE (with AWS services)
- **Commercial SSL**: $10-500/year
- **Extended Validation (EV)**: $150-1000/year

For the Investment Analysis App staying under $50/month, **Let's Encrypt is the recommended choice**.