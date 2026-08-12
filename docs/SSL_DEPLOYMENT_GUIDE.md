# SSL Deployment Guide

Version: 1.0.0
Last Updated: 2026-05-08
Status: scaffolding present, awaiting domain registration to provision certs.

## TL;DR

```bash
# Provision certs (requires public DNS pointing at the host + port 80 reachable)
./scripts/init-ssl.sh your-domain.com admin@your-domain.com

# Production mounts infrastructure/docker/nginx/nginx-ssl.conf as /etc/nginx/nginx.conf (Q1 2026-08); it requires ssl/ to hold fullchain.pem/privkey.pem/chain.pem/dhparam.pem
./start.sh prod
```

## What is already in the repo

| Component | Path | Purpose |
|---|---|---|
| Cert provisioning | `scripts/init-ssl.sh` | Let's Encrypt or self-signed; validates inputs, generates 4096-bit dhparam, creates symlinks for nginx. |
| nginx SSL config | `infrastructure/docker/nginx/nginx-ssl.conf` | TLS termination, HTTP/2, modern cipher suite, rate-limit zones, gzip. |
| Security headers | `config/services/nginx/security-headers.conf` | HSTS, X-Frame-Options, CSP, etc. |
| Production compose | `docker-compose.production.yml` | Wires nginx + backend + certbot. |
| Renewal | certbot container | Auto-renews on cron. |

## Prerequisites

1. Public DNS A record pointing at the production host
2. Port 80 reachable from the internet (Let's Encrypt HTTP-01 challenge)
3. Port 443 forwarded to the nginx container
4. Docker + docker-compose on the host

## Provisioning workflow

### Production (Let's Encrypt)
```bash
# 1. DNS: A record for your domain pointing at the server
# 2. Stop anything on port 80
sudo lsof -i :80
# 3. Provision
./scripts/init-ssl.sh investment.example.com ops@example.com
# Choose option 1 (Let's Encrypt)
# 4. Verify cert files
ls ssl/    # fullchain.pem privkey.pem chain.pem dhparam.pem
# 5. Start production stack
./start.sh prod
# 6. Smoke test
curl -I https://investment.example.com/health
```

### Staging / local prod-like (self-signed)
```bash
./scripts/init-ssl.sh local.investment.test ops@example.com
# Choose option 2 (self-signed)
```

## Renewal

Certbot runs in the production compose stack on a 12h schedule and renews any cert
within 30 days of expiry.
```bash
docker compose -f docker-compose.production.yml logs certbot --tail 50
```

Manual renewal:
```bash
docker compose -f docker-compose.production.yml exec certbot \
  certbot renew --webroot --webroot-path=/var/www/certbot
docker compose -f docker-compose.production.yml exec nginx nginx -s reload
```

## Troubleshooting

- **Port 80 already in use** — `sudo lsof -i :80` to find the holder.
- **DNS lookup failed** — verify `dig +short your-domain.com` returns the public IP.
- **Connection refused on 443** — confirm `docker compose ps` shows nginx as Up; check host firewall.
- **Renewal failures** — verify the webroot path `/var/www/certbot` is mounted and writable in both nginx and certbot containers.

## Rotation

```bash
./scripts/init-ssl.sh your-domain.com ops@example.com
# Choose option 1; --force-renewal is already enabled in the script
docker compose -f docker-compose.production.yml restart nginx
```

## See also
- `RUNBOOK.md` — incident response procedures
- `PRODUCTION_DEPLOYMENT_GUIDE.md` — full production deployment
- `SECURITY.md` — security posture summary