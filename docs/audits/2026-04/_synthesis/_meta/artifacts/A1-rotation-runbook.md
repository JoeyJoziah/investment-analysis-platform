# Workstream A1 — Rotation Runbook (Step 1)

**Status:** PARKED awaiting maintenance window. Authored by /loki-mode session 2026-04-28.
**Workpaper reference:** `docs/audits/2026-04/_synthesis/workpaper/A.md` §3 Step 1.
**PRD reference:** `docs/audits/2026-04/PRD-for-loki.md` §3 A, §6, §6.1.

## Decisions of record (from operator ack)

- **Secret manager:** 1Password.
  - Vault: `Personal`
  - Item naming: `audit-2026-04/<credential-name>`
  - Local dev access pattern: `op read "op://Personal/audit-2026-04/<name>/credential"`
  - Production injection: 1Password Connect, or `op run --env-file=<file>`
  - Step 2 env templates MUST be generated to be compatible with the above.
- **G3-phase-1 status:** in flight in a separate session, **not yet merged to `main`**. Per PRD §6.1, A's `.github/workflows/*` edits (part of Step 2 CI provisioning) remain blocked until G3-phase-1 PR is merged.
- **Rotation status:** NOT YET STARTED.
- **Step 11 (`git filter-repo` history purge):** deferred to post-program completion per PRD §3 A and §6 schedule. Out of scope for A1.

## Scope

Step 1 rotates every live credential currently exposed in committed source. Steps 2–17 of Workstream A (env templates, code/config replacements, docs) execute only after Step 1 is acknowledged complete.

## 1. Credentials to rotate (9 total)

| # | Credential | Current value (in repo) | Rotated by | Consumers to update |
|---|---|---|---|---|
| 1 | Postgres `POSTGRES_PASSWORD` | `9v1g^OV9XUwzUP6cEgCYgNOE` | DBA / `ALTER USER postgres WITH PASSWORD …` | backend API, ETL jobs, Alembic, scripts, Airflow, staging, CI |
| 2 | Redis `REDIS_PASSWORD` | `RsYque&Xh%TUD*Nv^7k7B8X3` | Redis `CONFIG SET requirepass` + restart consumers | backend cache, Celery/Redis consumers, rate-limiter |
| 3 | Elasticsearch `ELASTIC_PASSWORD` | `4Bx+UM1CdSiEbMlQueRVvda+A4fLzCRsyuHUHbv5wMw=` | ES `_security/user/elastic/_password` | search service, Logstash/Filebeat, Kibana |
| 4 | Grafana admin | `a2A5j4JQ0nF8aTLyIYwRgZnMLQpIu5lW9jYx6pB5Xdw=` | Grafana UI / `grafana-cli admin reset-admin-password` | Grafana provisioning, oncall |
| 5 | Airflow admin (Prometheus basic-auth) | `admin:admin` | `airflow users update -u admin --password …` | Prometheus scrape job (Step 9 reads from secret file) |
| 6 | JWT signing secret (`JWT_SECRET_KEY`, 64-char) | leaked in docs | `python -c "import secrets; print(secrets.token_hex(64))"` | backend auth — **invalidates all current sessions; B (jwt-auth) hard-blocks on this** |
| 7 | Fernet key `GDPR_ENCRYPTION_KEY` | leaked in docs | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` | **CRITICAL: re-encrypt any data encrypted with old key BEFORE revoking** |
| 8 | Google API key | leaked in docs | GCP console → rotate / regenerate | any Google-API caller |
| 9 | HuggingFace token | leaked in docs | HF account → revoke + new fine-grained token | ML model loaders |

## 2. Pre-rotation safety checklist

- [ ] Maintenance window scheduled (workpaper §10 risk: up to 30-min DB downtime worst-case). Anticipated within 1–2 weeks; not before E and G3-phase-1 PRs are merged to `main`.
- [ ] Backups taken: Postgres snapshot, Redis RDB, Elasticsearch snapshot.
- [ ] Inventory of consumers complete for each credential (especially #7 Fernet — any encrypted data at rest must be re-encrypted, not just have its key swapped).
- [ ] Each new credential provisioned in 1Password under `op://Personal/audit-2026-04/<name>` BEFORE rotating in the source system.
- [ ] Branch protection on `main` and `remediation/audit-2026-04` allows operator to merge/push as needed during the window.

## 3. Rotation order (blue/green where possible)

1. **JWT secret (#6)** — single-app rotation; expect global session invalidation. Coordinate with users / brief downtime banner.
2. **Fernet key (#7)** — re-encrypt PII columns with new key first, then revoke old. **Do NOT skip the re-encryption step or stored data becomes unreadable.**
3. **Postgres (#1)** — provision new password, update consumers, then `ALTER USER`. Verify `pg_stat_activity` shows zero connections on old auth before revoking.
4. **Redis (#2)** — same blue/green pattern.
5. **Elasticsearch (#3)**, **Grafana (#4)**, **Airflow (#5)** — same pattern, each ~5 min.
6. **Google (#8)** and **HuggingFace (#9)** — provision new, swap consumers, revoke old.

## 4. Post-rotation verification (gates Step 2+)

```bash
# Old DB password must fail
PGPASSWORD='9v1g^OV9XUwzUP6cEgCYgNOE' psql -h <prod> -U postgres -d investment_db -c '\q'   # MUST FAIL with auth error

# Old Redis password must fail
redis-cli -h <prod> -a 'RsYque&Xh%TUD*Nv^7k7B8X3' ping   # MUST FAIL (NOAUTH/WRONGPASS)

# JWT — old token must be rejected
curl -H "Authorization: Bearer <old-jwt>" https://<api>/v1/me   # 401

# Health checks pass with new creds
curl https://<api>/health   # 200 with all components green
```

## 5. Hand-back signal to /loki-mode

When Step 1 is complete and verifications pass, the next /loki-mode A1 session will be started with this acknowledgement format:

```
ack-rotation: complete
secret-manager: 1password (op://Personal/audit-2026-04/<name>)
g3-phase-1: <merged-PR-url | not-merged>
```

On `ack-rotation: complete` Loki will:
- Execute Step 2 (env templates only — file edits committed to `remediation/audit-2026-04`). CI/staging/prod provisioning stays with the operator.
- Execute Steps 3–10, 12–17 as separate logical-group PRs on `remediation/audit-2026-04`.
- For any `.github/workflows/*` touch: if `g3-phase-1` is not merged, halt and report (per PRD §6.1).
- Skip Step 11 entirely (deferred per PRD §3 A and §6).

## 6. Cross-workstream blocker note (must appear in every A1 PR description)

> **Workstream A1 hard-blocks Workstream B (jwt-auth)** on the JWT_SECRET-replacement step (Step 1, credential #6). B cannot start until this rotation is acknowledged complete.

## 7. Findings covered by Workstream A1 (25)

F-05-003, F-05-012, F-07-001, F-08-009, F-08-012, F-10-004, F-10-007, F-10-013, F-12-005 (absorbs F-12-016), F-12-010, F-13-004, F-13-020, F-15-012, F-15-024, F-16-003, F-16-006, F-16-007, F-16-009, F-16-010, F-17-001, F-17-002, F-17-007, F-17-009, F-18-005.

## 8. Park log

- 2026-04-28 — runbook authored, operator parked A1 awaiting maintenance window. Decisions captured above. Step 11 confirmed deferred. G3-phase-1 confirmed not yet merged.
