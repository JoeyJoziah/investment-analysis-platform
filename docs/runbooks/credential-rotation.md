# Credential Rotation Runbook — Live Postgres + Redis

**Status:** Authored, awaiting a scheduled maintenance window. This runbook is the
operational procedure that resolves the git-history credential exposure tracked in
**issue #219**.

**Scope of this runbook:** rotation of the two live data-store credentials that are
present in git history:

| # | Credential                  | Leaked value (in history) |
|---|-----------------------------|---------------------------|
| 1 | Postgres `POSTGRES_PASSWORD`| `9v1g^OV9XUwzUP6cEgCYgNOE` |
| 2 | Redis `REDIS_PASSWORD`      | `RsYque&Xh%TUD*Nv^7k7B8X3` |

**Prior art / cross-references:**
- A1 runbook Step 1: `docs/audits/2026-04/_synthesis/_meta/artifacts/A1-rotation-runbook.md`
  (broader 9-credential rotation; this runbook is the PG+Redis-only execution slice
  that closes #219).
- Workpaper: `docs/audits/2026-04/_synthesis/workpaper/A.md` §3 Step 1.

> ### ⚠️ This rotation is the ONLY thing that resolves #219
> Removing the literals from the working tree (the companion file-scrub PR) does
> **NOT** resolve the exposure. The secrets remain readable in git history, in any
> clone, and in any fork until the live credentials they authenticate are rotated and
> the old credentials revoked. Do not close #219 on the basis of a clean working tree.
> #219 closes only after this runbook completes Step 7 (old credentials revoked, new
> credentials confirmed live) for **both** stores.

---

## 0. Operating principles (read before executing)

1. **Pre-stage both old AND new credentials in the secret store first.** Nothing in
   the live systems is touched until the new secret exists alongside the old one in
   1Password (vault `Personal`, item `audit-2026-04/<credential-name>`).
2. **PG and Redis are independent, reversible units.** Rotate them one at a time. A
   failure in one store must never force a change in the other. See the abort matrix
   (§5).
3. **The OLD secret for each store stays VALID until the very last step.** Revocation
   (`ALTER USER` / `CONFIG SET requirepass` + dropping the old auth) is the FINAL
   action, never performed mid-sequence. Until revoke, rollback is a single env
   repoint.
4. **Health-check gate + soak before proceeding.** After repointing to the new
   credential, health checks must be GREEN, then a soak/observation window must pass
   with ZERO auth-failure errors (§4).
5. **Pre-revoke log-redaction check.** Before revoking, confirm the new secret values
   have not leaked into CI/deploy logs in plaintext (§6).
6. **Explicit human GO/NO-GO at revoke.** The revoke step (§7) is the only step that
   requires live synchronous operator presence.
7. **Record the pre-rotation state at every step** so each step is individually
   reversible (§8 rollback).

---

## 1. Pre-rotation safety checklist

- [ ] Maintenance window scheduled. Worst-case DB reconnect blip; size the window for
      one full ingest+API cycle plus margin (see soak, §4).
- [ ] Backups taken: Postgres snapshot, Redis RDB/AOF snapshot. Record snapshot IDs.
- [ ] Inventory of consumers confirmed for each store (below). A consumer missed here
      is a consumer that keeps using the old credential and silently blocks revoke.
- [ ] **NEW** PG password provisioned in 1Password: `op://Personal/audit-2026-04/postgres-password-new`.
- [ ] **NEW** Redis password provisioned in 1Password: `op://Personal/audit-2026-04/redis-password-new`.
- [ ] **OLD** values also stored in 1Password (`…/postgres-password-old`,
      `…/redis-password-old`) so rollback does not depend on git history.
- [ ] Operator has merge/deploy access for the window.
- [ ] Generate the new secrets with CSPRNG, e.g.
      `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.

### Consumer inventory (who reads each credential)

**Postgres** — backend API (`DATABASE_URL`), ETL / data-pipeline jobs, Alembic
migrations, Airflow workers, any operational scripts, staging, CI.

**Redis** — backend cache, Celery / Redis task consumers, the distributed
rate-limiter, session storage, feature store.

> The deploy environment(s) reference these via `DATABASE_URL` / `REDIS_URL` (and
> `POSTGRES_PASSWORD` / `REDIS_PASSWORD`). Repointing means updating those env values
> in the secret store / deploy config and rolling the relevant services.

---

## 2. Rotation order

Rotate **Postgres first, then Redis** — fully completing PG (through soak) before
starting Redis. They are independent units; doing them sequentially keeps the abort
matrix (§5) to two simple, isolated states instead of a combined one.

Each store independently walks Steps 3 → 4 → 6 → 7 below.

---

## 3. Pre-stage and dual-validity (per store)

Goal: reach a state where BOTH old and new credentials authenticate, so the env
repoint in Step 3b is a no-downtime cutover and rollback is instant.

### 3a. Postgres — create the new authentication alongside the old

Provision the new password so both work simultaneously. Preferred mechanism is a
parallel app role with the new password (old role untouched):

```sql
-- Option A (preferred): new role, both valid simultaneously
CREATE ROLE app_new LOGIN PASSWORD '<NEW_PG_PASSWORD>';
GRANT <same privileges as current app role> TO app_new;
-- Old role/password remains fully valid. Nothing is revoked here.
```

If a parallel role is not feasible and the same role must be reused, schedule the
`ALTER USER … PASSWORD` for the revoke step (§7) instead — do **not** change the live
password here, because that would break the old credential mid-sequence (principle 3).

Record current state for rollback:

```bash
# Capture the pre-rotation DATABASE_URL / role in 1Password before any change.
```

### 3b. Postgres — repoint the deploy env to the NEW credential

- Update `DATABASE_URL` / `POSTGRES_PASSWORD` in the deploy secret store to the new
  value. Roll the services that read it (rolling restart preferred).
- The OLD credential is still valid at this point — this is a forward cutover, fully
  reversible by repointing back.

Proceed to the health-check + soak gate (§4) for Postgres. Do not start Redis until
Postgres has passed soak.

### 3c. Redis — create the new authentication alongside the old

Redis `requirepass` is a single global password, so true dual-validity is limited.
Two supported approaches:

- **ACL approach (Redis 6+, preferred):** add a new ACL user with the new password,
  keep the `default` user valid, repoint consumers to the new user, soak, then in §7
  disable/repassword the old `default` auth.
  ```
  ACL SETUSER app_new on >'<NEW_REDIS_PASSWORD>' ~* +@all
  ```
- **requirepass approach (fallback):** because flipping `requirepass` breaks the old
  password instantly, treat the flip itself as the revoke action and run it inside the
  §7 GO/NO-GO window. Pre-stage by repointing every consumer's config to read the new
  password from the secret store first, so the actual flip is a single coordinated
  cutover.

### 3d. Redis — repoint the deploy env to the NEW credential

- Update `REDIS_URL` / `REDIS_PASSWORD` in the deploy secret store. Roll consumers.
- With the ACL approach the old `default` auth is still valid (reversible). With the
  requirepass fallback, the cutover happens in §7.

Proceed to §4 for Redis.

---

## 4. Health-check gate + SOAK window (per store)

After repointing a store's deploy env to the new credential:

1. **Health checks GREEN.** Run application health checks and confirm the relevant
   component is healthy on the new credential:
   ```bash
   curl https://<api>/health   # 200, all components green, no DB/Redis auth errors
   ```
2. **SOAK / observation window.** Hold and observe with **ZERO auth-failure errors**
   before proceeding. The soak window MUST be at least:
   - `>=` the maximum connection-pool TTL / `pool_recycle` (so every pooled connection
     has been re-established under the new credential), AND
   - `>=` one full ingest + API cycle (so batch jobs, Celery beat / scheduled tasks,
     and ETL runs have each exercised the new credential at least once).

   **Soak duration: `__MAINTAINER_INPUT__` minutes.**
   _Maintainer: set this to the larger of your DB/Redis pool TTL and one full
   ingest+API cycle, plus margin. Do not leave as a placeholder when executing._

3. **Zero-auth-failure assertion.** Grep logs/metrics for auth failures during the
   soak. Any of these aborts the step (see §5):
   - Postgres: `password authentication failed`, `FATAL: ... role`, connection-pool
     auth errors.
   - Redis: `NOAUTH`, `WRONGPASS`, `invalid password`.

Only after a clean soak does the store advance to the log-redaction check (§6) and
then revoke (§7).

---

## 5. Abort matrix (independent reversible units)

Because PG and Redis are rotated independently and the old secret stays valid until
§7, every pre-revoke failure recovers by repointing the affected store's env back to
its **still-valid OLD secret**. Recovery never touches the other store.

| State | What happened | Recovery action |
|-------|---------------|-----------------|
| **PG repointed, PG soak failing (auth errors / unhealthy)** | New PG credential not working for all consumers | Repoint `DATABASE_URL` / `POSTGRES_PASSWORD` back to the **still-valid OLD** PG secret; roll services. Redis untouched. Investigate before retry. |
| **PG rotated OK, Redis repointed but Redis soak failing** | PG is fine; Redis new credential failing | Repoint `REDIS_URL` / `REDIS_PASSWORD` back to the **still-valid OLD** Redis secret; roll consumers. **Leave PG on its new credential** — PG is an independent completed unit; do NOT roll PG back. Investigate Redis before retry. |
| **Redis rotated OK, PG repointed but PG soak failing** (if order reversed) | Redis is fine; PG new credential failing | Repoint PG env back to the **still-valid OLD** PG secret; roll services. **Leave Redis on its new credential.** Investigate PG before retry. |
| **Either store: failure discovered AFTER its old auth was revoked (§7)** | Old credential already dropped | This is why revoke is last + gated. Recover by re-creating the credential from the 1Password `…-new` item (still the live one) and, if needed, the `…-old` item; restore from the pre-rotation snapshot only as a last resort. Do NOT proceed to the other store until resolved. |

Key invariant: **a failure in one store is recovered using only that store's
still-valid old secret; the other store is never rolled back as a side effect.**

---

## 6. Pre-revoke log-redaction check (per store, gates §7)

Before revoking the old credential, confirm the **new** secret value has not leaked in
plaintext anywhere a reader could see it.

1. **Grep staging/deploy GitHub Actions logs** and health/diagnostic output for the new
   secret value:
   ```bash
   # Pull recent workflow run logs and search for the new secret literal.
   gh run list --limit 30
   gh run view <run-id> --log | grep -F '<NEW_SECRET_VALUE>'   # MUST return nothing
   # Also check any health endpoint / connection-info output captured in logs.
   ```
2. **Confirm GitHub Actions secret masking covers the new values.** The new PG/Redis
   secrets must be stored as GitHub Actions secrets (or injected via the secret store)
   so Actions masks them as `***` in logs. Verify a deliberate echo in a scratch run is
   masked.
3. **Zero plaintext leakage required.** Only when the grep returns nothing and masking
   is confirmed does the store proceed to revoke. If a plaintext leak is found, treat
   the new secret as compromised: provision another new secret, repeat §3–§4, fix the
   logging path that leaked it, and re-run this check.

> Note: the companion code change masks `show_connection_info()` / connection-string
> output so connection strings print redacted. This check is the runtime confirmation
> that the masking holds end-to-end.

---

## 7. Revoke old credential — GO/NO-GO (per store)

**This is the only step requiring live, synchronous operator presence.** It is also the
step that actually closes the exposure for that store.

**Pre-conditions (all must be true):**
- [ ] Health checks GREEN on the new credential.
- [ ] Soak window completed with ZERO auth failures (§4).
- [ ] Log-redaction check passed — zero plaintext leakage, masking confirmed (§6).
- [ ] Rollback path verified (old + new secrets both in 1Password; snapshot IDs
      recorded).

**Operator GO/NO-GO:** state explicitly `GO` or `NO-GO`. On NO-GO, stop and keep the
old credential valid.

On **GO**, revoke the old authentication:

```sql
-- Postgres: drop / invalidate the old auth.
-- If using the parallel-role approach: revoke + drop the old role.
DROP ROLE app_old;        -- or: ALTER ROLE app_old NOLOGIN;
-- If reusing the same role: this is where ALTER USER … PASSWORD '<NEW>' happens.
```

```
# Redis: ACL approach — remove/disable old default auth.
ACL SETUSER default off            # or set default to the new password
# requirepass fallback — perform the coordinated flip now:
CONFIG SET requirepass '<NEW_REDIS_PASSWORD>'
CONFIG REWRITE
```

**Post-revoke verification (old credential MUST now fail):**

```bash
# Old PG password must fail
PGPASSWORD='9v1g^OV9XUwzUP6cEgCYgNOE' psql -h <prod> -U postgres -d investment_db -c '\q'
#   MUST FAIL with auth error

# Old Redis password must fail
redis-cli -h <prod> -a 'RsYque&Xh%TUD*Nv^7k7B8X3' ping
#   MUST FAIL (NOAUTH / WRONGPASS)

# New credential still healthy
curl https://<api>/health   # 200, all green
```

When BOTH stores have completed §7 and the old credentials are confirmed dead, the
git-history exposure for these two credentials is resolved → **#219 can be closed**,
referencing the verification output above.

---

## 8. Rollback (every step individually reversible)

| Step | Pre-rotation state recorded | Rollback |
|------|-----------------------------|----------|
| §3 pre-stage | Old credential value (1Password `…-old`) | Drop the new role / ACL user; nothing else changed. |
| §3b/§3d repoint | Prior `DATABASE_URL` / `REDIS_URL` in secret store | Repoint env back to old secret; roll services. |
| §4 soak fail | — | Repoint to old secret (§5 abort matrix). |
| §7 revoke | Snapshot IDs + both secrets in 1Password | Re-create credential from 1Password; restore snapshot only as last resort. |

Record the pre-rotation `DATABASE_URL`, `REDIS_URL`, role names, and snapshot IDs in
1Password BEFORE starting, so rollback never depends on git history.

---

## 9. KDF rotation (F-08-001) — OPT-IN follow-up, NOT bundled

Finding **F-08-001** concerns the key-derivation function used by the secrets manager
(`backend/security/secrets_manager.py` — fixed-salt PBKDF2). Rotating the KDF / salt is
a **separate, opt-in maintenance activity** with its own window and abort path.

**It is intentionally NOT part of the #219-closing PG/Redis rotation**, because:
- It forces re-derivation / re-encryption of any data encrypted under the old KDF.
- Bundling it would couple a simple, reversible data-store credential swap to a
  data-migration with a heavier abort path — and could delay the #219 close.

**If/when KDF rotation is undertaken (separately):**
- Schedule its own window; do not run concurrently with PG/Redis rotation.
- Re-encrypt all data encrypted under the old KDF/salt with the new one **before**
  retiring the old key (mirror the Fernet re-encryption discipline in the A1 runbook).
- Its own abort path: keep the old KDF/key able to decrypt until re-encryption is
  verified complete; only then retire it.

Do not let KDF complications block or delay the PG/Redis rotation that closes #219.

---

## 10. Findings / issue coverage

- **issue #219** — live PG + Redis credentials in git history. Closed by §1–§8 of this
  runbook (rotation + revoke), NOT by the working-tree scrub.
- **F-08-001** — KDF rotation, deferred to §9 as an opt-in follow-up.
- Broader credential set (Elasticsearch decommissioned; Grafana, Airflow, JWT, Fernet,
  Google, HuggingFace): see the A1 runbook. Out of scope here.

## 11. Park log

- (authored) — PG+Redis rotation runbook created as the #219-closing procedure,
  awaiting maintenance window. Soak duration left as `__MAINTAINER_INPUT__` for the
  operator to size against pool TTL + one ingest/API cycle.
