# Prior-Reports Archive — Moved Out-of-Repo

Per PRD §2 Q8 (decision = A, recorded 2026-04-28), redacted prior-report archives
have been moved out of this repository to limit exposure if the repo's
visibility ever changes.

**Current location (durable, local):** `~/Documents/audit-archives-2026-04/`
**Access owner:** Devin McGrath
**File count at move time:** 125 redacted prior-report markdowns
**Total size:** ~1.2 MB

## Optional offsite migration

For an additional offsite/encrypted layer, the archives can be migrated to
1Password (recommended), private Google Drive, or a private S3 bucket. The
local `~/Documents` copy already satisfies Q8=A's "out-of-repo + access-controlled"
requirement (the home directory is private to the macOS user account); the
optional migration adds offsite backup and zero-knowledge encryption.

To migrate to 1Password (one Touch-ID prompt; requires 1Password 8 desktop
"Integrate with 1Password CLI" setting enabled):

```bash
# Sign in (Touch ID via desktop integration)
op signin

# Create a Document item in your Personal vault containing each archive file
for f in ~/Documents/audit-archives-2026-04/*.md; do
  op document create "$f" --title "audit-archives-2026-04/$(basename "$f")" --vault "Personal" >/dev/null
done

# Then update this README's "Current location" to point at the 1Password vault
```

## Original `/tmp` snapshot

The initial transition snapshot at `/tmp/audit-archives-2026-04/` is no longer
authoritative; the durable copy lives in `~/Documents/`. The `/tmp` snapshot
can be cleared at any time:

```bash
rm -rf /tmp/audit-archives-2026-04
```
