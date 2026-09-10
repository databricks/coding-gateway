# OS-Managed Agent Settings

## Summary

Claude Code and Codex give OS-managed settings higher precedence than user settings. Previously,
ucode could write only its local configuration while an existing machine-managed file silently
overrode the gateway endpoint, authentication helper, provider headers, or model.

The two stacked PRs make precedence handling deterministic:

1. The Claude PR adds the shared managed-file lifecycle and applies it to Claude Code.
2. The Codex PR reuses that lifecycle for TOML, applies it to Codex, and removes the old optional
   managed-settings path.

After both PRs merge, interactive configuration reconciles the agent's OS-managed file by default.
Non-interactive and CI execution never elevates privileges and instead uses local settings when the
managed file is compatible.

## Configuration Files

| Agent | Local ucode configuration | OS-managed configuration |
| --- | --- | --- |
| Claude Code | `~/.claude/ucode-settings.json` | Linux: `/etc/claude-code/managed-settings.json`; macOS: `/Library/Application Support/ClaudeCode/managed-settings.json` |
| Codex | `~/.codex/ucode.config.toml` | `/etc/codex/managed_config.toml` |

The local file is always written. The OS-managed file is additionally reconciled during interactive
configuration, except for Claude subscription relay.

## Interactive Detection

An invocation may modify OS-managed settings only when standard input is a TTY. Standard output does
not affect the decision, so piping logs does not disable an otherwise interactive configuration.
CI, pipes, cron jobs, and headless subprocesses normally have non-TTY standard input and therefore
remain local-only.

This is a new shared ucode distinction. The previous implementation inferred interactivity from
command shape in some flows and did not guard managed-file writes consistently.

## Behavior Matrix

| Invocation | Managed file | Behavior |
| --- | --- | --- |
| Interactive | Absent | Create it from the ucode configuration after recording an absent baseline. |
| Interactive | Unrelated or partially populated | Preserve unrelated values and add or update all ucode-owned values. |
| Interactive | Conflicting | Back up the baseline, replace the conflicting ucode-owned values, and verify. |
| Interactive | Already identical | Continue without a backup, write, or `sudo` invocation. |
| Non-interactive | Absent | Use the local ucode file. Do not create the managed file. |
| Non-interactive | Ucode-owned values absent or equal | Use the local ucode file. Do not modify the managed file. |
| Non-interactive | Ucode-owned value conflicts | Stop before launching because the higher-precedence value would override ucode. |
| Any | Invalid, unreadable, or symlinked | Stop without modifying the file because precedence cannot be established safely. |

`ucode configure`, first-time `ucode claude` or `ucode codex`, and later launches all use the same
agent-specific reconciliation path. A first-time launch from an interactive terminal can therefore
request administrator permission. A first-time non-interactive launch remains local-only.

## Interactive Reconciliation

For each agent, ucode:

1. Strictly parses the existing managed JSON or TOML document.
2. Produces the desired document by applying the same gateway overlay used for the local ucode file.
3. Preserves settings outside the paths owned by ucode.
4. Preserves enterprise Claude permission-deny entries while adding ucode-required entries.
5. Records the original baseline before the first change.
6. Requests administrator permission and performs an atomic privileged replacement.
7. Reads the installed file back and verifies its exact contents.
8. Records the last-applied snapshot, owned paths, and a launch fingerprint.

An existing managed file is reconciled even when it does not currently conflict. This ensures every
ucode-required value exists at the highest-precedence scope and avoids separate behavior for absent,
partial, and conflicting files.

## Privileged Write Transaction

The shared writer handles ordinary MDM-installed and root-owned files rather than treating them as
errors:

- refuses symlink destinations;
- creates a root-owned destination directory when it is absent;
- stages the new file in the destination directory for a same-filesystem atomic rename;
- clones an existing file before replacing its contents, preserving ownership, mode, ACLs, and
  extended attributes;
- creates new files as root-owned and world-readable;
- detects, clears, and restores macOS `schg`, `uchg`, `sappnd`, and `uappnd` flags;
- detects, clears, and restores Linux immutable and append-only attributes;
- verifies the resulting contents after replacement;
- retries once only when device management restored the exact pre-write contents;
- preserves a concurrently changed policy instead of overwriting it.

The privilege boundary is interactive. Non-interactive paths do not invoke either normal `sudo` or
`sudo -n`.

Root access cannot sustainably override an actively enforced policy. If an MDM process immediately
restores the original file twice, ucode stops with an error instead of repeatedly fighting the
management agent. A different concurrent update is also preserved and reported.

## Backup Model

Backups live under `~/.ucode/managed-backups/` with directory mode `0700` and file mode `0600`.
The manifest records, per agent:

- whether the managed file originally existed;
- its absolute path;
- the baseline snapshot and SHA-256 digest;
- the last ucode-applied snapshot and SHA-256 digest;
- the setting paths owned by ucode.

The baseline is created before the first managed change and is not replaced by subsequent
configurations. Later configurations update only the last-applied snapshot. Snapshot filenames are
validated, digests are checked before use, and symlinked backup directories or manifests are
rejected.

If ucode cannot complete a write or revert, the backup remains available for a later retry.

## `ucode revert`

`ucode revert` extends the existing local-file restoration with managed-file restoration:

- If the current file exactly matches ucode's last-applied snapshot, restore the exact baseline.
- If ucode created the file, delete it.
- If an administrator or MDM changed the file later, perform a three-way revert.
- Restore original values only where the current value still matches ucode's last-applied value.
- Remove only list entries added by ucode while preserving externally added entries.
- Preserve externally changed values rather than replacing them with stale baseline values.
- Refuse an unsafe or unparsable merge and retain the backup.

A revert that needs to change an OS-managed file must run interactively. A successful revert removes
that agent's backup record. The existing local configuration and ucode state cleanup still occur as
part of the command.

## Cached Launches

After a successful managed reconciliation or compatibility check, ucode stores:

- the managed path;
- the verification scope;
- device and inode numbers;
- size;
- nanosecond modification and change times.

A cached launch performs one `stat()` call and compares this fingerprint. When it matches, ucode
does not read, parse, back up, write, or invoke `sudo`. When it changes, the normal reconciliation or
compatibility path runs again. This catches later MDM replacement without adding meaningful latency
to unchanged launches.

The verification scopes distinguish:

- an interactively reconciled managed file;
- a managed file verified as compatible with local settings;
- a managed file verified as compatible with Claude relay.

A compatibility fingerprint created non-interactively cannot suppress the next interactive
reconciliation.

## Claude Subscription Relay

Claude relay is intentionally local-only. It routes through a loopback refresh proxy whose address
and lifetime belong to one `ucode claude` session. Persisting that address in OS-managed settings
would break bare `claude` launches and future sessions.

Relay therefore never creates or updates the managed file. It allows an absent file or unrelated
enterprise settings, but blocks these higher-precedence conflicts:

- `apiKeyHelper`;
- `env.ANTHROPIC_BASE_URL`;
- `env.ANTHROPIC_CUSTOM_HEADERS`.

If ucode wrote those values during an earlier standard configuration, the user runs `ucode revert`
interactively before switching to relay. External conflicting values require administrator action or
standard Databricks authentication.

## Status and Messages

`ucode status` reports, for Claude and Codex:

- managed settings path;
- state: not configured, current, compatible local settings, compatible relay settings, drifted,
  invalid, unreadable, missing, or unsupported;
- whether a managed baseline backup is available.

Interactive updates announce the backup location, administrator-permission request, and verified
result. An identical file produces no elevation message.

Representative blockers are:

```text
Claude Code configuration cannot be applied non-interactively because OS-managed settings at
<path> override ucode values: env.ANTHROPIC_BASE_URL. Run `ucode configure --agent claude` from an
interactive terminal or contact your administrator.
```

```text
Claude Code managed settings at <path> were updated but immediately restored by device management.
Contact your administrator.
```

```text
Cannot safely update Codex managed settings at <path>: <parse error>. ucode did not modify the file.
Repair it or contact your administrator.
```

Write, verification, parse, symlink, and managed-conflict failures block the agent launch. Recovery
information always identifies the file and recommends either an interactive configure/revert or
administrator help.

## PR Boundaries

### PR 1: Claude Code

- Add shared strict reads, fingerprints, compatibility checks, secure backups, atomic privileged
  writes, immutable-flag handling, verification, status, and three-way revert helpers.
- Make interactive Claude configuration reconcile OS-managed JSON by default.
- Add non-interactive local fallback with conflict detection.
- Add Claude relay-specific safety checks.
- Add Claude managed status and revert output.
- Remove Claude's old managed-settings scope choice.

### PR 2: Codex

- Stack on the Claude PR and reuse the shared lifecycle with strict TOML parsing and serialization.
- Make interactive Codex configuration reconcile OS-managed TOML by default.
- Add non-interactive local fallback with conflict detection.
- Add Codex fingerprinted cached launches, status, and revert behavior.
- Keep opt-in smart-routing hooks in the local Codex config rather than adding them by default to
  machine-managed policy.
- Remove the remaining managed-settings scope schema, resolution, setup prompt, summary, and tests.
