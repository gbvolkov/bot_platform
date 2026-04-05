# Mycroft Agent Working Rules

This file adds local rules for all work under `agents/mycroft_agent/`.
Root repository conventions still apply unless this file is stricter.

## Scope
- Primary code changes are allowed only in `agents/mycroft_agent/`.
- Cross-file support changes are allowed only in `/mycroft_agent_cli.py`.
- Tests for this agent may be added or updated only in unit or functional test files created for Mycroft work.
- Do not modify other agents, shared helpers, registry code, service code, or unrelated tests as part of Mycroft tasks.

## Implementation rules
- Keep simple things simple. Do not introduce extra layers, abstractions, or configuration without a concrete need.
- Do not add fallbacks.
- Do not add workaround branches.
- Do not silently degrade behavior when a dependency, tool, or environment setting is broken.
- If the environment is wrong, stop and report the issue clearly instead of compensating in code.
- Fix the direct problem only. Do not turn a local Mycroft task into a repo-wide refactor.

## Failure handling
- Treat missing credentials, missing binaries, broken permissions, invalid configuration, or unavailable services as environment issues.
- Report the exact failing dependency, command, or setting.
- Do not replace failing components with alternate providers, mock paths, optional behavior, or no-op behavior unless the user explicitly asks for that change.

## Editing guardrails
- Before editing, confirm the target files are inside the allowed scope.
- If a requested change appears to require edits outside the allowed scope, stop and ask the user.
- Prefer the smallest coherent patch that solves the issue.
- Preserve existing behavior unless the task explicitly changes it.

## Required control flow
- Before any file edit, print this block exactly and fill it with explicit values:

```text
PRE-EDIT CHECKLIST
target files: <explicit list>
outside-scope edits needed: no
fallback/workaround planned: no
environment issue detected: no
```

- Do not edit files before printing the checklist.
- If any required `no` becomes `yes`, stop and ask the user instead of editing.
- After edits, run the diff gate.
- Before concluding the task, print the review gate.

## Diff gate
- After each task, verify that every modified file is in the allowed scope.
- Use this PowerShell check or an equivalent stricter check:

```powershell
$allowedExact = @(
  "mycroft_agent_cli.py",
  "check-mycroft-scope.ps1"
)

$allowedPrefixes = @(
  "agents/mycroft_agent/",
  "tests/unit/test_mycroft_",
  "tests/functional/test_mycroft_"
)

$changed = git diff --name-only HEAD | ForEach-Object { $_.Replace("\", "/") }

$bad = @()
foreach ($file in $changed) {
  if ($allowedExact -contains $file) { continue }
  if (($allowedPrefixes | Where-Object { $file.StartsWith($_) }).Count -gt 0) { continue }
  $bad += $file
}

if ($bad.Count -gt 0) {
  Write-Host "Out-of-scope changes detected:"
  $bad | ForEach-Object { Write-Host "  $_" }
  exit 1
}

Write-Host "Scope check passed."
```

- If the diff gate fails, stop. Do not proceed with more edits until the user decides how to handle the out-of-scope files.

## Review gate
- Before finishing a task, print this block exactly:

```text
REVIEW GATE
scope respected: yes
fallbacks added: no
workarounds added: no
environment issues hidden in code: no
tests updated for changed behavior: yes/no
```

- If any line would be inaccurate, stop and report the problem.
- Do not describe workaround code as robustness, resilience, graceful degradation, compatibility handling, or optional behavior. Those still count as workarounds unless the user explicitly asked for them.

## Strong enforcement recommendation
- For near-hard enforcement, do Mycroft work in a separate worktree with a sparse checkout limited to:
  - `agents/mycroft_agent/`
  - `mycroft_agent_cli.py`
  - `tests/unit/`
  - `tests/functional/`
- Example setup:

```powershell
git worktree add C:\Projects\bot_platform_mycroft codex/mycroft-guard
Set-Location C:\Projects\bot_platform_mycroft
git sparse-checkout init --cone
git sparse-checkout set agents/mycroft_agent mycroft_agent_cli.py tests/unit tests/functional
```

- Stronger than that: use a container or sandbox where only the allowed paths are writable.

## Testing
- Add or update focused tests for Mycroft behavior when code changes justify it.
- Keep tests narrow and local to the changed behavior.
- Do not rewrite unrelated test structure.

## Review checklist
- Is this change still minimal?
- Did I avoid adding a fallback or workaround?
- Did I stop instead of coding around an environment problem?
- Are all modified files inside the allowed scope?
- Are tests limited to the changed Mycroft behavior?
