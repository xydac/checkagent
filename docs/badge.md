# Safety Badge

CheckAgent can generate an SVG badge showing the result of a safety scan. The badge displays the number of probes that passed out of the total probes run — for example, **CheckAgent · 64/68 safe**.

Add it to your project's README so contributors and users can see the current safety posture at a glance.

## What the badge shows

The badge reflects the most recent `checkagent scan` run:

- **Left label:** `CheckAgent` (dark gray background)
- **Right value:** `N/M safe` where N is the number of passing probes and M is the total
- **Color:**
  - Green (`#44cc11`) — all or nearly all probes passed
  - Yellow (`#dfb317`) — partial failures (some probes failed)
  - Red (`#e05d44`) — significant failures

The badge does not embed any API keys or secrets — it is a static SVG file safe to commit to a public repository.

## Example badge

![CheckAgent Badge](https://img.shields.io/badge/CheckAgent-64%2F68_safe-44cc11?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz48L3N2Zz4=)

This is what a generated badge looks like. The color changes based on your score: green (90%+), yellow (70-89%), red (below 70%). The badge includes the CheckAgent name so it's recognizable across repos.

## Generating the badge

Pass `--badge <output-path>` to `checkagent scan`. CheckAgent writes the SVG file after the scan completes, regardless of whether the scan passes or fails.

```bash
checkagent scan my_agent:run --badge badge.svg
```

The badge file is written to `badge.svg` in the current directory. You can specify any path:

```bash
checkagent scan my_agent:run --badge docs/badge.svg
```

To also fail the command with a non-zero exit code when probes fail (useful in CI), combine with `--fail-under`:

```bash
checkagent scan my_agent:run --badge badge.svg --fail-under 0.9
```

## Serving the badge

### Option 1: Commit the badge to the repository

The simplest approach. Commit `badge.svg` to your repository and reference it with a relative path in your README. Update it by re-running the scan and committing the new file.

```bash
checkagent scan my_agent:run --badge badge.svg
git add badge.svg
git commit -m "Update safety badge"
git push
```

Reference it in your README:

```markdown
![CheckAgent](./badge.svg)
```

### Option 2: Serve via GitHub Actions artifact URL

Generate the badge in CI and upload it as a workflow artifact. GitHub provides a stable URL pattern for the latest artifact from the default branch.

**Workflow snippet** (upload only):

```yaml
- name: Run safety scan
  run: checkagent scan my_agent:run --badge badge.svg

- name: Upload badge artifact
  uses: actions/upload-artifact@v4
  with:
    name: safety-badge
    path: badge.svg
```

Reference the badge in your README using the artifact download URL:

```markdown
![CheckAgent](https://github.com/<owner>/<repo>/actions/workflows/<workflow-file>/badge.svg)
```

> Note: GitHub's built-in workflow status badge (`/badge.svg`) shows workflow pass/fail, not the CheckAgent probe count. To serve the CheckAgent SVG badge via a stable public URL, use Option 1 (commit to repo) or a separate badge hosting service.

## GitHub Actions workflow: generate and commit on every push

This workflow runs the safety scan on every push to `main`, regenerates the badge, and commits it back to the repository automatically.

```yaml
name: Safety Scan

on:
  push:
    branches: [main]

permissions:
  contents: write

jobs:
  safety:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          # Fetch with a token that has write access so the commit push works
          token: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install CheckAgent
        run: pip install checkagent

      - name: Run safety scan
        run: checkagent scan my_agent:run --badge badge.svg
        # Exit code reflects probe failures; do not block the badge commit step
        continue-on-error: true

      - name: Commit updated badge
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add badge.svg
          # Only commit if the badge changed
          git diff --cached --quiet || git commit -m "Update safety badge [skip ci]"
          git push
```

After the first run, add the badge to your `README.md`:

```markdown
![CheckAgent](./badge.svg)
```

The `[skip ci]` tag in the commit message prevents the push from triggering another workflow run.
