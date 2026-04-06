"""CLI command: checkagent migrate-cassettes.

Discovers cassette files and upgrades them to the current schema version.

Implements F2.6 (Cassette Versioning) from the PRD.
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from checkagent.replay.cassette import CASSETTE_SCHEMA_VERSION
from checkagent.replay.migration import migrate_directory

console = Console()


@click.command("migrate-cassettes")
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="cassettes",
)
@click.option("--no-backup", is_flag=True, help="Skip creating .bak files.")
@click.option(
    "--dry-run", is_flag=True, help="Show what would change without writing."
)
def migrate_cmd(directory: Path, no_backup: bool, dry_run: bool) -> None:
    """Upgrade cassette files to the current schema version."""
    if dry_run:
        console.print("[dim]Dry run - no files will be modified.[/dim]\n")

    results = migrate_directory(
        directory, backup=not no_backup, dry_run=dry_run
    )

    if not results:
        console.print(f"No cassette files found in {directory}.")
        return

    migrated = 0
    skipped = 0
    failed = 0

    for r in results:
        if not r.success:
            console.print(f"  [red]FAIL[/red] {r.path}: {r.error}")
            failed += 1
        elif r.from_version == r.to_version:
            skipped += 1
        else:
            action = "would migrate" if dry_run else "migrated"
            console.print(
                f"  [green]OK[/green] {r.path}: "
                f"v{r.from_version} -> v{r.to_version} ({action})"
            )
            migrated += 1

    console.print()
    console.print(
        f"Target: v{CASSETTE_SCHEMA_VERSION} | "
        f"Migrated: {migrated} | Skipped: {skipped} | Failed: {failed}"
    )
