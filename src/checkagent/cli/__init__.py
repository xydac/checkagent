"""CheckAgent CLI — command-line interface for the testing framework."""

import click

from checkagent import __version__


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="checkagent")
@click.pass_context
def main(ctx: click.Context) -> None:
    """The open-source testing framework for AI agents."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Import subcommands to register them with the group
from checkagent.cli.analyze_prompt import analyze_prompt_cmd  # noqa: E402
from checkagent.cli.ci_init import ci_init_cmd  # noqa: E402
from checkagent.cli.demo import demo_cmd  # noqa: E402
from checkagent.cli.history_cmd import history_cmd  # noqa: E402
from checkagent.cli.import_trace import import_trace_cmd  # noqa: E402
from checkagent.cli.init import init_cmd  # noqa: E402
from checkagent.cli.migrate import migrate_cmd  # noqa: E402
from checkagent.cli.run import run_cmd  # noqa: E402
from checkagent.cli.scan import scan_cmd  # noqa: E402
from checkagent.cli.wrap import wrap_cmd  # noqa: E402

main.add_command(analyze_prompt_cmd)
main.add_command(ci_init_cmd)
main.add_command(demo_cmd)
main.add_command(history_cmd)
main.add_command(import_trace_cmd)
main.add_command(init_cmd)
main.add_command(migrate_cmd)
main.add_command(run_cmd)
main.add_command(scan_cmd)
main.add_command(wrap_cmd)
