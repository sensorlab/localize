from __future__ import annotations

from pathlib import Path

import click

from .commands.list_cmd import list_group
from .commands.run_cmd import run_cmd
from .commands.stage_cmd import stage as stage_group


DEFAULT_CONFIGS_ENV = "CONFIGS_DIR"


@click.group()
@click.option(
    "--configs",
    "configs_dir",
    type=click.Path(file_okay=False, path_type=Path),
    envvar=DEFAULT_CONFIGS_ENV,
    default=None,
    help=f"Path to experiments root (defaults to ${DEFAULT_CONFIGS_ENV} or ./configs).",
)
@click.pass_context
def cli(ctx: click.Context, configs_dir: Path | None):
    """Control multiple DVC experiment pipelines under CONFIGS_DIR/EXPERIMENT/."""

    if configs_dir is None:
        configs_dir = Path.cwd() / "configs"
    configs_dir = configs_dir.resolve()

    if not configs_dir.exists():
        raise click.ClickException(f"Configs dir not found: {configs_dir}")

    ctx.ensure_object(dict)
    ctx.obj["configs_dir"] = configs_dir


# wire subcommands
cli.add_command(list_group, name="list")
cli.add_command(run_cmd, name="run")
cli.add_command(stage_group, name="stage")
