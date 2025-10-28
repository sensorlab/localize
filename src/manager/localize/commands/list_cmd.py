from __future__ import annotations

from pathlib import Path

import click

from ..dvc_utils import (
    find_experiments,
    get_experiment,
    list_stages_status_ordered,
)


@click.group()
def list_group():
    """List avaliable experiments."""


@list_group.command("experiments")
@click.pass_context
def list_experiments(ctx: click.Context):
    """List all experiments under CONFIGS_DIR."""
    configs_dir: Path = ctx.obj["configs_dir"]
    exps = find_experiments(configs_dir)
    if not exps:
        click.echo("No experiments found.")
        return
    click.echo(f"Experiments in {configs_dir}:")
    for e in exps:
        click.echo(f" - '{click.style(e.name, bold=True)}'")


# commands/list_cmd.py


@list_group.command("stages")
@click.argument("experiment")
@click.option("--disabled", "filter_", flag_value="disabled", default=False)
@click.option("--enabled", "filter_", flag_value="enabled", default=False)
@click.option("--all", "filter_", flag_value="all", default=True)
@click.pass_context
def list_stages(ctx, experiment, filter_):
    configs_dir: Path = ctx.obj["configs_dir"]
    exp = get_experiment(configs_dir, experiment)
    rows = list_stages_status_ordered(exp.path)
    if not rows:
        click.echo("No stages found.")
        return
    if not filter_:
        filter_ = "enabled"
    click.echo(f"Stages in {click.style(exp.name, bold=True)}:")
    for name, status in rows:
        if filter_ != "all" and status != filter_:
            continue
        mark = "✓" if status == "enabled" else "◻"
        color = "green" if status == "enabled" else "yellow"
        click.echo(f"{mark} {click.style(name, fg=color)}  ({status})")
