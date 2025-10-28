from __future__ import annotations

from pathlib import Path

import click

from ..dvc_utils import get_experiment, run_subprocess


@click.command()
@click.argument("experiment")
@click.argument("stage", required=False)
@click.option("--force", is_flag=True, help="Force rebuild stage(s).")
@click.option("--dry", is_flag=True, help="Show what would run without executing.")
@click.option("--no-commit", is_flag=True, help="Do not commit outputs to cache.")
@click.pass_context
def run_cmd(ctx: click.Context, experiment: str, stage: str | None, force: bool, dry: bool, no_commit: bool):
    """Reproduce an entire experiment pipeline, or a single STAGE within it."""
    configs_dir: Path = ctx.obj["configs_dir"]
    exp = get_experiment(configs_dir, experiment)

    # `dvc repro` in the experiment directory runs that experiment's DAG.
    cmd = ["dvc", "repro"] if stage is None else ["dvc", "repro", stage]
    if force:
        cmd.append("--force")
    if dry:
        cmd.append("--dry")
    if no_commit:
        cmd.append("--no-commit")

    run_subprocess(cmd, cwd=exp.path)
