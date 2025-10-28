from __future__ import annotations

from pathlib import Path

import click
from ruamel.yaml.comments import CommentedMap

from ..dvc_utils import (
    _active_file,
    _disabled_path,
    _dump_doc,
    _ensure_disabled_map,
    _get_and_drop_pos_hint_from_node,
    _load_disabled_doc,
    _load_doc,
    _set_pos_hint_in_node,
    get_experiment,
    parse_selector,
)
from .list_cmd import list_stages as list_stages_cmd


def disable_stage_yaml(exp_dir: Path, stage_name: str, dry_run: bool = False) -> bool:
    # main dvc.yaml
    dvc_path = _active_file(exp_dir)
    dvc_doc = _load_doc(dvc_path)
    stages = dvc_doc.get("stages")
    if not isinstance(stages, CommentedMap):
        return False
    if stage_name not in stages:
        return False

    # determine adjacency hints before removal
    names = list(stages.keys())
    idx = names.index(stage_name)
    prev_name = names[idx - 1] if idx > 0 else None
    next_name = names[idx + 1] if idx < len(names) - 1 else None

    # pop exact node (preserves formatting/comments)
    node = stages.pop(stage_name)
    _set_pos_hint_in_node(node, prev_name, next_name)

    # sidecar disabled.yaml
    disabled_doc = _load_disabled_doc(exp_dir)
    disabled_map = _ensure_disabled_map(disabled_doc)
    disabled_map[stage_name] = node

    if dry_run:
        return True

    # write both files
    _dump_doc(dvc_path, dvc_doc)
    _dump_doc(_disabled_path(exp_dir), disabled_doc)
    return True


def enable_stage_yaml(exp_dir: Path, stage_name: str, force: bool = False, dry_run: bool = False) -> bool:
    # main dvc.yaml
    dvc_path = _active_file(exp_dir)
    dvc_doc = _load_doc(dvc_path)
    stages = dvc_doc.get("stages")
    if not isinstance(stages, CommentedMap):
        return False

    # sidecar disabled.yaml
    disabled_doc = _load_disabled_doc(exp_dir)
    disabled_map = _ensure_disabled_map(disabled_doc)
    if stage_name not in disabled_map:
        return False

    # stage node + placement hints
    node = disabled_map.pop(stage_name)
    prev_hint, next_hint = _get_and_drop_pos_hint_from_node(node)

    if stage_name in stages and not force:
        raise click.ClickException(f"Stage '{stage_name}' already enabled. Use --force to replace it.")
    if stage_name in stages and force:
        stages.pop(stage_name)

    # compute insertion index using hints
    names = list(stages.keys())
    if prev_hint in stages:
        insert_idx = names.index(prev_hint) + 1
    elif next_hint in stages:
        insert_idx = names.index(next_hint)
    else:
        insert_idx = len(names)

    stages.insert(insert_idx, stage_name, node)

    # keep a minimal, valid map in sidecar (optional: leave empty mapping)
    if len(disabled_map) == 0:
        disabled_doc["disabled_stages"] = CommentedMap()

    if dry_run:
        return True

    # write both files
    _dump_doc(dvc_path, dvc_doc)
    _dump_doc(_disabled_path(exp_dir), disabled_doc)
    return True


@click.group(help="Manage stages inside an experiment.")
def stage():
    pass


@stage.command("disable")
@click.argument("selector")  # EXPERIMENT:STAGE
@click.option("--dry", is_flag=True, help="Show what would change without writing.")
@click.pass_context
def disable_cmd(ctx: click.Context, selector: str, dry: bool):
    configs_dir: Path = ctx.obj["configs_dir"]
    exp_name, stage_name = parse_selector(selector)
    exp = get_experiment(configs_dir, exp_name)
    changed = disable_stage_yaml(exp.path, stage_name, dry_run=dry)
    if changed:
        msg = f"Disabled stage '{stage_name}' in {exp.name}"
        click.echo(click.style(msg + (" (dry)" if dry else ""), fg="yellow"))
    else:
        raise click.ClickException(f"Stage '{stage_name}' is not enabled (nothing to disable).")


@stage.command("enable")
@click.argument("selector")  # EXPERIMENT:STAGE
@click.option("--force", is_flag=True, help="Overwrite if an enabled stage with same name exists.")
@click.option("--dry", is_flag=True, help="Show what would change without writing.")
@click.pass_context
def enable_cmd(ctx: click.Context, selector: str, force: bool, dry: bool):
    configs_dir: Path = ctx.obj["configs_dir"]
    exp_name, stage_name = parse_selector(selector)
    exp = get_experiment(configs_dir, exp_name)
    changed = enable_stage_yaml(exp.path, stage_name, force=force, dry_run=dry)
    if changed:
        msg = f"Enabled stage '{stage_name}' in {exp.name}"
        click.echo(click.style(msg + (" (dry)" if dry else ""), fg="green"))
    else:
        raise click.ClickException(f"Stage '{stage_name}' is not disabled (nothing to enable).")


@stage.command("ls", help="List stages for an experiment with enabled/disabled status.")
@click.argument("experiment")
@click.option("--disabled", "filter_", flag_value="disabled", default=False, help="Show only disabled stages.")
@click.option("--enabled", "filter_", flag_value="enabled", help="Show only enabled stages.")
@click.option("--all", "filter_", flag_value="all", help="Show all stages.")
@click.pass_context
def ls_cmd(ctx: click.Context, experiment: str, filter_: str | None):
    # Delegate to the canonical implementation
    return ctx.invoke(list_stages_cmd, experiment=experiment, filter_=filter_)
