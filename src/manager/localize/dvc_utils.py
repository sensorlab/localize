from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import click
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap


# ---------- Constants ----------

DVC_FILENAMES: tuple[str, ...] = ("dvc.yaml", "dvc.yml")
DISABLED_KEY = "disabled_stages"
SIDECAR_FILE = "localize.yaml"
POS_HINT_KEY = "x-manage-pos"  # used only under disabled_stages/<stage>

# ---------- YAML setup ----------

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096
_yaml.indent(mapping=2, sequence=4, offset=2)


# ---------- Data types ----------


@dataclass(frozen=True, slots=True)
class Experiment:
    """A DVC experiment folder with its active DVC file."""

    name: str
    path: Path
    dvc_file: Path


# ---------- YAML helpers ----------
def _load_doc(path: Path) -> CommentedMap:
    """Load YAML preserving order/comments; return empty map if file missing/empty."""
    if not path.exists():
        return CommentedMap()
    with path.open("r", encoding="utf-8") as f:
        doc = _yaml.load(f) or CommentedMap()
    if not isinstance(doc, CommentedMap):
        # Ensure we always return a mapping
        return CommentedMap(doc)
    return doc


def _dump_doc(path: Path, doc: CommentedMap) -> None:
    """Write YAML preserving order/comments."""
    with path.open("w", encoding="utf-8") as f:
        _yaml.dump(doc, f)


# ---------- Sidecar helpers ----------


def _disabled_path(exp_dir: Path) -> Path:
    return exp_dir / SIDECAR_FILE


def _load_disabled_doc(exp_dir: Path) -> CommentedMap:
    return _load_doc(_disabled_path(exp_dir))


def _ensure_disabled_map(doc: CommentedMap) -> CommentedMap:
    """Ensure and return the `disabled_stages` map inside disabled.yaml."""
    disabled = doc.get(DISABLED_KEY)
    if not isinstance(disabled, CommentedMap):
        disabled = CommentedMap()
        doc[DISABLED_KEY] = disabled
    return disabled


# ---------- DVC file discovery ----------


def _active_file(exp_dir: Path) -> Path:
    """Return the first existing DVC file in the preferred order, or default path."""
    for fname in DVC_FILENAMES:
        p = exp_dir / fname
        if p.exists():
            return p
    return exp_dir / DVC_FILENAMES[0]


def find_experiments(configs_dir: Path) -> list[Experiment]:
    """Find experiments (subdirs with a DVC file) in configs_dir."""
    exps: list[Experiment] = []
    for child in sorted(configs_dir.iterdir()):
        if not child.is_dir():
            continue
        dvc_file = next((child / f for f in DVC_FILENAMES if (child / f).is_file()), None)
        if dvc_file:
            exps.append(Experiment(child.name, child, dvc_file))
    return exps


def get_experiment(configs_dir: Path, name: str) -> Experiment:
    """Return an experiment by name, or raise a friendly Click error."""
    for exp in find_experiments(configs_dir):
        if exp.name == name:
            return exp
    raise click.ClickException(f"Experiment '{name}' not found in {configs_dir}")


# ---------- DVC content helpers ----------


def load_stages(dvc_file: Path) -> dict[str, dict]:
    """Return plain dict of stages from a DVC file (empty if missing)."""
    doc = _load_doc(dvc_file)
    stages = doc.get("stages")
    return dict(stages) if isinstance(stages, (dict, CommentedMap)) else {}


def _stage_names_in_order(stages_map: CommentedMap) -> list[str]:
    """Keep insertion order of the YAML mapping."""
    return list(stages_map.keys())


def _set_pos_hint_in_node(stage_node: CommentedMap, prev_name: str | None, next_name: str | None) -> None:
    """Store adjacency hints on a disabled stage node."""
    stage_node[POS_HINT_KEY] = {"prev": prev_name or "", "next": next_name or ""}


def _get_and_drop_pos_hint_from_node(stage_node: CommentedMap) -> tuple[str | None, str | None]:
    """Extract and remove adjacency hints from a disabled stage node."""
    d = stage_node.get(POS_HINT_KEY) or {}
    prev = d.get("prev") or None
    nxt = d.get("next") or None
    if POS_HINT_KEY in stage_node:
        del stage_node[POS_HINT_KEY]
    return prev, nxt


def parse_selector(selector: str) -> tuple[str, str]:
    """Parse 'EXPERIMENT:STAGE' into (experiment, stage) with validation."""
    left, sep, right = selector.partition(":")
    if sep != ":" or not left or not right:
        raise click.ClickException("Selector must be in the form EXPERIMENT:STAGE")
    return left, right


def list_stages_status_ordered(exp_dir: Path) -> list[tuple[str, str]]:
    """
    Return a list of (stage_name, status) where status is 'enabled' or 'disabled'.
    Disabled stages are interleaved according to adjacency hints (prev/next) if present.

    """
    dvc_path = _active_file(exp_dir)
    doc = _load_doc(dvc_path)

    stages = doc.get("stages")
    if not isinstance(stages, (dict, CommentedMap)):
        stages = CommentedMap()

    # Read disabled stages from sidecar
    sidecar_doc = _load_disabled_doc(exp_dir)
    disabled = sidecar_doc.get(DISABLED_KEY)
    if not isinstance(disabled, (dict, CommentedMap)):
        disabled = CommentedMap()

    order: list[tuple[str, str]] = []
    names_only: list[str] = []

    # 1) Enabled stages in their YAML order
    for name in stages.keys():
        if name in names_only:
            continue
        order.append((name, "enabled"))
        names_only.append(name)

    # 2) Disabled stages placed relative to hints, if available
    for name, node in disabled.items():
        if name in names_only:
            continue
        hint = node.get(POS_HINT_KEY) or {}
        prev_hint = hint.get("prev") or None
        next_hint = hint.get("next") or None

        if prev_hint in names_only:
            idx = names_only.index(prev_hint) + 1
        elif next_hint in names_only:
            idx = names_only.index(next_hint)
        else:
            idx = len(names_only)

        order.insert(idx, (name, "disabled"))
        names_only.insert(idx, name)

    return order


# ---------- Subprocess ----------


def run_subprocess(cmd: Sequence[str], cwd: Path | None = None) -> int:
    """
    Run a command, echoing a shell-like line. Raise ClickException on failure.
    Returns the process return code (0 on success).
    """
    click.echo(click.style(f"$ {' '.join(cmd)}", dim=True))
    try:
        proc = subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)
        return proc.returncode
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Command failed (exit {e.returncode}): {' '.join(cmd)}") from e
