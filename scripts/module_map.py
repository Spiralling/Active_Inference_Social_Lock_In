"""Map and check internal imports for the project.

This is intentionally dependency-free. It parses Python files under ``src/`` with
``ast`` and reports imports whose target starts with ``src``. The default output is
Markdown so it can be pasted into notes; ``--check`` turns the architecture rules
into a lightweight regression guard.

Usage:
    python scripts/module_map.py
    python scripts/module_map.py --format dot > results/module_map.dot
    python scripts/module_map.py --check
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
LAYERS = ("root", "structural", "pomdp", "unified")


@dataclass(frozen=True, order=True)
class ImportEdge:
    source: str
    target: str
    line: int


def module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def discover_modules() -> set[str]:
    return {module_name(path) for path in SRC.rglob("*.py")}


def layer_of(module: str) -> str:
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == "src" and parts[1] in LAYERS[1:]:
        return parts[1]
    if parts and parts[0] == "src":
        return "root"
    return "external"


def resolve_relative(source: str, level: int, module: str | None) -> str | None:
    if level == 0:
        return module
    parts = source.split(".")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    package = parts[: max(0, len(parts) - level)]
    if module:
        package.extend(module.split("."))
    return ".".join(package) if package else None


def best_internal_target(base: str, alias: str | None, modules: set[str]) -> str | None:
    if not base.startswith("src"):
        return None
    candidate = f"{base}.{alias}" if alias else base
    if candidate in modules:
        return candidate
    if f"{candidate}.__init__" in modules:
        return candidate
    if base in modules or f"{base}.__init__" in modules:
        return base
    parts = candidate.split(".")
    while len(parts) > 1:
        trial = ".".join(parts)
        if trial in modules or f"{trial}.__init__" in modules:
            return trial
        parts.pop()
    return base


def parse_imports(path: Path, modules: set[str]) -> list[ImportEdge]:
    source = module_name(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    edges: list[ImportEdge] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = best_internal_target(alias.name, None, modules)
                if target:
                    edges.append(ImportEdge(source, target, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative(source, node.level, node.module)
            if not base:
                continue
            for alias in node.names:
                if alias.name == "*":
                    target = best_internal_target(base, None, modules)
                else:
                    target = best_internal_target(base, alias.name, modules)
                if target:
                    edges.append(ImportEdge(source, target, node.lineno))
    return edges


def collect_edges() -> list[ImportEdge]:
    modules = discover_modules()
    edges: list[ImportEdge] = []
    for path in sorted(SRC.rglob("*.py")):
        edges.extend(parse_imports(path, modules))
    return sorted(set(edges))


def layer_edges(edges: list[ImportEdge]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        src_layer = layer_of(edge.source)
        tgt_layer = layer_of(edge.target)
        if src_layer != tgt_layer:
            counts[(src_layer, tgt_layer)] += 1
    return dict(sorted(counts.items()))


def contract_violations(edges: list[ImportEdge]) -> list[str]:
    violations: list[str] = []
    for edge in edges:
        src_layer = layer_of(edge.source)
        tgt_layer = layer_of(edge.target)
        if src_layer == "root" and tgt_layer in {"structural", "pomdp", "unified"}:
            violations.append(
                f"root module {edge.source}:{edge.line} imports downward into {edge.target}"
            )
        if src_layer == "structural" and tgt_layer in {"pomdp", "unified"}:
            violations.append(
                f"structural module {edge.source}:{edge.line} imports {edge.target}"
            )
        if src_layer == "pomdp" and tgt_layer in {"structural", "unified"}:
            violations.append(
                f"pomdp module {edge.source}:{edge.line} imports {edge.target}"
            )
        if src_layer == "unified" and tgt_layer not in {"root", "structural", "pomdp", "unified"}:
            violations.append(
                f"unified module {edge.source}:{edge.line} imports unexpected layer {edge.target}"
            )
    return violations


def render_markdown(edges: list[ImportEdge]) -> str:
    by_source: dict[str, list[ImportEdge]] = defaultdict(list)
    for edge in edges:
        by_source[edge.source].append(edge)

    lines = [
        "# Module Import Map",
        "",
        "Generated by `python scripts/module_map.py`.",
        "",
        "## Layer Edges",
        "",
        "| Source layer | Target layer | Imports |",
        "|---|---|---:|",
    ]
    for (src_layer, tgt_layer), count in layer_edges(edges).items():
        lines.append(f"| `{src_layer}` | `{tgt_layer}` | {count} |")

    violations = contract_violations(edges)
    lines.extend(["", "## Contract Check", ""])
    if violations:
        lines.extend(f"- {item}" for item in violations)
    else:
        lines.append("No violations found.")

    lines.extend(["", "## Module Imports", ""])
    for source in sorted(by_source):
        targets = sorted({edge.target for edge in by_source[source] if edge.target != source})
        if targets:
            lines.append(f"- `{source}` -> " + ", ".join(f"`{target}`" for target in targets))
    return "\n".join(lines) + "\n"


def render_dot(edges: list[ImportEdge]) -> str:
    lines = ["digraph imports {", "  rankdir=LR;"]
    for edge in edges:
        if edge.source != edge.target:
            lines.append(f'  "{edge.source}" -> "{edge.target}";')
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("md", "dot", "json"), default="md")
    parser.add_argument("--check", action="store_true", help="fail on layer contract violations")
    args = parser.parse_args()

    edges = collect_edges()
    if args.format == "md":
        print(render_markdown(edges), end="")
    elif args.format == "dot":
        print(render_dot(edges), end="")
    else:
        print(json.dumps([edge.__dict__ for edge in edges], indent=2))

    violations = contract_violations(edges)
    if args.check and violations:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
