"""Rebased jagged tiles: ``base + tile.index`` over a relative tile extent.

A jagged row range can be written two ways::

    for st in hl.tile(s, e):
        out[st, kt] = ...
    for st in hl.tile(e - s):
        out[s + st.index, kt] = ...

Both walk the same contiguous rows ``[s, e)``.  The second spells the row offset
as a dynamic scalar carried in ``tile_with_offset`` metadata (see
``add_tile_with_offset_metadata``), which the Pallas backend resolves back to an
absolute interval so BlockSpecs, masks, grid sizing, and the ordered carry can
all use it.

The base is a *value*, not a name: each pipeline scope binds it to a different
local variable.  ``DeviceFunction.rebased_tile_bases`` records that binding as
the loop body is emitted, keyed by both the defining node and its name, and the
helpers here are the single place that reads it back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import torch

if TYPE_CHECKING:
    from helion._compiler.device_ir import GraphInfo
    from helion._compiler.inductor_lowering import CodegenState

_NEW_VAR = "_new_var"


def unwrap_new_var(node: object) -> object:
    """Strip the ``_new_var`` copies that carry a value into a nested scope."""
    while (
        isinstance(node, torch.fx.Node)
        and node.op == "call_function"
        and getattr(node.target, "__name__", None) == _NEW_VAR
        and len(node.args) == 1
        and isinstance(node.args[0], torch.fx.Node)
    ):
        node = node.args[0]
    return node


def base_name(state: CodegenState, offset: object) -> str | None:
    """Variable holding a rebased tile's dynamic base in the current scope.

    ``None`` when ``offset`` is not a dynamic base (an ordinary int/SymInt tile
    offset), or when no binding has been recorded for it -- callers then keep
    their non-rebased lowering rather than emitting a name that does not exist.
    """
    if isinstance(offset, str):
        return offset  # already resolved in place by the loop codegen
    if not isinstance(offset, torch.fx.Node):
        return None
    bases = state.device_function.rebased_tile_bases
    node = unwrap_new_var(offset)
    if isinstance(node, torch.fx.Node):
        name = bases.get(node) or bases.get(node.name)
        if name is not None:
            return name
    return bases.get(offset) or bases.get(offset.name)


def offset_expr(state: CodegenState, offset: object) -> str:
    """Index-expression text for any ``TileIndexWithOffsetPattern`` offset."""
    name = base_name(state, offset)
    if name is not None:
        return name
    if isinstance(offset, torch.fx.Node):
        # A dynamic base with no binding in this scope: ``literal_expr`` would
        # fall through to ``repr()`` and emit an undefined name.
        raise NotImplementedError(
            "Pallas: rebased tile base "
            f"{offset.name!r} is not bound in this scope; it cannot be lowered "
            "as a contiguous tile offset here."
        )
    return state.device_function.literal_expr(offset)


def store_base(state: CodegenState, subscript: object) -> str | None:
    """Dynamic scalar base of a ``base + tile.index`` subscript, if it has one."""
    if not isinstance(subscript, torch.fx.Node):
        return None
    metadata = subscript.meta.get("tile_with_offset")
    if not isinstance(metadata, dict):
        return None
    cached = subscript.meta.get("pallas_rebased_tile_base")
    if isinstance(cached, str):
        return cached
    return base_name(state, metadata.get("offset"))


# --- device-IR reconciliation ------------------------------------------------
#
# ``add_tile_with_offset_metadata`` decides one node at a time, but whether a
# rebase is *sound* is a property of the whole row tile: every access on it must
# move by the same base.  Two bases resolve to two different windows, and the
# lowering can only pick one -- so the rest would silently read or write the
# wrong rows.  This reconciliation drops the metadata in that case, before
# ``plan_tiling`` builds any pattern from it.


class _CallSite(NamedTuple):
    """Where a sub-graph is entered from, and the values bound to its
    placeholders.  Loop bodies and ``if``/``else`` branches both qualify, so a
    base can be traced outward through either."""

    parent_graph_id: int
    args: tuple[object, ...]


def _call_sites(graphs: list[GraphInfo]) -> dict[int, _CallSite]:
    """Map sub-graph id -> the single site that enters it.

    Tracing a base outward assumes one site per graph, so a graph entered from
    more than one place is dropped rather than guessed at.
    """
    from ...language import _tracing_ops

    sites: dict[int, _CallSite] = {}
    ambiguous: set[int] = set()

    def record(graph_id: object, parent: int, args: object) -> None:
        if not isinstance(graph_id, int) or not isinstance(args, (list, tuple)):
            return
        if graph_id in sites:
            ambiguous.add(graph_id)
        sites[graph_id] = _CallSite(parent, tuple(args))

    loop_targets = (_tracing_ops._for_loop, _tracing_ops._for_loop_step)
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            if node.target in loop_targets:
                # _for_loop(graph_id, begin, end, args[, step])
                record(node.args[0], graph_info.graph_id, node.args[3])
            elif node.target is _tracing_ops._if:
                # _if(test, if_graph_id, else_graph_id, if_args, else_args)
                record(node.args[1], graph_info.graph_id, node.args[3])
                record(node.args[2], graph_info.graph_id, node.args[4])
    for graph_id in ambiguous:
        del sites[graph_id]
    return sites


def _canonical_base(node: object, graph_id: int, sites: dict[int, _CallSite]) -> object:
    """Trace a base out to its outermost definition.

    The same value is a distinct node in every scope it is copied into, so
    bases are only comparable once resolved this far out.  Returns the node
    itself when it cannot be traced further, which keeps unresolvable bases
    distinct and therefore conservative.
    """
    seen: set[int] = set()
    while True:
        node = unwrap_new_var(node)
        if not isinstance(node, torch.fx.Node) or node.op != "placeholder":
            return node
        if id(node) in seen:
            return node
        seen.add(id(node))
        site = sites.get(graph_id)
        if site is None:
            return node
        placeholders = [n for n in node.graph.nodes if n.op == "placeholder"]
        index = placeholders.index(node)
        if index >= len(site.args):
            return node
        arg = site.args[index]
        if not isinstance(arg, torch.fx.Node):
            return node
        node, graph_id = arg, site.parent_graph_id


def drop_conflicting_bases(graphs: list[GraphInfo]) -> None:
    """Un-mark rebased tiles whose row tile is accessed at more than one base."""
    sites = _call_sites(graphs)
    bases: dict[int, set[object]] = {}
    marked: list[tuple[torch.fx.Node, int]] = []
    for graph_info in graphs:
        for node in graph_info.graph.nodes:
            metadata = node.meta.get("tile_with_offset")
            if not isinstance(metadata, dict):
                continue
            offset = metadata.get("offset")
            if not isinstance(offset, torch.fx.Node):
                continue
            block_id = metadata.get("block_id")
            if not isinstance(block_id, int):
                continue
            base = _canonical_base(offset, graph_info.graph_id, sites)
            bases.setdefault(block_id, set()).add(id(base))
            marked.append((node, block_id))
    for node, block_id in marked:
        if len(bases[block_id]) > 1:
            del node.meta["tile_with_offset"]
