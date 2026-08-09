#!/usr/bin/env python3
"""Add DSV4 support to ROCm/Megatron-LM (rocm_dev branch).

Adds the minimal TransformerConfig fields and TransformerBlock/Layer hooks
needed by Lumen's ``get_dsv4_spec()`` to construct the DSV4 model.  The actual
DSV4 attention/compressor/indexer/HC layers come from the Lumen library — this
patch only makes Megatron's config and block infrastructure accept them.

Changes:
  1. TransformerConfig: add 'dsv4' to experimental_attention_variant Literal,
     add dsv4_* fields (hc_mult, compress_ratios, window_size, etc.),
     add dsv4_mode post_init flag.
  2. TransformerBlock: add HCHeadParams on last PP rank when dsv4_mode.
  3. TransformerLayer: add per-layer HC params (hc_attn_fn/base/scale,
     hc_ffn_fn/base/scale) when dsv4_mode.

Usage (inside container):
    python3 examples/GRPO/dsv4/patch_rocm_megatron_dsv4.py /workspace/Megatron-LM
"""

from __future__ import annotations

import ast
import os
import re
import sys


CHECKPOINT_REPLAY_CAPABILITY = (
    "LUMENRL_R3_CAPABILITY_CHECKPOINT_REPLAY_BACKWARD"
)
ROUTER_REPLAY_FIFO_CAPABILITY = "LUMENRL_R3_CAPABILITY_ROUTER_REPLAY_FIFO"
REPLAY_DIAGNOSTICS_CAPABILITY = (
    "LUMENRL_R3_CAPABILITY_REPLAY_DIAGNOSTICS"
)


def _stamp_capabilities(content: str, markers: tuple[str, ...]) -> str:
    """Add import-visible capability flags without duplicating existing flags."""
    missing = []
    for marker in markers:
        true_assignment = re.compile(
            rf"(?m)^{re.escape(marker)}\s*=\s*True\s*$"
        )
        if true_assignment.search(content):
            continue
        existing_assignment = re.compile(
            rf"(?m)^{re.escape(marker)}\s*=.*$"
        )
        if existing_assignment.search(content):
            content = existing_assignment.sub(f"{marker} = True", content, count=1)
        else:
            missing.append(marker)
    if not missing:
        return content
    separator = "" if content.endswith("\n") else "\n"
    assignments = "\n".join(f"{marker} = True" for marker in missing)
    return f"{content}{separator}\n{assignments}\n"


def patch_file(path: str, replacements: list[tuple[str, str]]) -> bool:
    with open(path) as f:
        content = f.read()
    original = content
    for old, new in replacements:
        content = content.replace(old, new)
    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def _top_level_source_extent(content: str, start: int) -> tuple[int, int]:
    """Return the extent of one top-level def/class starting at ``start``."""
    line_end = content.find("\n", start)
    search_start = len(content) if line_end < 0 else line_end + 1
    boundary = re.search(
        r"(?m)^(?:def|class)\s+[A-Za-z_]\w*",
        content[search_start:],
    )
    end = len(content) if boundary is None else search_start + boundary.start()
    return start, end


def _find_router_definition(
    megatron_root: str,
) -> tuple[str, str, int, int] | None:
    """Find the preferred file that actually defines the routing function."""
    moe_dir = os.path.join(
        megatron_root, "megatron", "core", "transformer", "moe"
    )
    for filename in ("moe_utils.py", "router.py"):
        candidate = os.path.join(moe_dir, filename)
        if not os.path.isfile(candidate):
            continue
        with open(candidate) as f:
            content = f.read()
        definition = re.search(
            r"(?m)^def topk_routing_with_score_function\(",
            content,
        )
        if definition is not None:
            start, end = _top_level_source_extent(content, definition.start())
            return candidate, content, start, end
    return None


def _literal_tokens(values: str) -> set[str]:
    return set(re.findall(r"""["']([^"']+)["']""", values))


def _config_has_exact_sqrtsoftplus(content: str) -> bool:
    score_literal = re.search(
        r"moe_router_score_function:\s*Literal\[(?P<values>[^\]]+)\]",
        content,
    )
    return (
        score_literal is not None
        and "sqrtsoftplus" in _literal_tokens(score_literal.group("values"))
    )


def patch_transformer_config(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_config.py")

    with open(path) as f:
        content = f.read()
    original = content

    # 1. Add 'dsv4' to the Literal type
    content = content.replace(
        "Literal['gated_delta_net', 'dsa']",
        "Literal['gated_delta_net', 'dsa', 'dsv4']",
    )

    # 2. Admit the DSV4 checkpoint's score function without changing defaults.
    score_literal = re.compile(
        r"(moe_router_score_function:\s*Literal\[)(?P<values>[^\]]+)(\])"
    )

    def add_sqrtsoftplus(match: re.Match[str]) -> str:
        values = match.group("values")
        if "sqrtsoftplus" in _literal_tokens(values):
            return match.group(0)
        quote = '"' if '"' in values else "'"
        return (
            f"{match.group(1)}{values}, {quote}sqrtsoftplus{quote}"
            f"{match.group(3)}"
        )

    content = score_literal.sub(add_sqrtsoftplus, content, count=1)

    # 3. DSV4 uses its checkpoint expert bias with sqrtsoftplus. Megatron's
    # upstream validation only admits expert bias with sigmoid.
    expert_bias_guard = re.compile(
        r"(self\.moe_router_enable_expert_bias\s+and\s+)"
        r"self\.moe_router_score_function\s*!=\s*[\"']sigmoid[\"']"
    )
    content = expert_bias_guard.sub(
        r'\1self.moe_router_score_function not in ("sigmoid", "sqrtsoftplus")',
        content,
        count=1,
    )

    # 4. Add DSV4 fields after the DSA section
    # Find the right place — after the last DSA field
    dsa_marker = "    ####################\n    # DSA\n    ####################"
    if dsa_marker in content and "dsv4_mode: bool = False" not in content:
        # Find end of DSA fields section (next ## or blank-line-then-field)
        # Insert DSV4 fields after all dsa_ fields
        # Find last dsa_ field
        last_dsa_match = None
        for m in re.finditer(r'    dsa_\w+:.*\n(?:    """.*?"""\n)?', content):
            last_dsa_match = m
        if last_dsa_match:
            insert_pos = last_dsa_match.end()
            dsv4_fields = '''
    ####################
    # DSV4
    ####################
    dsv4_mode: bool = False
    dsv4_hc_mult: Optional[int] = None
    dsv4_hc_sinkhorn_iters: int = 20
    dsv4_hc_eps: float = 1e-6
    dsv4_compress_ratios: Optional[list] = None
    dsv4_compress_rope_theta: float = 160000.0
    dsv4_o_groups: Optional[int] = None
    dsv4_o_lora_rank: Optional[int] = None
    dsv4_n_hash_layers: int = 0
    dsv4_window_size: int = 128

'''
            content = content[:insert_pos] + dsv4_fields + content[insert_pos:]

    # 5. Add dsv4_mode = True in __post_init__ when variant == "dsv4"
    dsa_post_init = '        if self.experimental_attention_variant == "dsa":'
    if (
        dsa_post_init in content
        and 'if self.experimental_attention_variant == "dsv4":' not in content
    ):
        dsv4_post_init = '''        if self.experimental_attention_variant == "dsv4":
            self.dsv4_mode = True

'''
        # Insert before the dsa check
        content = content.replace(dsa_post_init, dsv4_post_init + dsa_post_init)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_moe_router_score_function(megatron_root: str) -> bool:
    """Add sqrt-softplus routing while keeping expert bias selection-only."""
    definition = _find_router_definition(megatron_root)
    if definition is None:
        return False
    path, content, function_start, function_end = definition
    function_source = content[function_start:function_end]
    if 'elif score_function in ("sigmoid", "sqrtsoftplus"):' in function_source:
        return False

    branch = re.search(
        r'(?m)^(?P<indent>[ \t]*)elif score_function == ["\']sigmoid["\']:\s*$',
        function_source,
    )
    if branch is None:
        return False
    branch_start = function_start + branch.start()
    branch_body_start = function_start + branch.end()
    indent = branch.group("indent")
    error_else = re.search(
        rf"(?m)^{re.escape(indent)}else:\s*\n"
        rf"{re.escape(indent)}    raise ValueError\(",
        content[branch_body_start:function_end],
    )
    if error_else is None:
        return False
    branch_end = branch_body_start + error_else.start()
    replacement = (
        f'{indent}elif score_function in ("sigmoid", "sqrtsoftplus"):\n'
        f'{indent}    if score_function == "sigmoid":\n'
        f"{indent}        scores = torch.sigmoid(logits.float()).type_as(logits)\n"
        f"{indent}    else:\n"
        f"{indent}        scores = torch.nn.functional.softplus("
        "logits.float()).sqrt().type_as(logits)\n"
        f"{indent}    if expert_bias is not None:\n"
        f"{indent}        scores_for_routing = scores + expert_bias\n"
        f"{indent}        _, top_indices = compute_topk("
        "scores_for_routing, topk, num_groups, group_topk)\n"
        f"{indent}        scores = torch.gather("
        "scores, dim=1, index=top_indices).type_as(logits)\n"
        f"{indent}    else:\n"
        f"{indent}        scores, top_indices = compute_topk("
        "scores, topk, num_groups, group_topk)\n"
        f"{indent}    probs = scores / "
        "scores.sum(dim=-1, keepdim=True).clamp(min=1e-20) "
        "if topk > 1 else scores\n"
    )
    content = content[:branch_start] + replacement + content[branch_end:]
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_checkpoint_router_replay(megatron_root: str) -> bool:
    """Scope backward routing replay to activation-checkpoint recomputation."""
    path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "random.py"
    )
    with open(path) as f:
        content = f.read()
    original = content
    class_definition = re.search(r"(?m)^class CheckpointFunction\(", content)
    if class_definition is None:
        return False
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    marker = "router_replay_actions = ["
    if marker not in content[class_start:class_end]:
        anchor = re.search(
            r"(?m)^(?P<indent>[ \t]*)with torch\.enable_grad\(\):\s*\n"
            r"(?P=indent)    outputs = ctx\.run_function\(\*detached_inputs\)\s*$",
            content[class_start:class_end],
        )
        if anchor is None:
            return False
        anchor_start = class_start + anchor.start()
        anchor_end = class_start + anchor.end()
        indent = anchor.group("indent")
        replacement = (
            f"{indent}from megatron.core.transformer.moe.router_replay import (\n"
            f"{indent}    RouterReplay,\n"
            f"{indent}    RouterReplayAction,\n"
            f"{indent})\n"
            f"{indent}router_replays = list(RouterReplay.global_router_replay_instances)\n"
            f"{indent}router_replay_actions = [\n"
            f"{indent}    instance.router_replay_action for instance in router_replays\n"
            f"{indent}]\n"
            f"{indent}try:\n"
            f"{indent}    for instance in router_replays:\n"
            f"{indent}        instance.set_router_replay_action(\n"
            f"{indent}            RouterReplayAction.REPLAY_BACKWARD\n"
            f"{indent}        )\n"
            f"{indent}    with torch.enable_grad():\n"
            f"{indent}        outputs = ctx.run_function(*detached_inputs)\n"
            f"{indent}finally:\n"
            f"{indent}    for instance, prior_action in zip(\n"
            f"{indent}        router_replays, router_replay_actions\n"
            f"{indent}    ):\n"
            f"{indent}        if prior_action is None:\n"
            f"{indent}            instance.clear_router_replay_action()\n"
            f"{indent}        else:\n"
            f"{indent}            instance.set_router_replay_action(prior_action)"
        )
        content = content[:anchor_start] + replacement + content[anchor_end:]

    class_definition = re.search(r"(?m)^class CheckpointFunction\(", content)
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    checkpoint_source = content[class_start:class_end]
    if not (
        "router_replay_actions = [" in checkpoint_source
        and "RouterReplayAction.REPLAY_BACKWARD" in checkpoint_source
        and "finally:" in checkpoint_source
        and "instance.set_router_replay_action(prior_action)" in checkpoint_source
    ):
        raise RuntimeError(
            "CheckpointFunction replay patch failed structural postconditions; "
            "refusing to stamp its runtime capability"
        )
    content = _stamp_capabilities(
        content,
        (CHECKPOINT_REPLAY_CAPABILITY,),
    )
    if content == original:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def _router_replay_diagnostics_state(content: str) -> str:
    """Return ``absent``, ``complete``, or ``partial`` for the diagnostics."""
    diagnostic_names = {
        "recompute_forward_indices",
        "recompute_compared_ids",
        "recompute_flips",
        "reset_recompute_diagnostics",
        "get_recompute_diagnostics",
    }
    has_diagnostic_marker = any(name in content for name in diagnostic_names)
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "partial" if has_diagnostic_marker else "absent"

    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RouterReplay"
    ]
    if len(classes) != 1:
        return "partial" if has_diagnostic_marker else "absent"
    methods = {
        node.name: node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required_methods = {
        "__init__",
        "reset_recompute_diagnostics",
        "get_recompute_diagnostics",
        "get_replay_topk",
    }
    if not required_methods <= methods.keys():
        return "partial" if has_diagnostic_marker else "absent"

    def self_attr(node, name: str) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == name
        )

    def assigned_attrs(method) -> dict[str, ast.AST]:
        assigned = {}
        for node in ast.walk(method):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        assigned[target.attr] = node.value
        return assigned

    init_assignments = assigned_attrs(methods["__init__"])
    init_ok = (
        isinstance(
            init_assignments.get("recompute_forward_indices"),
            ast.List,
        )
        and not init_assignments["recompute_forward_indices"].elts
        and all(
            isinstance(init_assignments.get(name), ast.Constant)
            and init_assignments[name].value == 0
            for name in ("recompute_compared_ids", "recompute_flips")
        )
    )

    reset = methods["reset_recompute_diagnostics"]
    reset_assignments = assigned_attrs(reset)
    reset_clears = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear"
        and self_attr(node.func.value, "recompute_forward_indices")
        for node in ast.walk(reset)
    )
    reset_ok = reset_clears and all(
        isinstance(reset_assignments.get(name), ast.Constant)
        and reset_assignments[name].value == 0
        for name in ("recompute_compared_ids", "recompute_flips")
    )

    getter = methods["get_recompute_diagnostics"]
    getter_ok = any(
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and self_attr(node.value.elts[0], "recompute_compared_ids")
        and self_attr(node.value.elts[1], "recompute_flips")
        for node in ast.walk(getter)
    )

    replay = methods["get_replay_topk"]
    append_calls = [
        node
        for node in ast.walk(replay)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and self_attr(node.func.value, "recompute_forward_indices")
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "top_indices"
    ]
    pop_assignments = [
        node
        for node in ast.walk(replay)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "forward_indices"
            for target in node.targets
        )
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "pop"
        and self_attr(node.value.func.value, "recompute_forward_indices")
        and len(node.value.args) == 1
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value == 0
    ]
    missing_forward_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and self_attr(node.test.operand, "recompute_forward_indices")
        and any(isinstance(child, ast.Raise) for child in ast.walk(node))
        for node in ast.walk(replay)
    )
    augmented = {
        node.target.attr
        for node in ast.walk(replay)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.op, ast.Add)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
    }
    replay_ok = (
        len(append_calls) == 1
        and len(pop_assignments) == 1
        and missing_forward_guard
        and {"recompute_compared_ids", "recompute_flips"} <= augmented
    )
    return "complete" if init_ok and reset_ok and getter_ok and replay_ok else (
        "partial" if has_diagnostic_marker else "absent"
    )


def _router_replay_has_fifo_semantics(content: str) -> bool:
    """Verify RouterReplay records and consumes backward routes in FIFO order."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False
    replay_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RouterReplay"
        ),
        None,
    )
    if replay_class is None:
        return False
    methods = {
        node.name: node
        for node in replay_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    record = methods.get("set_target_indices")
    replay = methods.get("get_replay_topk")
    if record is None or replay is None:
        return False

    def replay_list(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and node.attr == "replay_backward_list"
        )

    appends = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and replay_list(node.func.value)
        for node in ast.walk(record)
    )
    pops_front = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "pop"
        and replay_list(node.func.value)
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
        for node in ast.walk(replay)
    )
    return appends and pops_front


def patch_router_replay_diagnostics(megatron_root: str) -> bool:
    """Track exact forward/recompute replay agreement without tensor clones."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "moe",
        "router_replay.py",
    )
    with open(path) as f:
        content = f.read()
    original = content
    diagnostic_state = _router_replay_diagnostics_state(content)
    if diagnostic_state == "complete":
        if not _router_replay_has_fifo_semantics(content):
            raise RuntimeError(
                "RouterReplay lacks FIFO record/replay semantics; "
                "refusing to stamp runtime capabilities"
            )
        content = _stamp_capabilities(
            content,
            (
                ROUTER_REPLAY_FIFO_CAPABILITY,
                REPLAY_DIAGNOSTICS_CAPABILITY,
            ),
        )
        if content == original:
            return False
        with open(path, "w") as f:
            f.write(content)
        return True
    if diagnostic_state == "partial":
        raise RuntimeError(
            "RouterReplay has a partial or malformed diagnostic patch; "
            "refusing to modify it"
        )

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    if class_definition is None:
        return False
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    class_source = content[class_start:class_end]

    init_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)RouterReplay"
        r"\.global_router_replay_instances\.append\(self\)\s*$",
        class_source,
    )
    methods_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)def set_target_indices"
        r"\(self,\s*topk_indices(?::[^)]*)?\):\s*$",
        class_source,
    )
    forward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = self\.target_topk_idx\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    backward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = "
        r"self\.replay_backward_list\.pop\(0\)\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    if (
        init_anchor is None
        or methods_anchor is None
        or forward_anchor is None
        or backward_anchor is None
    ):
        return False

    init_indent = init_anchor.group("indent")
    init_insert = (
        f"{init_indent}self.recompute_forward_indices: List[torch.Tensor] = []\n"
        f"{init_indent}self.recompute_compared_ids = 0\n"
        f"{init_indent}self.recompute_flips = 0\n"
    )
    init_pos = class_start + init_anchor.start()
    content = content[:init_pos] + init_insert + content[init_pos:]

    # Recompute offsets after the first insertion.
    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    class_source = content[class_start:class_end]
    methods_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)def set_target_indices"
        r"\(self,\s*topk_indices(?::[^)]*)?\):\s*$",
        class_source,
    )
    method_indent = methods_anchor.group("indent")
    body_indent = method_indent + "    "
    diagnostics_methods = (
        f"{method_indent}def reset_recompute_diagnostics(self):\n"
        f'{body_indent}"""Reset counters and references without touching backward replay."""\n'
        f"{body_indent}self.recompute_forward_indices.clear()\n"
        f"{body_indent}self.recompute_compared_ids = 0\n"
        f"{body_indent}self.recompute_flips = 0\n\n"
        f"{method_indent}def get_recompute_diagnostics(self):\n"
        f'{body_indent}"""Return ``(compared expert ids, changed expert ids)``."""\n'
        f"{body_indent}return self.recompute_compared_ids, self.recompute_flips\n\n"
    )
    methods_pos = class_start + methods_anchor.start()
    content = content[:methods_pos] + diagnostics_methods + content[methods_pos:]

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    class_source = content[class_start:class_end]
    forward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = self\.target_topk_idx\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    forward_pos = class_start + forward_anchor.end()
    forward_indent = forward_anchor.group("indent")
    content = (
        content[:forward_pos]
        + f"\n{forward_indent}self.recompute_forward_indices.append(top_indices)"
        + content[forward_pos:]
    )

    class_definition = re.search(r"(?m)^class RouterReplay(?:\([^)]*\))?:", content)
    class_start, class_end = _top_level_source_extent(
        content, class_definition.start()
    )
    class_source = content[class_start:class_end]
    backward_anchor = re.search(
        r"(?m)^(?P<indent>[ \t]*)top_indices = "
        r"self\.replay_backward_list\.pop\(0\)\s*\n"
        r"(?:(?P=indent)#[^\n]*\n)?"
        r"(?P=indent)top_indices = top_indices\.to\(scores\.device\)\s*$",
        class_source,
    )
    backward_pos = class_start + backward_anchor.end()
    backward_indent = backward_anchor.group("indent")
    nested_indent = backward_indent + "    "
    compare = (
        f"\n{backward_indent}if not self.recompute_forward_indices:\n"
        f"{nested_indent}raise RuntimeError(\n"
        f'{nested_indent}    "RouterReplay recompute has no matching forward indices"\n'
        f"{nested_indent})\n"
        f"{backward_indent}forward_indices = self.recompute_forward_indices.pop(0)\n"
        f"{backward_indent}forward_flat = forward_indices.reshape(-1)\n"
        f"{backward_indent}backward_flat = top_indices.reshape(-1)\n"
        f"{backward_indent}overlap = min(forward_flat.numel(), backward_flat.numel())\n"
        f"{backward_indent}self.recompute_compared_ids += max(\n"
        f"{nested_indent}forward_flat.numel(), backward_flat.numel()\n"
        f"{backward_indent})\n"
        f"{backward_indent}self.recompute_flips += abs(\n"
        f"{nested_indent}forward_flat.numel() - backward_flat.numel()\n"
        f"{backward_indent})\n"
        f"{backward_indent}if overlap:\n"
        f"{nested_indent}self.recompute_flips += int(\n"
        f"{nested_indent}    (forward_flat[:overlap] != backward_flat[:overlap])\n"
        f"{nested_indent}    .sum().item()\n"
        f"{nested_indent})"
    )
    content = content[:backward_pos] + compare + content[backward_pos:]
    if (
        _router_replay_diagnostics_state(content) != "complete"
        or not _router_replay_has_fifo_semantics(content)
    ):
        raise RuntimeError(
            "RouterReplay diagnostic/FIFO patch failed structural postconditions; "
            "refusing to stamp runtime capabilities"
        )
    content = _stamp_capabilities(
        content,
        (
            ROUTER_REPLAY_FIFO_CAPABILITY,
            REPLAY_DIAGNOSTICS_CAPABILITY,
        ),
    )
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_transformer_block(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_block.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add HC head params creation after _build_layers() call
    build_layers_call = "        self._build_layers()"
    if build_layers_call in content and "HCHeadParams" not in content:
        hc_block = '''
        # DSV4 Hyper-Connection head params (last PP rank only)
        if getattr(self.config, 'dsv4_mode', False):
            from lumen.models.dsv4.ops.hyper_connection import HCHeadParams
            from megatron.core import parallel_state as mpu
            if mpu.is_pipeline_last_stage():
                self.hc_head_params = HCHeadParams(self.config)
'''
        content = content.replace(
            build_layers_call,
            build_layers_call + hc_block,
        )

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_transformer_layer(megatron_root: str) -> bool:
    path = os.path.join(megatron_root, "megatron", "core", "transformer", "transformer_layer.py")

    with open(path) as f:
        content = f.read()
    original = content

    # Add per-layer HC params after self.mlp assignment in __init__
    # Find a stable anchor point — after the mlp is built
    if "dsv4_mode" not in content and "self.mlp = build_module" in content:
        # Find the class __init__ and add HC params
        # Look for the end of __init__ where self.mlp is assigned
        anchor = "        self.bias_dropout_add_exec_handler = torch.enable_grad"
        if anchor in content:
            hc_layer = '''
        # DSV4 Hyper-Connection per-layer params
        if getattr(self.config, 'dsv4_mode', False):
            import torch.nn as nn
            hc_mult = self.config.dsv4_hc_mult or 4
            hc_dim = hc_mult * self.config.hidden_size
            mix_size = (2 + hc_mult) * hc_mult
            self.hc_attn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_attn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            self.hc_ffn_fn = nn.Parameter(torch.zeros(mix_size, hc_dim, dtype=torch.float32))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix_size, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.zeros(3, dtype=torch.float32))
            for p in [self.hc_attn_fn, self.hc_attn_base, self.hc_attn_scale,
                       self.hc_ffn_fn, self.hc_ffn_base, self.hc_ffn_scale]:
                p._keep_fp32 = True

'''
            content = content.replace(anchor, hc_layer + anchor)

    # Lumen's DSV4 attention returns a tensor, while Megatron BDA expects
    # the standard (output, bias) pair.
    attention_anchor = '        nvtx_range_pop(suffix="self_attention")'
    attention_compat = '''

        if isinstance(attention_output_with_bias, torch.Tensor):
            attention_output_with_bias = (attention_output_with_bias, None)
'''
    if (
        attention_anchor in content
        and "isinstance(attention_output_with_bias, torch.Tensor)" not in content
    ):
        content = content.replace(
            attention_anchor, attention_anchor + attention_compat, 1
        )

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_eav_specs(megatron_root: str) -> bool:
    """Add dsv4 branch to get_experimental_attention_variant_module_spec."""
    path = os.path.join(megatron_root, "megatron", "core", "models", "gpt",
                        "experimental_attention_variant_module_specs.py")
    with open(path) as f:
        content = f.read()
    original = content

    # Replace the else branch to handle dsv4 before raising.
    # Lumen's get_dsv4_spec monkey-patches this at runtime, but the Literal
    # type needs to accept 'dsv4' without erroring.
    old_else = '''    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    new_else = '''    elif config.experimental_attention_variant == "dsv4":
        # DSV4 spec is injected by Lumen's get_dsv4_spec() monkey-patch at runtime
        raise ValueError(
            "DSV4 attention variant requires Lumen's get_dsv4_spec() — "
            "call it before get_experimental_attention_variant_module_spec()"
        )
    else:
        raise ValueError(
            f"Invalid experimental attention variant: {config.experimental_attention_variant}"
        )'''
    if old_else in content and "dsv4" not in content:
        content = content.replace(old_else, new_else)

    if content != original:
        with open(path, "w") as f:
            f.write(content)
        return True
    return False


def patch_tp_layers(megatron_root: str) -> bool:
    """Add condition_init_method to tensor_parallel/layers.py (needed by Lumen linears)."""
    path = os.path.join(megatron_root, "megatron", "core", "tensor_parallel", "layers.py")
    with open(path) as f:
        content = f.read()
    if "def condition_init_method" in content:
        return False
    stub = '''

def condition_init_method(config, init_method):
    """Condition weight initialization on config (Lumen compatibility shim).

    Returns the init_method unchanged — Lumen's LumenColumnParallelLinear calls
    this during CPU initialization. Xavier-uniform override is not used for DSV4.
    """
    if getattr(config, "init_method_xavier_uniform", False):
        import torch.nn.init as init
        return init.xavier_uniform_
    return init_method

'''
    content += stub
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_distrib_optimizer_fp32_detach(megatron_root: str) -> bool:
    """Fix missing .detach() on FP32 param shard views in DistributedOptimizer.

    The BF16 path uses model_param.detach().view(-1)[...] but the FP32 path
    uses model_param.view(-1)[...] without .detach(). When FP32 params have
    requires_grad=True (e.g. DSV4 HC params, attn_sink, compressor APE), the
    view creates a non-leaf tensor, causing 'can't optimize a non-leaf Tensor'
    when HybridDeviceOptimizer (CPU offload) validates param groups.
    """
    path = os.path.join(megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py")
    with open(path) as f:
        content = f.read()
    old = "shard_model_param = model_param.view(-1)["
    if old not in content:
        return False
    content = content.replace(old, "shard_model_param = model_param.detach().view(-1)[")
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_distrib_optimizer_grad_copy(megatron_root: str) -> bool:
    """Route fully offloaded grads to HybridDeviceOptimizer without an FP32 GPU copy."""
    path = os.path.join(megatron_root, "megatron", "core", "optimizer", "distrib_optimizer.py")
    with open(path) as f:
        content = f.read()
    marker = "shard_main_param in self.optimizer.gpu_params_map_cpu_copy"
    if marker in content:
        return False
    old = next(
        (
            candidate
            for candidate in (
                "shard_main_param.grad = shard_model_grad.float()",
                "shard_main_param.grad = shard_model_grad.to(shard_main_param)",
            )
            if candidate in content
        ),
        None,
    )
    if old is None:
        return False
    new = (
        "if (\n"
        "                            isinstance(self.optimizer, HybridDeviceOptimizer)\n"
        "                            and shard_main_param in self.optimizer.gpu_params_map_cpu_copy\n"
        "                        ):\n"
        "                            shard_main_param.decoupled_grad = shard_model_grad\n"
        "                        else:\n"
        "                            shard_main_param.grad = shard_model_grad.float()"
    )
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_tp_copy_fp32_gradient_reduce(megatron_root: str) -> bool:
    """Support Lumen's FP32 tensor-parallel gradient reduction option."""
    path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "mappings.py"
    )
    with open(path) as f:
        content = f.read()
    old_signature = "def copy_to_tensor_model_parallel_region(input_, group=None):"
    if old_signature not in content:
        return False

    content = content.replace(
        old_signature,
        "def copy_to_tensor_model_parallel_region("
        "input_, group=None, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "return _CopyToModelParallelRegion.apply(input_, group)",
        "return _CopyToModelParallelRegion.apply("
        "input_, group, all_reduce_grad_fp32)",
        1,
    )
    content = content.replace(
        "def symbolic(graph, input_, group):",
        "def symbolic(graph, input_, group, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "def forward(ctx, input_, group):",
        "def forward(ctx, input_, group, all_reduce_grad_fp32=False):",
        1,
    )
    content = content.replace(
        "        ctx.group = group\n        return input_",
        "        ctx.group = group\n"
        "        ctx.all_reduce_grad_fp32 = all_reduce_grad_fp32\n"
        "        return input_",
        1,
    )
    content = content.replace(
        "        return _reduce(grad_output, ctx.group), None",
        "        if ctx.all_reduce_grad_fp32:\n"
        "            grad_input = _reduce(grad_output.float(), ctx.group).to(grad_output.dtype)\n"
        "        else:\n"
        "            grad_input = _reduce(grad_output, ctx.group)\n"
        "        return grad_input, None, None",
        1,
    )
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_hybrid_optimizer_disable_foreach(megatron_root: str) -> bool:
    """Avoid full-gradient foreach temporaries in torch CPU/GPU optimizers."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "optimizer",
        "cpu_offloading",
        "hybrid_optimizer.py",
    )
    with open(path) as f:
        content = f.read()
    replacements = {
        "self.cpu_optimizer_cls(self.cpu_param_groups)":
            "self.cpu_optimizer_cls(self.cpu_param_groups, foreach=False)",
        "self.gpu_optimizer_cls(self.gpu_param_groups)":
            "self.gpu_optimizer_cls(self.gpu_param_groups, foreach=False)",
        "cpu_optimizer_cls([_cpu_param_group])":
            "cpu_optimizer_cls([_cpu_param_group], foreach=False)",
    }
    original = content
    for old, new in replacements.items():
        content = content.replace(old, new)
    if content == original:
        return False
    with open(path, "w") as f:
        f.write(content)
    return True


def patch_hybrid_optimizer_streaming_sgd(megatron_root: str) -> bool:
    """Stream fully offloaded zero-momentum SGD grads through one CPU buffer."""
    path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "optimizer",
        "cpu_offloading",
        "hybrid_optimizer.py",
    )
    with open(path) as f:
        content = f.read()
    marker = "def _can_stream_full_offload_sgd(self):"
    if marker in content:
        return False
    step_signature = "    def step(self, closure=None):\n"
    if step_signature not in content:
        return False
    methods = '''    def _can_stream_full_offload_sgd(self):
        """Return whether the bounded-memory SGD fast path is semantically safe."""
        if self.offload_fraction != 1.0 or self.gpu_optimizer is not None:
            return False
        if not self.cpu_optimizers:
            return False
        for optimizer in self.cpu_optimizers:
            if not isinstance(optimizer, torch.optim.SGD):
                return False
            for group in optimizer.param_groups:
                if float(group.get("momentum", 0.0)) != 0.0:
                    return False
                if float(group.get("dampening", 0.0)) != 0.0:
                    return False
                if bool(group.get("nesterov", False)) or bool(group.get("maximize", False)):
                    return False
        return True

    def _stream_full_offload_sgd_step(self, closure=None):
        """Update CPU masters with one reusable FP32 gradient staging tensor."""
        if closure is not None:
            raise RuntimeError("streaming full-offload SGD does not support closures")
        self._sync_hdo_param_groups_to_sub_optimizers()
        cpu_params = [
            cpu_param
            for optimizer in self.cpu_optimizers
            for group in optimizer.param_groups
            for cpu_param in group["params"]
        ]
        max_numel = max((cpu_param.numel() for cpu_param in cpu_params), default=0)
        if max_numel == 0:
            self._sync_sub_optimizers_state_to_hdo()
            return None
        staging = torch.empty(
            (2, max_numel),
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )
        if not getattr(self, "_streaming_sgd_announced", False):
            print(
                "[LumenRL] enabled double-buffered streaming CPU SGD; "
                f"staging={staging.numel() * staging.element_size() / 2**20:.1f} MiB",
                flush=True,
            )
            self._streaming_sgd_announced = True
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        pending = [None, None]

        def finish_slot(slot):
            item = pending[slot]
            if item is None:
                return
            event, cpu_param, orig_param, staging_view, lr, weight_decay = item
            event.synchronize()
            if weight_decay:
                cpu_param.mul_(1.0 - lr * weight_decay)
            cpu_param.add_(staging_view, alpha=-lr)
            with torch.cuda.stream(self._h2d_stream):
                orig_param.copy_(cpu_param, non_blocking=True)
            pending[slot] = None

        with torch.no_grad():
            sequence = 0
            for optimizer in self.cpu_optimizers:
                for group in optimizer.param_groups:
                    lr = float(group["lr"])
                    weight_decay = float(group.get("weight_decay", 0.0))
                    for cpu_param in group["params"]:
                        orig_param = self.cpu_copys_map_gpu_param[cpu_param]
                        grad = getattr(orig_param, "decoupled_grad", orig_param.grad)
                        if grad is None:
                            continue
                        if grad.is_sparse:
                            raise RuntimeError(
                                "streaming full-offload SGD requires dense gradients"
                            )
                        slot = sequence % 2
                        finish_slot(slot)
                        staging_view = staging[slot, : cpu_param.numel()].view_as(cpu_param)
                        with torch.cuda.stream(self._d2h_stream):
                            staging_view.copy_(grad, non_blocking=True)
                            d2h_event = self._d2h_stream.record_event()
                        pending[slot] = (
                            d2h_event,
                            cpu_param,
                            orig_param,
                            staging_view,
                            lr,
                            weight_decay,
                        )
                        sequence += 1
            for index in range(max(0, sequence - 2), sequence):
                finish_slot(index % 2)
        self._h2d_stream.synchronize()
        self._sync_sub_optimizers_state_to_hdo()
        return None

'''
    content = content.replace(step_signature, methods + step_signature, 1)
    step_preamble = (
        "        if self._can_stream_full_offload_sgd():\n"
        "            return self._stream_full_offload_sgd_step(closure)\n\n"
    )
    content = content.replace(
        step_signature,
        step_signature + step_preamble,
        1,
    )
    with open(path, "w") as f:
        f.write(content)
    return True


def _validate_required_patches(megatron_root: str) -> str:
    """Validate required DSV4 patches and return the router source path."""
    problems = []
    config_path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "transformer_config.py",
    )
    try:
        with open(config_path) as f:
            config_content = f.read()
    except OSError:
        config_content = ""
    if not _config_has_exact_sqrtsoftplus(config_content):
        problems.append(
            "transformer_config.py moe_router_score_function lacks exact "
            "'sqrtsoftplus' Literal token"
        )

    router_definition = _find_router_definition(megatron_root)
    router_path = ""
    if router_definition is None:
        problems.append("topk_routing_with_score_function definition not found")
    else:
        router_path, router_content, router_start, router_end = router_definition
        router_source = router_content[router_start:router_end]
        required_router_markers = (
            'elif score_function in ("sigmoid", "sqrtsoftplus"):',
            "torch.nn.functional.softplus(logits.float()).sqrt().type_as(logits)",
            "scores_for_routing = scores + expert_bias",
            "scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)",
        )
        if not all(marker in router_source for marker in required_router_markers):
            problems.append(
                "topk_routing_with_score_function has missing or incompatible "
                "sqrtsoftplus patch anchors"
            )

    random_path = os.path.join(
        megatron_root, "megatron", "core", "tensor_parallel", "random.py"
    )
    try:
        with open(random_path) as f:
            random_content = f.read()
    except OSError:
        random_content = ""
    checkpoint_definition = re.search(
        r"(?m)^class CheckpointFunction\(", random_content
    )
    checkpoint_source = ""
    if checkpoint_definition is not None:
        checkpoint_start, checkpoint_end = _top_level_source_extent(
            random_content, checkpoint_definition.start()
        )
        checkpoint_source = random_content[checkpoint_start:checkpoint_end]
    if not (
        "router_replay_actions = [" in checkpoint_source
        and "RouterReplayAction.REPLAY_BACKWARD" in checkpoint_source
        and "finally:" in checkpoint_source
    ):
        problems.append(
            "CheckpointFunction.backward lacks checkpoint router replay marker"
        )
    if f"{CHECKPOINT_REPLAY_CAPABILITY} = True" not in random_content:
        problems.append(
            f"tensor_parallel.random lacks {CHECKPOINT_REPLAY_CAPABILITY}"
        )

    replay_path = os.path.join(
        megatron_root,
        "megatron",
        "core",
        "transformer",
        "moe",
        "router_replay.py",
    )
    try:
        with open(replay_path) as f:
            replay_content = f.read()
    except OSError:
        replay_content = ""
    if _router_replay_diagnostics_state(replay_content) != "complete":
        problems.append(
            "RouterReplay lacks structurally complete forward/recompute "
            "ID diagnostics"
        )
    if not _router_replay_has_fifo_semantics(replay_content):
        problems.append("RouterReplay lacks FIFO record/replay semantics")
    for capability in (
        ROUTER_REPLAY_FIFO_CAPABILITY,
        REPLAY_DIAGNOSTICS_CAPABILITY,
    ):
        if f"{capability} = True" not in replay_content:
            problems.append(f"router_replay.py lacks {capability}")

    if problems:
        raise RuntimeError(
            "Required ROCm Megatron DSV4 patch validation failed: "
            + "; ".join(problems)
        )
    return router_path


def main(megatron_root: str) -> None:
    try:
        replay_path = os.path.join(
            megatron_root,
            "megatron",
            "core",
            "transformer",
            "moe",
            "router_replay.py",
        )
        with open(replay_path) as replay_file:
            replay_content = replay_file.read()
        if _router_replay_diagnostics_state(replay_content) == "partial":
            raise RuntimeError(
                "RouterReplay has a partial or malformed diagnostic patch; "
                "refusing all writes"
            )
        transformer_config_changed = patch_transformer_config(megatron_root)
        router_changed = patch_moe_router_score_function(megatron_root)
        checkpoint_changed = patch_checkpoint_router_replay(megatron_root)
        replay_diagnostics_changed = patch_router_replay_diagnostics(
            megatron_root
        )
    except OSError as exc:
        raise RuntimeError(
            f"Required ROCm Megatron patch inputs are unavailable: {exc}"
        ) from exc
    router_path = _validate_required_patches(megatron_root)
    router_label = os.path.relpath(
        router_path,
        os.path.join(megatron_root, "megatron", "core"),
    ).replace(os.sep, "/")
    results = {
        "transformer_config.py": transformer_config_changed,
        router_label: router_changed,
        "tensor_parallel/random.py": checkpoint_changed,
        "transformer/moe/router_replay.py diagnostics": replay_diagnostics_changed,
        "transformer_block.py": patch_transformer_block(megatron_root),
        "transformer_layer.py": patch_transformer_layer(megatron_root),
        "experimental_attention_variant_module_specs.py": patch_eav_specs(megatron_root),
        "tensor_parallel/layers.py": patch_tp_layers(megatron_root),
        "tensor_parallel/mappings.py": patch_tp_copy_fp32_gradient_reduce(megatron_root),
        "optimizer/hybrid_optimizer.py": patch_hybrid_optimizer_disable_foreach(megatron_root),
        "optimizer/hybrid_optimizer.py streaming SGD": patch_hybrid_optimizer_streaming_sgd(
            megatron_root
        ),
        "optimizer/distrib_optimizer.py": patch_distrib_optimizer_fp32_detach(megatron_root),
        "optimizer/distrib_optimizer.py grad copy": patch_distrib_optimizer_grad_copy(megatron_root),
    }
    print(f"Patched ROCm Megatron at {megatron_root}:")
    for name, ok in results.items():
        print(f"  {'PATCHED' if ok else 'skipped'}: {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <megatron-root>")
        sys.exit(1)
    main(sys.argv[1])
