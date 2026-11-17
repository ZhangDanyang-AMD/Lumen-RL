"""SWE task datasets in the v5 SWE-bench row format the environment grades, and as RL rows.

SWE-bench/SWE-bench_Verified already is that format. SWE-rebench (nebius/SWE-rebench) ships
SWE-bench-style images and an `install_config` with the test command and log parser, so its
rows convert: the eval script below is the shape of SWE-bench's own (reset the test files,
apply the test patch, run the tests between the markers `swebench` reads the log by).

Its images need repairing, which `swe_rebench_setup` does in every sandbox of the task before
anything else: the repository was installed in editable mode from `/<name>` and then moved to
`/testbed`, so nothing imports until that path points there again; some images do not have it
installed at all, so the row's install command is tried (offline: it fails where it needs the
network, which the symlink then covers); and `/root/.bashrc` activates the conda base env,
which does not have the repository, rather than `testbed`.

Some images cannot run their tests at all (a pytest plugin that does not match the pytest
installed, say), so keep only tasks whose gold patch grades resolved (`evaluate` with the gold
patch, or `scratch/swe_rebench_oracle.py`).
"""

from __future__ import annotations

import json
import re
from typing import Any

from lumenrl.agent.swe.swebench import SweBenchTask

# Some SWE-rebench images have conda in /opt/miniconda3, others only in /opt/conda.
ACTIVATE = ("if [ -f /opt/miniconda3/bin/activate ]; then source /opt/miniconda3/bin/activate; "
            "else source /opt/conda/bin/activate; fi\nconda activate testbed")

EVAL_SCRIPT = """#!/bin/bash
set -uxo pipefail
{activate}
cd /testbed
git config --global --add safe.directory /testbed
cd /testbed
git status
git show
git -c core.fileMode=false diff {base}
git checkout {base} {test_files}
git apply -v - <<'EOF_114329324912'
{test_patch}
EOF_114329324912
: '>>>>> Start Test Output'
{test_cmd} {test_files}
: '>>>>> End Test Output'
git checkout {base} {test_files}
"""


_SWE_REBENCH_SETUP = r"""set -euo pipefail
{activate}
(cd /testbed && {install}) > /tmp/lumenrl_install.log 2>&1 || echo "install failed (offline?), kept the image's"
python - <<'LUMENRL_EOF'
import glob, json, os
pattern = os.path.join(os.environ["CONDA_PREFIX"], "lib/python*/site-packages/*.dist-info/direct_url.json")
for path in glob.glob(pattern):
    with open(path) as f:
        info = json.load(f)
    url = info.get("url", "")
    if info.get("dir_info", {}).get("editable") and url.startswith("file://"):
        source = url[len("file://"):]
        if not os.path.exists(source):
            os.symlink("/testbed", source)
LUMENRL_EOF
grep -qx "conda activate testbed" /root/.bashrc || echo "conda activate testbed" >> /root/.bashrc
"""


def swe_rebench_setup(install: str) -> str:
    return _SWE_REBENCH_SETUP.replace("{activate}", ACTIVATE).replace("{install}", install or "true")


def modified_files(patch: str) -> list[str]:
    """The files a unified diff touches, in order (`swebench.harness.utils.get_modified_files`)."""
    return list(dict.fromkeys(re.findall(r"^diff --git a/(\S+) b/\S+$", patch, flags=re.MULTILINE)))


def _listed(value: Any) -> list[str]:
    if isinstance(value, str):
        return list(json.loads(value))
    return [str(v) for v in value]


def swe_rebench_row(row: dict[str, Any]) -> dict[str, Any]:
    """A SWE-rebench row in the v5 format."""
    cfg = row["install_config"]
    cfg = json.loads(cfg) if isinstance(cfg, str) else dict(cfg)
    test_patch = row["test_patch"]
    files = " ".join(modified_files(test_patch))
    return {
        "instance_id": row["instance_id"],
        "repo": row["repo"],
        "version": str(row.get("version") or ""),
        "base_commit": row["base_commit"],
        "problem_statement": row["problem_statement"],
        "patch": row["patch"],
        "test_patch": test_patch,
        "FAIL_TO_PASS": _listed(row["FAIL_TO_PASS"]),
        "PASS_TO_PASS": _listed(row["PASS_TO_PASS"]),
        "image": row.get("docker_image") or row["image_name"],
        "log_parser": cfg.get("log_parser") or "parse_log_pytest",
        "eval_type": "pass_and_fail",
        "eval_script": EVAL_SCRIPT.format(activate=ACTIVATE, base=row["base_commit"], test_files=files,
                                          test_patch=test_patch.rstrip("\n"), test_cmd=cfg["test_cmd"]),
        "setup_script": swe_rebench_setup(cfg.get("install") or ""),
    }


TASK_FIELDS = ("instance_id", "repo", "version", "base_commit", "problem_statement", "patch", "test_patch",
               "FAIL_TO_PASS", "PASS_TO_PASS", "image", "log_parser", "eval_type", "eval_script", "image_assets",
               "setup_script")


def training_row(task_row: dict[str, Any], data_source: str = "swe") -> dict[str, Any]:
    """The trainer's row for a task: its mini-SWE prompt, and the task itself as the ground
    truth the SWE environment rolls out and grades (`multi_turn.swe`). Only the fields grading
    reads are kept."""
    row = {k: task_row[k] for k in TASK_FIELDS if k in task_row and task_row[k] is not None}
    for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        row[k] = _listed(row[k])
    return {
        "data_source": data_source,
        "prompt": SweBenchTask(row).messages(),
        "reward_model": {"style": "environment", "ground_truth": json.dumps(row)},
        "extra_info": {"instance_id": row["instance_id"], "repo": row.get("repo", "")},
    }
