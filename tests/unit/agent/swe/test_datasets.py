import json

import numpy as np

from lumenrl.agent.swe import mini_swe
from lumenrl.agent.swe.datasets import (
    TASK_FIELDS,
    modified_files,
    swe_rebench_row,
    training_row,
)

TEST_PATCH = (
    "diff --git a/tests/test_a.py b/tests/test_a.py\n--- a/tests/test_a.py\n+++ b/tests/test_a.py\n@@ -1 +1 @@\n-x\n+y\n"
    "diff --git a/tests/new_test.py b/tests/new_test.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/new_test.py\n"
    "@@ -0,0 +1 @@\n+z\n"
    "diff --git a/tests/test_a.py b/tests/test_a.py\n"
)
REBENCH = {
    "instance_id": "org__proj-7", "repo": "org/proj", "version": "1.0", "base_commit": "abc123",
    "problem_statement": "It breaks.", "patch": "diff --git a/src/p.py b/src/p.py\n", "test_patch": TEST_PATCH,
    "FAIL_TO_PASS": np.array(["tests/test_a.py::test_x"]), "PASS_TO_PASS": np.array([], dtype=object),
    "docker_image": "swerebench/sweb.eval.x86_64.org_1776_proj-7", "image_name": "ignored",
    "install_config": {"install": "pip install -e .[test]", "log_parser": "parse_log_pytest",
                       "test_cmd": "pytest --no-header -rA"},
    "created_at": "2024-01-01", "meta": {"x": np.array([1])},
}


def test_modified_files_are_each_listed_once_in_order():
    assert modified_files(TEST_PATCH) == ["tests/test_a.py", "tests/new_test.py"]


def test_a_swe_rebench_row_becomes_a_v5_row():
    row = swe_rebench_row(REBENCH)

    assert row["image"] == "swerebench/sweb.eval.x86_64.org_1776_proj-7"
    assert (row["log_parser"], row["eval_type"]) == ("parse_log_pytest", "pass_and_fail")
    assert row["FAIL_TO_PASS"] == ["tests/test_a.py::test_x"] and row["PASS_TO_PASS"] == []
    script = row["eval_script"]
    assert "git checkout abc123 tests/test_a.py tests/new_test.py\n" in script
    assert ("git apply -v - <<'EOF_114329324912'\n" + TEST_PATCH.rstrip("\n") + "\nEOF_114329324912\n") in script
    assert ("\n: '>>>>> Start Test Output'\npytest --no-header -rA tests/test_a.py tests/new_test.py\n"
            ": '>>>>> End Test Output'\n") in script
    assert "(cd /testbed && pip install -e .[test])" in row["setup_script"]
    assert 'os.symlink("/testbed", source)' in row["setup_script"]
    for s in (script, row["setup_script"]):
        assert "else source /opt/conda/bin/activate; fi\nconda activate testbed\n" in s, "either conda location"


def test_a_training_row_carries_the_task_as_json_ground_truth():
    task = swe_rebench_row(REBENCH)
    task["created_at"] = "dropped"

    row = training_row(task)

    gt = json.loads(row["reward_model"]["ground_truth"])
    assert set(gt) <= set(TASK_FIELDS) and "created_at" not in gt
    assert gt["eval_script"] == task["eval_script"] and gt["FAIL_TO_PASS"] == ["tests/test_a.py::test_x"]
    assert row["prompt"][0]["content"] == mini_swe.SYSTEM_TEMPLATE.rstrip("\n")
    assert "It breaks." in row["prompt"][1]["content"]
    assert row["extra_info"] == {"instance_id": "org__proj-7", "repo": "org/proj"}
