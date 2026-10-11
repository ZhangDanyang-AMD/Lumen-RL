"""Freeze and verify pinned MEnvData-SWE C++ General Coding Replay rows."""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests

from .general_coding_quota_override import (
    _docker_hub_digest,
    _exclusion_dimensions,
    _get_json,
    overlap_reasons,
)
from .general_coding_rebench import (
    _normalized_patch,
    _supplemental_exclusions,
    _write_frozen_jsonl,
)
from .general_coding_replay import (
    PERMISSIVE_SOURCE_LICENSES,
    _docker_image_identity,
    _patch_paths,
    atomic_write,
    canonical_json,
    derive_repository_task_type,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    verify_candidates,
    write_json,
    write_jsonl,
)


DATASET_ID = "ernie-research/MEnvData-SWE"
DATASET_REVISION = "edfaa7bf15ada849c3bd63f55a5e3ab9e85359c2"
SOURCE_FILE = "swe-images.jsonl"
SOURCE_FILE_SHA256 = "e111fa1a8c4565f427d928652fddde7ce36a9d32973bb554beb60fba2c6055aa"
SOURCE_ROWS = 3005
CPP_SOURCE_ROWS = 31
SOURCE_ID = "menvdata_swe_cpp"
TARGET_ROWS = 9
FULL_COMMIT = re.compile(r"[0-9a-f]{40}\Z")

# These are repository licenses, not the Apache-2.0 dataset license.  Every
# claim is checked against immutable license content at the row's base commit.
REPOSITORY_LICENSES = {
    "ArthurSonzogni/FTXUI": "MIT",
    "BehaviorTree/BehaviorTree.CPP": "MIT",
    "CLIUtils/CLI11": "BSD-3-Clause",
    "CrowCpp/Crow": "BSD-3-Clause",
    "FastLED/FastLED": "MIT",
    "InsightSoftwareConsortium/ITK": "Apache-2.0",
    "KhronosGroup/SPIRV-Tools": "Apache-2.0",
    "KhronosGroup/glslang": "BSD-3-Clause",
    "NVIDIA/stdexec": "Apache-2.0",
    "OSGeo/PROJ": "MIT",
    "OpenNMT/CTranslate2": "MIT",
    "Tencent/rapidjson": "MIT",
}
REPOSITORY_LICENSE_FILES = {
    "ArthurSonzogni/FTXUI": "LICENSE",
    "BehaviorTree/BehaviorTree.CPP": "LICENSE",
    "CLIUtils/CLI11": "LICENSE",
    "CrowCpp/Crow": "LICENSE",
    "FastLED/FastLED": "LICENSE",
    "InsightSoftwareConsortium/ITK": "LICENSE",
    "KhronosGroup/SPIRV-Tools": "LICENSE",
    "KhronosGroup/glslang": "LICENSE.txt",
    "NVIDIA/stdexec": "LICENSE",
    "OSGeo/PROJ": "COPYING",
    "OpenNMT/CTranslate2": "LICENSE",
    "Tencent/rapidjson": "license.txt",
}


def _image_tag(row: Mapping[str, Any]) -> str:
    value = str(row.get("image_name", "")).removeprefix("docker.io/")
    if not value or ":" not in value:
        raise ValueError("tagged source image missing")
    return value if "/" in value else f"mcatwj/{value}"


def adapt_menvdata_cpp_row(
    row: Mapping[str, Any], *, source_sha256: str = SOURCE_FILE_SHA256
) -> dict[str, Any]:
    """Adapt one C++ row while retaining all immutable replay evidence."""
    if row.get("language") != "C++":
        raise ValueError("row is not C++")
    repository = str(row.get("repo", ""))
    license_spdx = REPOSITORY_LICENSES.get(repository)
    if license_spdx not in PERMISSIVE_SOURCE_LICENSES:
        raise ValueError("upstream repository lacks permissive license allowlist evidence")
    instance_id = str(row.get("instance_id", ""))
    commit = str(row.get("base_commit", ""))
    patch = str(row.get("patch", ""))
    test_patch = str(row.get("test_patch", ""))
    env_script = str(row.get("env_setup_script", ""))
    eval_script = str(row.get("eval_script", ""))
    if not instance_id or "/" not in repository:
        raise ValueError("immutable row identity missing")
    if not FULL_COMMIT.fullmatch(commit):
        raise ValueError("exact base commit missing")
    if not patch.strip() or not test_patch.strip():
        raise ValueError("solution or test patch missing")
    if not env_script.strip() or not eval_script.strip():
        raise ValueError("environment or evaluation script missing")
    image_name = _image_tag(row)
    task_type, task_evidence = derive_repository_task_type(
        {
            "title": str(row.get("problem_statement", "")).splitlines()[0],
            "body": str(row.get("problem_statement", "")),
            "fix_patch": patch,
            "test_patch": test_patch,
        }
    )
    test_patch_hash = sha256_bytes(test_patch.encode())
    return {
        "schema_version": "general_coding_replay_candidate_source_v3",
        "case_id": f"gc-replay-{SOURCE_ID}-{instance_id}",
        "source_id": SOURCE_ID,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_row_id": instance_id,
        "source_file": SOURCE_FILE,
        "source_file_sha256": source_sha256,
        "source_record_sha256": sha256_bytes(canonical_json(row).encode()),
        "source_lineage_id": f"{DATASET_ID}@{DATASET_REVISION}:{instance_id}",
        "upstream_repository": f"https://github.com/{repository}.git",
        "repository": repository,
        "base_commit": commit,
        "problem_id": instance_id,
        "problem_statement": str(row.get("problem_statement", "")),
        "target_patch": patch,
        "target_patch_sha256": sha256_bytes(patch.encode()),
        "normalized_patch_hash": sha256_bytes(_normalized_patch(patch).encode()),
        "test_patch": test_patch,
        "test_patch_sha256": test_patch_hash,
        "test_set_hash": sha256_bytes(
            canonical_json({"test_patch_sha256": test_patch_hash}).encode()
        ),
        "env_setup_script": env_script,
        "env_setup_script_sha256": sha256_bytes(env_script.encode()),
        "eval_script": eval_script,
        "eval_script_sha256": sha256_bytes(eval_script.encode()),
        "image_name": image_name,
        "primary_language": "cpp",
        "primary_task_type": task_type,
        "task_type_evidence": task_evidence,
        "dataset_license_spdx": "Apache-2.0",
        "claimed_upstream_license_spdx": license_spdx,
        "network_policy": "disabled",
        "gpu_required": False,
        "verified": False,
        "local_replay_passes": 0,
    }


REVIEWED_EXECUTIONS: dict[str, dict[str, Any]] = {
    "CLIUtils__CLI11-421": {
        "eval_script_sha256": "3c81e2fee394fa4ab794857109308299eb44d70a94a7dd851fc1553e1e573f84",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo"
                " && cmake --build build --target AppTest -j8"
            ),
            "targeted_test_command": (
                "./build/tests/AppTest 'TApp: stringLikeTests'"
            ),
            "full_regression_command": "true",
        },
    },
    "CLIUtils__CLI11-926": {
        "eval_script_sha256": "0767b245854a7362772e3d1b33a8b5fb041eedbbd96e734a82c5010b8de718f9",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target HelpTest SubcommandTest -j$(nproc)"
            ),
            "targeted_test_command": (
                "./build/tests/HelpTest && ./build/tests/SubcommandTest"
            ),
            "full_regression_command": "true",
        },
    },
    "CLIUtils__CLI11-1203": {
        "eval_script_sha256": "d62b2cf919419fbb359e692a94008c099b150c484d3978b70282e4699904b6a5",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DCMAKE_CXX_STANDARD=17 -DCLI11_BUILD_TESTS=ON"
                " -DCLI11_BUILD_EXAMPLES=OFF -DCLI11_ENABLE_EXTRA_VALIDATORS=1"
                " && cmake --build build --target ExtraValidatorsTest -j$(nproc)"
            ),
            "targeted_test_command": (
                "./build/tests/ExtraValidatorsTest"
                " 'FileExistsForRead,FileExistsForWrite,FileExistsForExec,noPermissionCheck'"
            ),
            "full_regression_command": "true",
        },
    },
    "CLIUtils__CLI11-370": {
        "eval_script_sha256": "871097f4b3c10f84170febc3d12d9cb3aeb8f441915a7c3443b9daba74aa0e93",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DCMAKE_CXX_STANDARD=11 -DCLI11_BUILD_TESTS=ON"
                " -DCLI11_BUILD_EXAMPLES=OFF"
                " && cmake --build build --target TransformTest"
            ),
            "targeted_test_command": (
                "./build/tests/TransformTest"
                " 'TApp: EnumCheckedDefaultTransformCallback'"
            ),
            "full_regression_command": "true",
        },
    },
    "CLIUtils__CLI11-1199": {
        "eval_script_sha256": "a832cb868c52f3de421272695e997003d2f310c3cecf1278412a1193b8a16bbc",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DCMAKE_CXX_STANDARD=11 -DCLI11_BUILD_TESTS=ON"
                " -DCLI11_BUILD_EXAMPLES=OFF"
                " && cmake --build build --target ConfigFileTest HelpersTest -j$(nproc)"
            ),
            "targeted_test_command": (
                "./build/tests/ConfigFileTest 'TApp: CrashTest'"
                " && ./build/tests/HelpersTest"
                " 'Types: LexicalConversionEmptyVectorDouble'"
            ),
            "full_regression_command": "true",
        },
    },
    "CLIUtils__CLI11-1058": {
        "eval_script_sha256": "1851d7c2e5a79191886a313543f9148803ac0427435603d29168f1744ffa9074",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target HelpTest -j$(nproc)"
            ),
            "targeted_test_command": (
                "./build/tests/HelpTest 'THelp: multiple_group'"
            ),
            "full_regression_command": "true",
        },
    },
    "ArthurSonzogni__FTXUI-121": {
        "eval_script_sha256": "3a44928b7b8035f8a7cd31c5c83d0233bef9889186e85bd1ccf3b9d704a06c45",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": "cmake --build build --target tests -j1",
            "targeted_test_command": (
                './build/tests --gtest_filter="TextTest.CombiningCharacters"'
            ),
            "full_regression_command": "true",
        },
    },
    "ArthurSonzogni__FTXUI-260": {
        "eval_script_sha256": "c8c00b366e9ae132133b1a8b9b9d66bdb59d6d88e59de0081656003813a2821c",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": "cmake --build build --target tests -j1",
            "targeted_test_command": (
                './build/tests --gtest_filter="GridboxTest.MissingCells"'
            ),
            "full_regression_command": "true",
        },
    },
    "ArthurSonzogni__FTXUI-298": {
        "eval_script_sha256": "3e847b9b518b93e0152035396f0b616e215ae0da22810acf5a2993f1a1e7fa00",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": "cmake --build build --target tests -j1",
            "targeted_test_command": (
                './build/tests --gtest_filter="*RemoveEntries*"'
            ),
            "full_regression_command": "true",
        },
    },
    "ArthurSonzogni__FTXUI-755": {
        "eval_script_sha256": "c5959b73c29a4229d41d688a2ff235fbc6b87de7121b399eb1c11240b48cb2fd",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target ftxui-tests -j1"
            ),
            "targeted_test_command": (
                "ctest --test-dir build -V -R 'ScrollIndicator.*Colorable'"
                " --output-on-failure"
            ),
            "full_regression_command": "true",
        },
    },
    "CrowCpp__Crow-897": {
        "eval_script_sha256": "1169aef08019f580a3c699f6cddb3e601ff18266a9955e649a1f6a49a7636846",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DCROW_BUILD_TESTS=ON -DCROW_BUILD_EXAMPLES=OFF"
                " -DCROW_FEATURES='ssl;compression' -DCROW_AMALGAMATE=ON"
                " && cmake --build build --target unittest"
            ),
            "targeted_test_command": "./build/tests/unittest task_timer",
            "full_regression_command": "true",
        },
    },
    "CrowCpp__Crow-918": {
        "eval_script_sha256": "5b0efbcdb3028e4b15a07bac9bcd6429e639071501cfb4ffa7cf9ab540699b53",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DCROW_BUILD_TESTS=ON -DCROW_BUILD_EXAMPLES=OFF"
                " -DCROW_FEATURES='ssl;compression' -DCROW_AMALGAMATE=ON"
                " && cmake --build build --target unittest"
            ),
            "targeted_test_command": (
                "./build/tests/unittest"
                " 'server_invalid_ip_address,server_dynamic_port_allocation,middleware_simple'"
            ),
            "full_regression_command": "true",
        },
    },
    "BehaviorTree__BehaviorTree.CPP-424": {
        "eval_script_sha256": "983fc78caca968354b7de1262a6a2fb8f4a25b1eb8ac0b4f588b60da30d2ef13",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target behaviortree_cpp_v3_test --parallel 2"
            ),
            "targeted_test_command": (
                "./build/tests/behaviortree_cpp_v3_test"
                " --gtest_filter='*DecoratorWithoutChildThrows*:*DecoratorWithTwoChildrenThrows*'"
            ),
            "full_regression_command": "true",
        },
    },
    "BehaviorTree__BehaviorTree.CPP-885": {
        "eval_script_sha256": "c764bbb1ca5f8b69bbd9c8fe46522f173b5db853e1eccd433a7d6be60b62b724",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target behaviortree_cpp_test --parallel 2"
            ),
            "targeted_test_command": (
                "./build/tests/behaviortree_cpp_test"
                " --gtest_filter='Reactive.TwoAsyncNodesInReactiveSequence'"
            ),
            "full_regression_command": "true",
        },
    },
    "NVIDIA__stdexec-744": {
        "eval_script_sha256": "34b0138a567e2085d97891d2bce42f155d7ff78b90183191d22ff60c76cd10df",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo"
                " -DSTDEXEC_BUILD_TESTS=ON"
                " && cmake --build build --target test.stdexec --parallel 2"
            ),
            "targeted_test_command": (
                "ctest --test-dir build -V -j8 -R ensure_started"
                " --output-on-failure"
            ),
            "full_regression_command": "true",
        },
    },
    "Tencent__rapidjson-2207": {
        "eval_script_sha256": "62ea273889d93eee484447ec9f52be141d3b4199fbf158b3e75c3c1fa7add82c",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DRAPIDJSON_BUILD_TESTS=ON -DRAPIDJSON_BUILD_EXAMPLES=OFF"
                " -DRAPIDJSON_BUILD_DOC=OFF"
                " && cmake --build build --target unittest"
            ),
            "targeted_test_command": (
                "./build/bin/unittest --gtest_filter='SchemaValidator.Hasher'"
            ),
            "full_regression_command": "true",
        },
    },
    "FastLED__FastLED-1842": {
        "eval_script_sha256": "f6ce34dede9cbf0b2e9e5fe391c17f379049e17a3d2cbe329c4055a714516374",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build tests/.build --target test_apa102_hd"
            ),
            "targeted_test_command": (
                "./tests/.build/bin/test_apa102_hd"
                " 'five_bit_bitshift,__builtin_five_bit_hd_gamma_bitshift'"
            ),
            "full_regression_command": "true",
        },
    },
    "OpenNMT__CTranslate2-898": {
        "eval_script_sha256": "c87775f6209ef455363d33bc679d5fe8a415a97f63ebc7d2fa70a737e7149afa",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target ctranslate2_test -j1"
            ),
            "targeted_test_command": (
                "./build/tests/ctranslate2_test /testbed/tests/data"
                " --gtest_filter='*BiasedDecodingDeviceFPTest*'"
            ),
            "full_regression_command": "true",
        },
    },
    "KhronosGroup__SPIRV-Tools-5025": {
        "eval_script_sha256": "b6692101ac959648e092a6f14043d64e78fbdd5550f3a63f4596005a8cb5e1b4",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release"
                " -DSPIRV_SKIP_TESTS=OFF -DSPIRV_SKIP_EXECUTABLES=OFF"
                " -DSPIRV_BUILD_FUZZER=OFF -DSPIRV_WERROR=ON"
                " && cmake --build build --target test_spirv_unit_tests"
                " --parallel $(nproc)"
            ),
            # The pinned source script exposes this test file through one
            # monolithic gtest/CTest target, so this is the narrowest supported
            # source-derived invocation.
            "targeted_test_command": (
                "ctest --test-dir build -V -j$(nproc)"
                " -R spirv-tools-test_spirv_unit_tests --output-on-failure"
            ),
            "full_regression_command": "true",
        },
    },
    "KhronosGroup__glslang-4005": {
        "eval_script_sha256": "be8476067d7c8153ba91735c60644b43084595f2259d2e9bf8b4da387fd0caf7",
        "commands": {
            "install_command": "true",
            "compile_or_typecheck_command": (
                "cmake --build build --target glslangtests --parallel 1"
            ),
            "targeted_test_command": (
                "./build/gtests/glslangtests"
                " --gtest_filter='*StructNameTest*'"
            ),
            "full_regression_command": "true",
        },
    },
}


def derive_execution_commands(row: Mapping[str, Any]) -> dict[str, str]:
    """Return commands reviewed against an exact pinned eval script."""
    instance_id = str(row["dataset_row_id"])
    reviewed = REVIEWED_EXECUTIONS.get(instance_id)
    if reviewed is None:
        raise ValueError(
            f"{instance_id}: eval script has not received source-specific command review"
        )
    if row.get("eval_script_sha256") != reviewed["eval_script_sha256"]:
        raise ValueError(f"{instance_id}: reviewed eval script hash drifted")
    commands = dict(reviewed["commands"])
    combined = "\n".join(commands.values())
    forbidden = re.compile(
        r"(?i)(?:\b(?:curl|wget|apt(?:-get)?|pip|git|docker)\b|"
        r"/replay/|https?://|--network)"
    )
    if forbidden.search(combined):
        raise ValueError(f"{instance_id}: reviewed commands contain forbidden operation")
    return commands


def _freeze_license(control_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    repository = str(row["repository"])
    commit = str(row["base_commit"])
    claimed = str(row["claimed_upstream_license_spdx"])
    root = control_root / "license-evidence" / "menvdata-swe"
    root.mkdir(parents=True, exist_ok=True)
    stem = repository.replace("/", "__")
    api_path = root / f"{stem}.{commit}.github-license-api.json"
    payload = _get_json(
        f"https://api.github.com/repos/{repository}/license?ref={commit}", api_path
    )
    observed = str((payload.get("license") or {}).get("spdx_id", ""))
    if observed == claimed and observed in PERMISSIVE_SOURCE_LICENSES:
        content = base64.b64decode(str(payload["content"]), validate=False)
        method = "github_license_api_at_commit"
        source_url = str(payload.get("html_url", ""))
    else:
        license_file = REPOSITORY_LICENSE_FILES[repository]
        source_url = (
            f"https://raw.githubusercontent.com/{repository}/{commit}/{license_file}"
        )
        response = requests.get(source_url, timeout=(30, 120))
        response.raise_for_status()
        content = response.content
        text = content.decode("utf-8", errors="replace").lower()
        markers = {
            "MIT": ("permission is hereby granted",),
            "BSD-3-Clause": (
                "redistributions of source code",
                "neither the name",
            ),
            "Apache-2.0": ("apache license", "version 2.0"),
        }[claimed]
        if not all(marker in text for marker in markers):
            raise RuntimeError(
                f"{repository}@{commit}: immutable license text does not prove {claimed}"
            )
        observed = claimed
        method = "validated_immutable_license_file"
    license_path = root / f"{stem}.{commit}.LICENSE"
    if license_path.is_file() and license_path.read_bytes() != content:
        raise RuntimeError(f"{repository}@{commit}: frozen license drift")
    if not license_path.is_file():
        atomic_write(license_path, content)
    return {
        "license_spdx": observed,
        "license_evidence_path": license_path.relative_to(control_root).as_posix(),
        "license_evidence_sha256": sha256_file(license_path),
        "license_api_evidence_path": api_path.relative_to(control_root).as_posix(),
        "license_api_evidence_sha256": sha256_file(api_path),
        "license_evidence_method": method,
        "license_source_url": source_url,
    }


def _materialize_artifacts(control_root: Path, row: dict[str, Any]) -> None:
    root = (
        control_root
        / "source-cache"
        / "menvdata-swe"
        / DATASET_REVISION
        / "candidates"
        / str(row["problem_id"])
    )
    artifacts = {
        "target.patch": str(row.pop("target_patch")).encode(),
        "test.patch": str(row.pop("test_patch")).encode(),
        "problem.md": str(row.pop("problem_statement")).encode(),
        "env_setup.sh": str(row.pop("env_setup_script")).encode(),
        "eval.sh": str(row.pop("eval_script")).encode(),
    }
    for name, content in artifacts.items():
        path = root / name
        if path.is_file() and path.read_bytes() != content:
            raise RuntimeError(f"{row['case_id']}: frozen artifact drift at {name}")
        if not path.is_file():
            atomic_write(path, content)
    row.update(
        {
            "target_patch_path": (root / "target.patch").relative_to(control_root).as_posix(),
            "test_patch_path": (root / "test.patch").relative_to(control_root).as_posix(),
            "problem_statement_path": (root / "problem.md").relative_to(control_root).as_posix(),
            "env_setup_script_path": (root / "env_setup.sh").relative_to(control_root).as_posix(),
            "eval_script_path": (root / "eval.sh").relative_to(control_root).as_posix(),
        }
    )


def freeze_menvdata_cpp(control_root: Path) -> dict[str, Any]:
    cache = control_root / "source-cache" / "menvdata-swe" / DATASET_REVISION
    source = cache / SOURCE_FILE
    if not source.is_file() or sha256_file(source) != SOURCE_FILE_SHA256:
        raise RuntimeError("pinned MEnvData-SWE source file missing or drifted")
    raw_rows = read_jsonl(source)
    if len(raw_rows) != SOURCE_ROWS:
        raise RuntimeError("pinned MEnvData-SWE row count drifted")
    cpp_rows = [row for row in raw_rows if row.get("language") == "C++"]
    if len(cpp_rows) != CPP_SOURCE_ROWS:
        raise RuntimeError("pinned MEnvData-SWE C++ row count drifted")
    registry = json.loads(
        (control_root / "exclusion-registry.json").read_text(encoding="utf-8")
    )
    dimensions = _exclusion_dimensions(
        registry, _supplemental_exclusions(control_root)
    )
    eligible: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for raw in cpp_rows:
        try:
            candidate = adapt_menvdata_cpp_row(raw)
        except ValueError as exc:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "schema_or_license",
                    "detail": str(exc),
                }
            )
            continue
        reasons = overlap_reasons(candidate, dimensions)
        if reasons:
            rejections.append(
                {
                    "instance_id": raw.get("instance_id"),
                    "reason": "overlap",
                    "dimensions": reasons,
                }
            )
            continue
        eligible.append(candidate)
    eligible.sort(
        key=lambda row: (
            sha256_bytes(
                f"{DATASET_REVISION}\0{row['problem_id']}\0"
                f"{row['normalized_patch_hash']}".encode()
            ),
            str(row["case_id"]),
        )
    )
    frozen = []
    image_cache: dict[str, dict[str, str]] = {}
    for candidate in eligible:
        try:
            license_evidence = _freeze_license(control_root, candidate)
            image_name = str(candidate.pop("image_name"))
            if image_name not in image_cache:
                image_cache[image_name] = _docker_hub_digest(image_name)
            candidate.update(license_evidence)
            candidate["container_image"] = image_cache[image_name]
            _materialize_artifacts(control_root, candidate)
            candidate["_manifest_dir"] = str(control_root.resolve())
            frozen.append(candidate)
        except (KeyError, OSError, RuntimeError, requests.RequestException) as exc:
            rejections.append(
                {
                    "instance_id": candidate["problem_id"],
                    "reason": "immutable_license_or_image",
                    "detail": str(exc),
                }
            )
    if len(frozen) < TARGET_ROWS:
        raise RuntimeError(
            f"MEnvData-SWE strict capacity is {len(frozen)}, need at least {TARGET_ROWS}"
        )
    manifest = control_root / "menvdata-swe-cpp-candidate-source-manifest.jsonl"
    _write_frozen_jsonl(manifest, frozen)
    write_jsonl(control_root / "menvdata-swe-cpp-freeze-rejections.jsonl", rejections)
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_cpp_freeze_v1",
        "status": "strict_candidate_pool_ready",
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "source_file_sha256": SOURCE_FILE_SHA256,
        "source_rows": len(raw_rows),
        "cpp_source_rows": len(cpp_rows),
        "strict_capacity": len(frozen),
        "capacity_required_to_close_gap": TARGET_ROWS,
        "selected_by_repository": dict(
            sorted(Counter(str(row["repository"]) for row in frozen).items())
        ),
        "overlap_rejections": sum(r["reason"] == "overlap" for r in rejections),
        "schema_or_license_rejections": sum(
            r["reason"] == "schema_or_license" for r in rejections
        ),
        "immutable_license_or_image_rejections": sum(
            r["reason"] == "immutable_license_or_image" for r in rejections
        ),
        "manifest": {
            "path": manifest.name,
            "rows": len(frozen),
            "sha256": sha256_file(manifest),
        },
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "menvdata-swe-cpp-freeze-report.json", report)
    return report


def prepare_menvdata_cpp(
    control_root: Path, *, case_ids: Sequence[str] = (), limit: int | None = None
) -> dict[str, Any]:
    rows = read_jsonl(control_root / "menvdata-swe-cpp-candidate-source-manifest.jsonl")
    if case_ids:
        requested = set(case_ids)
        rows = [row for row in rows if str(row["case_id"]) in requested]
        missing = sorted(requested - {str(row["case_id"]) for row in rows})
        if missing:
            raise ValueError("unknown MEnvData-SWE case IDs: " + ", ".join(missing))
    if limit is not None:
        rows = rows[:limit]
    prepared, rejected = [], []
    for raw in rows:
        row = dict(raw)
        try:
            commands = derive_execution_commands(row)
            expected = dict(row["container_image"])
            pull = subprocess.run(
                ["docker", "pull", str(expected["repo_tag"])],
                capture_output=True,
                text=True,
                timeout=1800,
                check=False,
            )
            if pull.returncode:
                raise RuntimeError(pull.stderr[-2000:])
            actual = _docker_image_identity(str(expected["repo_tag"]))
            if actual is None or actual["repo_digest"] != expected["repo_digest"]:
                raise RuntimeError("image digest mismatch")
            patch_path = control_root / str(row["target_patch_path"])
            row.update(
                {
                    "container_image": actual,
                    "container_workdir": str(actual["working_dir"] or "/testbed"),
                    "container_reset_command": (
                        f"git reset --hard {row['base_commit']}"
                    ),
                    "verification_backend": "docker",
                    "allowed_paths": _patch_paths(patch_path.read_text(encoding="utf-8")),
                    **commands,
                    "benchmark_exclusion_gate": "ready",
                    "timeouts": {
                        "checkout_dependency": 120,
                        "parent": 1800,
                        "target": 1800,
                        "regression": 600,
                        "fresh_total": 7200,
                    },
                }
            )
            prepared.append(row)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            rejected.append({"case_id": raw["case_id"], "detail": str(exc)[-2000:]})
    manifest = control_root / "menvdata-swe-cpp-executable-manifest.jsonl"
    write_jsonl(manifest, prepared)
    write_jsonl(control_root / "menvdata-swe-cpp-image-rejections.jsonl", rejected)
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_cpp_images_v1",
        "requested": len(rows),
        "prepared": len(prepared),
        "rejected": len(rejected),
        "manifest": {
            "path": manifest.name,
            "rows": len(prepared),
            "sha256": sha256_file(manifest),
        },
    }
    write_json(control_root / "menvdata-swe-cpp-image-report.json", report)
    return report


def verify_menvdata_cpp(
    control_root: Path, *, case_ids: Sequence[str] = (), limit: int | None = None
) -> dict[str, Any]:
    images = prepare_menvdata_cpp(control_root, case_ids=case_ids, limit=limit)
    if images["prepared"] != images["requested"]:
        return {"status": "blocked_image_gate", "image_report": images, "verified_rows": 0}
    verification = verify_candidates(
        control_root / "menvdata-swe-cpp-executable-manifest.jsonl",
        control_root / "menvdata-swe-cpp-verification" / "verified",
        resume=True,
    )
    report = {
        "schema_version": "general_coding_replay_menvdata_swe_cpp_preflight_v1",
        "status": (
            "passed"
            if verification["verified"] == images["prepared"]
            and verification["rejected"] == 0
            else "failed"
        ),
        "scope": "bounded" if case_ids or limit is not None else "full",
        "image_report": images,
        "verification": verification,
        "verified_rows": verification["verified"],
        "required_local_replay_passes": 2,
        "network_policy": "disabled",
        "gpu_policy": "cpu_only",
    }
    write_json(control_root / "menvdata-swe-cpp-preflight-report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    for name in ("prepare", "verify"):
        subparser = commands.add_parser(name)
        subparser.add_argument("--case-id", action="append", default=[])
        subparser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        report = freeze_menvdata_cpp(args.control_root)
    elif args.command == "prepare":
        report = prepare_menvdata_cpp(
            args.control_root, case_ids=args.case_id, limit=args.limit
        )
    else:
        report = verify_menvdata_cpp(
            args.control_root, case_ids=args.case_id, limit=args.limit
        )
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
