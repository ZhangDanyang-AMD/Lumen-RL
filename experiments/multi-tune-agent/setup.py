from setuptools import setup


GEAK_UTILS_EXAMPLES = [
    (
        "share/geak-utils/examples/tasks/gemm",
        [
            "examples/tasks/gemm/config.yaml",
            "examples/tasks/gemm/kernel.py",
        ],
    ),
    (
        "share/geak-utils/examples/tasks/gemm/scripts",
        ["examples/tasks/gemm/scripts/task_runner.py"],
    ),
    (
        "share/geak-utils/examples/tasks/gemm_fp8",
        [
            "examples/tasks/gemm_fp8/config.yaml",
            "examples/tasks/gemm_fp8/kernel.py",
        ],
    ),
    (
        "share/geak-utils/examples/tasks/gemm_fp8/scripts",
        ["examples/tasks/gemm_fp8/scripts/task_runner.py"],
    ),
    (
        "share/geak-utils/examples/tasks/gemm_mxfp4",
        [
            "examples/tasks/gemm_mxfp4/config.yaml",
            "examples/tasks/gemm_mxfp4/kernel.py",
            "examples/tasks/gemm_mxfp4/metadata.json",
        ],
    ),
    (
        "share/geak-utils/examples/tasks/gemm_mxfp4/scripts",
        ["examples/tasks/gemm_mxfp4/scripts/task_runner.py"],
    ),
    (
        "share/geak-utils/examples/tasks/fused_attention",
        [
            "examples/tasks/fused_attention/config.yaml",
            "examples/tasks/fused_attention/kernel.py",
        ],
    ),
    (
        "share/geak-utils/examples/tasks/fused_attention/scripts",
        ["examples/tasks/fused_attention/scripts/task_runner.py"],
    ),
    (
        "share/geak-utils/examples/tasks/grouped_gemm",
        [
            "examples/tasks/grouped_gemm/config.yaml",
            "examples/tasks/grouped_gemm/kernel.py",
        ],
    ),
    (
        "share/geak-utils/examples/tasks/grouped_gemm/scripts",
        ["examples/tasks/grouped_gemm/scripts/task_runner.py"],
    ),
]


setup(
    name="multi-tune-agent",
    version="0.1.0",
    package_dir={
        "multi_tune_agent": "src/multi_tune_agent",
        "geak_utils": "geak_utils",
    },
    packages=["multi_tune_agent", "geak_utils"],
    data_files=GEAK_UTILS_EXAMPLES,
    install_requires=[
        "PyYAML>=6.0",
        "requests>=2.31",
        "geak @ git+https://github.com/AMD-AGI/GEAK.git",
    ],
    extras_require={"test": ["pytest>=8.0"]},
    entry_points={"console_scripts": ["multi-tune=multi_tune_agent.cli:main"]},
)

