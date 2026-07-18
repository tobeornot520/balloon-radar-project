from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "依次训练H-only、V-only和H+V的"
            "Sigma基线及Dual FCN消融实验"
        )
    )
    parser.add_argument(
        "--experiment-prefix",
        default="hv_ablation",
        help=(
            "实验名称前缀；正式实验默认hv_ablation，"
            "试运行可使用hv_smoke等其他前缀"
        ),
    )

    parser.add_argument(
        "--channels",
        nargs="+",
        choices=["H", "V", "HV"],
        default=["H", "V", "HV"],
    )
    parser.add_argument(
        "--sigma-epochs",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--dual-epochs",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "跳过已经完整生成best.pt和summary.json的阶段；"
            "不执行epoch级断点续训"
        ),
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("\n" + "=" * 82)
    print("执行命令：")
    print(" ".join(command))
    print("=" * 82 + "\n")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def stage_complete(
    experiment_name: str,
) -> bool:
    experiment_dir = (
        PROJECT_ROOT
        / "results"
        / "experiments"
        / experiment_name
    )
    return (
        experiment_dir
        / "checkpoints"
        / "best.pt"
    ).exists() and (
        experiment_dir
        / "tables"
        / "summary.json"
    ).exists()


def main() -> None:
    args = parse_arguments()

    for channel in args.channels:
        sigma_name = (
            f"{args.experiment_prefix}_sigma_{channel}"
        )
        dual_name = (
            f"{args.experiment_prefix}_dual_{channel}"
        )
        sigma_best = (
            PROJECT_ROOT
            / "results"
            / "experiments"
            / sigma_name
            / "checkpoints"
            / "best.pt"
        )

        print("\n" + "#" * 82)
        print(f"开始消融模式：{channel}")
        print("#" * 82)

        if args.resume and stage_complete(
            sigma_name
        ):
            print(
                f"{sigma_name}已完成，跳过。"
            )
        else:
            run([
                sys.executable,
                "scripts/train_sigma_experiment.py",
                "--name", sigma_name,
                "--channel", channel,
                "--range-sigma", "3.0",
                "--velocity-sigma", "1.0",
                "--epochs", str(args.sigma_epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--prediction-count", "0",
            ])

        if not sigma_best.exists():
            raise FileNotFoundError(
                f"找不到{channel}对应Sigma最佳模型："
                f"{sigma_best}"
            )

        if args.resume and stage_complete(
            dual_name
        ):
            print(
                f"{dual_name}已完成，跳过。"
            )
        else:
            run([
                sys.executable,
                "training/train_dual_fcn.py",
                "--pretrained", str(sigma_best),
                "--name", dual_name,
                "--channel", channel,
                "--epochs", str(args.dual_epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
            ])

    complete_all = all(
        stage_complete(
            f"{args.experiment_prefix}_dual_{channel}"
        )
        for channel in ["H", "V", "HV"]
    )

    if (
        not args.skip_analysis
        and complete_all
    ):
        run([
            sys.executable,
            "evaluation/analyze_hv_ablation.py",
            "--prefix",
            f"{args.experiment_prefix}_dual_",
        ])
    elif not args.skip_analysis:
        print(
            "\n尚未完成H、V、HV全部三组实验，"
            "暂不生成总表。"
        )


if __name__ == "__main__":
    main()
