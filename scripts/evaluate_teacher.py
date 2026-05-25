import argparse
from dataclasses import asdict, replace
from pprint import pprint

from common.runtime_config import ExperimentConfig, ObservationConfig, TrainingScheduleConfig
from training.teacher_eval import evaluate_soft_teacher


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark the soft teacher without PPO training.")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--lidar-beams", type=int, default=None)
    parser.add_argument("--max-macro-steps", type=int, default=None)
    parser.add_argument("--warmup-progress", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config = ExperimentConfig()
    if args.lidar_beams is not None:
        config = replace(config, observation=replace(config.observation, lidar_num_beams=int(args.lidar_beams)))
    if args.max_macro_steps is not None:
        config = replace(
            config,
            schedule=replace(config.schedule, max_macro_steps_per_episode=int(args.max_macro_steps)),
        )

    cases = [("Warmup", {"warmup_progress": float(progress)}, f"Warmup(progress={float(progress):.2f})") for progress in args.warmup_progress]
    cases.append(("Normal", None, "Normal"))

    for case_idx, (level, reset_options, label) in enumerate(cases):
        summary = evaluate_soft_teacher(
            config=config,
            level=level,
            episodes=int(args.episodes),
            seed_offset=int(args.seed_offset) + case_idx * 1000,
            reset_options=reset_options,
        )
        print(label)
        pprint(asdict(summary), sort_dicts=True)


if __name__ == "__main__":
    main()