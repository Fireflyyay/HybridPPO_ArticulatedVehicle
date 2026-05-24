import argparse
from dataclasses import replace
from typing import Optional, Sequence

from common.runtime_config import ExperimentConfig
from training.train_loop import ExperimentTrainer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Hybrid PPO on the articulated vehicle task.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--episodes-per-update", type=int, default=None)
    parser.add_argument("--max-macro-steps", type=int, default=None)
    parser.add_argument("--max-low-level-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--warmup-min-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-root", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--lidar-beams", type=int, default=None)
    parser.add_argument("--proxy-safety-sidecar", type=str, default=None)
    parser.add_argument("--enable-soft-mask", action="store_true")
    parser.add_argument("--soft-mask-gamma", type=float, default=None)
    parser.add_argument("--soft-mask-logit-scale", type=float, default=None)
    parser.add_argument("--soft-mask-floor", type=float, default=None)
    parser.add_argument("--safety-loss-coef", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--disable-teacher", action="store_true", default=False)
    return parser


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig()
    observation = replace(
        config.observation,
        lidar_num_beams=int(args.lidar_beams) if args.lidar_beams is not None else config.observation.lidar_num_beams,
    )
    agent = replace(
        config.agent,
        soft_mask_enabled=bool(args.enable_soft_mask) or config.agent.soft_mask_enabled,
        soft_mask_gamma=float(args.soft_mask_gamma) if args.soft_mask_gamma is not None else config.agent.soft_mask_gamma,
        soft_mask_logit_scale=(
            float(args.soft_mask_logit_scale)
            if args.soft_mask_logit_scale is not None
            else config.agent.soft_mask_logit_scale
        ),
        soft_mask_floor=(
            float(args.soft_mask_floor)
            if args.soft_mask_floor is not None
            else config.agent.soft_mask_floor
        ),
        safety_loss_coef=float(args.safety_loss_coef) if args.safety_loss_coef is not None else config.agent.safety_loss_coef,
    )
    proxy_safety = replace(
        config.proxy_safety,
        sidecar_path=str(args.proxy_safety_sidecar) if args.proxy_safety_sidecar is not None else config.proxy_safety.sidecar_path,
    )
    env = replace(
        config.env,
        max_low_level_steps_per_episode=(
            int(args.max_low_level_steps)
            if args.max_low_level_steps is not None
            else config.env.max_low_level_steps_per_episode
        ),
    )
    schedule = replace(
        config.schedule,
        total_episodes=int(args.episodes) if args.episodes is not None else config.schedule.total_episodes,
        episodes_per_update=(
            int(args.episodes_per_update)
            if args.episodes_per_update is not None
            else config.schedule.episodes_per_update
        ),
        max_macro_steps_per_episode=(
            int(args.max_macro_steps)
            if args.max_macro_steps is not None
            else config.schedule.max_macro_steps_per_episode
        ),
        warmup_min_episodes=(
            int(args.warmup_min_episodes)
            if args.warmup_min_episodes is not None
            else config.schedule.warmup_min_episodes
        ),
    )
    evaluation = replace(
        config.evaluation,
        interval=int(args.eval_interval) if args.eval_interval is not None else config.evaluation.interval,
        episodes_per_level=(
            int(args.eval_episodes)
            if args.eval_episodes is not None
            else config.evaluation.episodes_per_level
        ),
    )
    logging = replace(
        config.logging,
        log_root=str(args.log_root) if args.log_root is not None else config.logging.log_root,
        run_name=str(args.run_name) if args.run_name is not None else config.logging.run_name,
    )
    checkpoint = replace(
        config.checkpoint,
        save_interval=(
            int(args.save_interval)
            if args.save_interval is not None
            else config.checkpoint.save_interval
        ),
    )
    return replace(
        config,
        seed=int(args.seed) if args.seed is not None else config.seed,
        device=str(args.device) if args.device is not None else config.device,
        observation=observation,
        agent=agent,
        proxy_safety=proxy_safety,
        env=env,
        schedule=schedule,
        logging=logging,
        checkpoint=checkpoint,
        evaluation=evaluation,
        teacher_enabled=not args.disable_teacher,
    )


def main(argv: Optional[Sequence[str]] = None) -> str:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    config = build_experiment_config(args)
    trainer = ExperimentTrainer(config, resume_path=args.resume)
    run_dir = trainer.run()
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()