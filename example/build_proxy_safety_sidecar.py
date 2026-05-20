import argparse

from common.config import PrimitiveExecutorConfig, VehicleConfig
from common.runtime_config import ObservationConfig
from primitives import build_default_primitive_library, build_proxy_safety_sidecar, save_proxy_safety_sidecar


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an articulation-aware proxy safety sidecar for HybridPPO.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--lidar-beams", type=int, default=108)
    parser.add_argument("--lidar-range", type=float, default=30.0)
    parser.add_argument("--articulation-bins", type=int, default=7)
    parser.add_argument("--proxy-resolution", type=int, default=2)
    parser.add_argument("--max-proxies-per-action", type=int, default=None)
    parser.add_argument("--max-macro-steps", type=int, default=32)
    return parser


def main() -> str:
    args = build_argument_parser().parse_args()
    observation = ObservationConfig(lidar_num_beams=int(args.lidar_beams), lidar_max_range=float(args.lidar_range))
    vehicle = VehicleConfig()
    executor = PrimitiveExecutorConfig(max_macro_steps=int(args.max_macro_steps))
    library = build_default_primitive_library()
    sidecar = build_proxy_safety_sidecar(
        library=library,
        vehicle_config=vehicle,
        executor_config=executor,
        lidar_num=int(observation.lidar_num_beams),
        lidar_range=float(observation.lidar_max_range),
        articulation_bin_count=int(args.articulation_bins),
        proxy_resolution=int(args.proxy_resolution),
        max_proxies_per_action=None if args.max_proxies_per_action is None else int(args.max_proxies_per_action),
    )
    save_proxy_safety_sidecar(str(args.output), sidecar)
    print(str(args.output))
    return str(args.output)


if __name__ == "__main__":
    main()