import pandas as pd
import hjson
from pathlib import Path
import subprocess
import progressbar
import argparse

base_config = {
    "input_dim": {
        "channels": 16,
        "height": 8,
        "width": 8,
    },
    "eps": 1e-5,
    "tile_ci": 16,
    "prec": 64,
    "impl_opt_level": "MULTICORE_OPT",
    "is_forward": False,
    "is_training": False,
}


def format_size(C, H, W, tile_ci=None):
    return {
        "input_dim": {
            "channels": C,
            "height": H,
            "width": W,
        },
        "tile_ci": C if tile_ci is None else tile_ci,
    }


base_path = Path(__file__).resolve()
config_path = base_path.parent / "data" / "params.hjson"
target_snitch_cluster_path = (
    base_path.parent.parent.parent.parent / "target" / "snitch_cluster"
)
print(target_snitch_cluster_path)

columns = [
    "impl",
    "C",
    "H",
    "W",
    "TILE_CI",
    "num_data_points",
    "kernel_cycles",
    "fpu_occupancy",
    "total_ipc",
]
index = ["impl", "C", "H", "W", "TILE_CI"]

def get_layout_path(is_whole_block):
    return "backwards_mcycle_layout_whole_block.csv" if is_whole_block else "backwards_mcycle_layout_sectioned.csv"

def get_all_configs(config_modifiers):
    return [
        (impl, config)
        for impl, configs in config_modifiers.items()
        for config in configs
    ]


def get_scaling_results_path(whole_block):
    return (
        base_path.parent
        / f"scaling_results{'' if not whole_block else '_whole_block'}.csv"
    )


def read_existing_results(whole_block):
    scaling_results_path = get_scaling_results_path(whole_block)
    if scaling_results_path.exists():
        scaling_results_df = pd.read_csv(scaling_results_path, index_col=index)
    else:
        scaling_results_df = pd.DataFrame(columns=columns)
        scaling_results_df.set_index(index, drop=True, inplace=True)
    return scaling_results_df


def write_results(data, existing_df: pd.DataFrame, is_whole_block=False):
    df = pd.DataFrame(data, columns=columns)
    df.set_index(index, drop=True, inplace=True)
    existing_df = pd.concat([existing_df, df], axis=0)
    existing_df.sort_index(inplace=True)
    existing_df.to_csv(get_scaling_results_path(is_whole_block))


small_sizes = [
    format_size(*dims)
    for dims in [
        (2, 2, 2),
        (4, 2, 2),
        (4, 4, 2),
        (4, 4, 4),
        (8, 4, 4),
        (8, 8, 4),
        (8, 8, 8),
    ]
]
# print(target_snitch_cluster_path.absolute())
config_modifiers = {
    # "SINGLE_CORE": [*small_sizes, format_size(16, 8, 8)],
    "SINGLE_CORE_OPT": [
        *small_sizes,
        format_size(16, 8, 8),
        format_size(16, 16, 8),
        format_size(16, 16, 16),
    ],
    "MULTICORE_OPT": [
        *small_sizes,
        format_size(16, 8, 8),
        format_size(16, 16, 8),
        format_size(16, 16, 16),
        format_size(32, 16, 16),
        format_size(32, 32, 16),
        format_size(32, 32, 32),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(
        "generate_scaling_results.py",
        description="Generates scaling results for various input sizes for batchnorm backward",
    )

    parser.add_argument("--whole-block", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    is_whole_block = args.whole_block
    scaling_results_df = read_existing_results(is_whole_block)
    data = []
    try:
        for impl, config in progressbar.progressbar(get_all_configs(config_modifiers)):
            merged_config = {**base_config, **config, "impl_opt_level": impl}
            C = merged_config["input_dim"]["channels"]
            H = merged_config["input_dim"]["height"]
            W = merged_config["input_dim"]["width"]
            TILE_CI = merged_config["tile_ci"]
            if (impl, C, H, W, TILE_CI) in scaling_results_df.index:
                continue
            total_size = C * H * W
            with open(config_path, "w") as config_file:
                hjson.dump(merged_config, config_file)
            layout_path = get_layout_path(is_whole_block)
            p = subprocess.run(
                f"""make DEBUG=ON sw \
                    && make verify-batchnorm \
                    && make -j traces logs/perf.csv logs/event.csv \
                    && ../../util/trace/layout_events.py logs/event.csv ../../sw/dnn/batchnorm/{layout_path} --cfg cfg/lru.hjson -o logs/trace.csv
                """,
                shell=True,
                cwd=target_snitch_cluster_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            p.check_returncode()
            # what information do I want to get? overall cycles obviously. Maybe utilization of the main loop?
            with open(
                target_snitch_cluster_path / "logs" / "trace.csv"
            ) as trace_results:
                trace_df = pd.read_csv(trace_results)
            # Ignore the tile size calculation - that can be assumed to stay static for a given input size
            cycles = (trace_df["done"] - trace_df["start_main"]).iloc[0]

            # currently section 8 is main loop

            with open(
                target_snitch_cluster_path / "logs" / "perf.csv"
            ) as raw_perf_results:
                raw_perf_df = pd.read_csv(raw_perf_results)

            main_loop_mcycle_section = 8 if not is_whole_block else 2
            main_loop_fpu_occupancy = raw_perf_df.loc[0, f"{main_loop_mcycle_section}_fpss_fpu_occupancy"]
            main_loop_total_ipc = raw_perf_df.loc[0, f"{main_loop_mcycle_section}_total_ipc"]
            data.append(
                (
                    impl,
                    C,
                    H,
                    W,
                    TILE_CI,
                    total_size,
                    cycles,
                    main_loop_fpu_occupancy,
                    main_loop_total_ipc,
                )
            )
            subprocess.run(
                f"cp logs/hart_00000000_perf.json ../../sw/dnn/batchnorm/scaling_raw_results/{impl}_{C}_{H}_{W}_{TILE_CI}_hart_00000000_perf.json",
                shell=True,
                cwd=target_snitch_cluster_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            ).check_returncode()
            subprocess.run(
                f"cp logs/trace_hart_00000000.txt ../../sw/dnn/batchnorm/scaling_raw_results/{impl}_{C}_{H}_{W}_{TILE_CI}_trace_hart_00000000.txt",
                shell=True,
                cwd=target_snitch_cluster_path,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            ).check_returncode()
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        write_results(data, scaling_results_df, args.whole_block)
        exit(130)

    write_results(data, scaling_results_df, args.whole_block)


if __name__ == "__main__":
    main()
