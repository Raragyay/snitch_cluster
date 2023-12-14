import hjson
from pathlib import Path
import subprocess
import argparse
import traceback
import pandas as pd
import time

NAME_FILE = 'tiling_loop.csv'

base_config = {
    "M" : 16,
    "N" : 16,
    "K" : 16,
    "M_tiles" : 2,
    "N_tiles" : 2,
    "K_tiles" : 2,
    "prec" : 64
}

IMPL_TYPES = {
    "MULTICORE_OPT" : "MULTICORE_OPT",
    "MULTICORE" : "MULTICORE",
    "SINGLE_CORE_OPT" : "SINGLE_CORE_OPT",
    "SINGLE_CORE" : "SINGLE_CORE"
}


base_path = Path(__file__).resolve()
config_path = base_path.parent/ "data" / "params.hjson"
target_snitch_cluster_path =  base_path.parent.parent.parent.parent / "target" / "snitch_cluster"
gradient_res_path = target_snitch_cluster_path / "gradient_results.csv"
logs_path = target_snitch_cluster_path / "logs" / "hart_00000000_perf.json"
result_path = base_path.parent / NAME_FILE

columns = [
    "prec",
    "impl",
    "M",
    "N",
    "K",
    "M_tiles",
    "N_tiles",
    "K_tiles",
    "cycles_grad_B",
    "total_ipc_grad_B",
    "fpss_fpu_occupancy_grad_B",
    "cycles_grad_A",
    "total_ipc_grad_A",
    "fpss_fpu_occupancy_grad_A",    
]
index = ["prec", "impl", "M", "N", "K"]

def write_results(data, existing_df):
    df = pd.DataFrame(data, columns=columns,index=[0])
    df.set_index(index, drop=True, inplace=True)
    existing_df = pd.concat([existing_df, df], axis=0)
    existing_df.sort_values(by=["prec", "impl", "M", "N", "K"], inplace=True)
    existing_df.to_csv(result_path)



def read_existing_results():
    if result_path.exists():
        results_df = pd.read_csv(result_path, index_col=index)
    else:
        results_df = pd.DataFrame(columns=columns)
        results_df.set_index(index, drop=True, inplace=True)
    return results_df

def main():
   # tiling = [[2,1,1],[1,2,1],[1,1,2]]
   # iter = 0
    i=0
    while i < 1:
        i += 1
        try:
            print(f"Running for {i}")
            # base_config["M"] = i
            # base_config["N"] = i  
            # base_config["K"] = i
            # base_config["M_tiles"] = tiling[iter][0]
            # base_config["N_tiles"] = tiling[iter][1]
            # base_config["K_tiles"] = tiling[iter][2] 

            with open(config_path, "w") as f:
                hjson.dump(base_config, f, indent=4, ensure_ascii=False)

            time.sleep(3)

            p1 = subprocess.run(
                        f"""
                            rm data/data.h \
                            && make \
                        """,
                        shell=True,
                        cwd=base_path.parent,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
            p1.check_returncode()

            time.sleep(3)

            p1 = subprocess.run(
                        f"""
                            make clean-work\
                            && make DEBUG=on sw \
                        """,
                        shell=True,
                        cwd=target_snitch_cluster_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL            
                    )
            p1.check_returncode()

            time.sleep(3)

            p1 = subprocess.run(
                        f"""
                            bin/snitch_cluster.vlt sw/apps/training/gradient/build/gradient.elf \
                        """,
                        shell=True,
                        cwd=target_snitch_cluster_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=3600            
                    )
            p1.check_returncode()

            time.sleep(3)

            p1 = subprocess.run(
                        f"""
                            make -j traces \
                        """,
                        shell=True,
                        cwd=target_snitch_cluster_path,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            p1.check_returncode()

            with open(logs_path, "r") as f:
                data = hjson.load(f)
            selected_stats = {
                "prec": base_config["prec"],
                "impl": "MULTICORE_OPT",
                "M": base_config["M"],
                "N": base_config["N"],
                "K": base_config["K"],
                "M_tiles": base_config["M_tiles"],
                "N_tiles": base_config["N_tiles"],
                "K_tiles": base_config["K_tiles"],
                "cycles_grad_B": data[1]["cycles"],
                "total_ipc_grad_B": data[1]["total_ipc"],
                "fpss_fpu_occupancy_grad_B": data[1]["fpss_fpu_occupancy"],

                "cycles_grad_A": data[3]["cycles"],
                "total_ipc_grad_A": data[3]["total_ipc"],
                "fpss_fpu_occupancy_grad_A": data[3]["fpss_fpu_occupancy"]
            }

            df = pd.DataFrame(selected_stats,columns=columns,index=[0])
            df.set_index(index, drop=True, inplace=True)
            selected_stats_df = read_existing_results()

            write_results(selected_stats,selected_stats_df)
        except:
            print("Failed for ", i) 
        #iter +=1

        





#                     && ../../util/trace/layout_events.py logs/event.csv ../../sw/dnn/batchnorm/{layout_path} --cfg cfg/lru.hjson -o logs/trace.csv


# def get_layout_path(is_whole_block):
#     return (
#         "backwards_mcycle_layout_whole_block.csv"
#         if is_whole_block
#         else "backwards_mcycle_layout_sectioned.csv"
#     )


# def flatten_config_list(config_modifiers):
#     return [
#         (prec, impl, config)
#         for prec, prec_configs in config_modifiers.items()
#         for impl, configs in prec_configs.items()
#         for config in configs
#     ]


# def get_scaling_results_path(whole_block):
#     return (
#         base_path.parent
#         / f"scaling_results{'' if not whole_block else '_whole_block'}.csv"
#     )


# def read_existing_results(whole_block):
#     scaling_results_path = get_scaling_results_path(whole_block)
#     if scaling_results_path.exists():
#         scaling_results_df = pd.read_csv(scaling_results_path, index_col=index)
#     else:
#         scaling_results_df = pd.DataFrame(columns=columns)
#         scaling_results_df.set_index(index, drop=True, inplace=True)
#     return scaling_results_df




# small_sizes = [
#     format_size(*dims)
#     for dims in [
#         (2, 2, 2),
#         (4, 2, 2),
#         (4, 4, 2),
#         (4, 4, 4),
#         (8, 4, 4),
#         (8, 8, 4),
#         (8, 8, 8),
#     ]
# ]

# non_aligned_sizes = [
#     format_size(*dims)
#     for dims in [
#         (3, 2, 2),
#         (5, 6, 6),
#         (7, 5, 5),
#         (9, 6, 5),
#         (10, 8, 4),
#         (11, 4, 4),
#         (13, 5, 5),
#     ]
# ]
# # print(target_snitch_cluster_path.absolute())
# config_modifiers = {
#     64: {
#         # "SINGLE_CORE": [*small_sizes, format_size(16, 8, 8)],
#         "SINGLE_CORE_OPT": [
#             *small_sizes,
#             format_size(16, 8, 8),
#             format_size(16, 16, 8),
#             format_size(16, 16, 16),
#         ],
#         "MULTICORE_OPT": [
#             *small_sizes,
#             format_size(16, 8, 8),
#             format_size(16, 16, 8),
#             format_size(16, 16, 16),
#             format_size(16, 64, 64),
#             format_size(32, 16, 16),
#             format_size(32, 32, 16),
#             format_size(32, 32, 32),
#             format_size(32, 64, 32),
#             format_size(32, 64, 64),
#         ],
#     },
#     32: {
#         "SINGLE_CORE_OPT": [
#             *small_sizes,
#             *non_aligned_sizes,
#             format_size(16, 8, 8),
#             format_size(16, 16, 8),
#             format_size(16, 16, 16),
#         ],
#         "MULTICORE_OPT": [
#             *small_sizes,
#             format_size(15, 16, 16),
#             format_size(15, 23, 23),
#             format_size(16, 8, 8),
#             format_size(16, 16, 8),
#             format_size(16, 16, 16),
#             format_size(16, 64, 64),
#             format_size(32, 16, 16),
#             format_size(32, 32, 16),
#             format_size(32, 32, 32),
#             format_size(32, 64, 32),
#             format_size(32, 64, 64),
#             format_size(64, 64, 64),
#         ],
#     },
#     # 16: {
#     #     "SINGLE_CORE_OPT": [
#     #         *small_sizes,
#     #         *non_aligned_sizes,
#     #         format_size(16, 8, 8),
#     #         format_size(16, 16, 8),
#     #         format_size(16, 16, 16),
#     #     ],
#     # },
# }

# # config_modifiers = {32: {"SINGLE_CORE_OPT": [format_size(16, 16, 16)]}}


# def parse_args():
#     parser = argparse.ArgumentParser(
#         "generate_scaling_results.py",
#         description="Generates scaling results for various input sizes for batchnorm backward",
#     )

#     parser.add_argument("--whole-block", action="store_true")
#     parser.add_argument(
#         "--perf-counter-label", action="append", dest="perf_counter_labels"
#     )

#     return parser.parse_args()


# def main():
#     args = parse_args()
#     is_whole_block = args.whole_block
#     perf_counter_labels = args.perf_counter_labels
#     # yes this is mutation of a global variable but this is a script
#     columns.extend(perf_counter_labels)
#     scaling_results_df = read_existing_results(is_whole_block)
#     data = []
#     try:
#         for prec, impl, config in progressbar.progressbar(
#             flatten_config_list(config_modifiers)
#         ):
#             merged_config = {
#                 **base_config,
#                 **config,
#                 "impl_opt_level": impl,
#                 "prec": prec,
#             }
#             C = merged_config["input_dim"]["channels"]
#             H = merged_config["input_dim"]["height"]
#             W = merged_config["input_dim"]["width"]
#             TILE_CI = merged_config["tile_ci"]
#             if (prec, impl, C, H, W) in scaling_results_df.index:
#                 continue
#             total_size = C * H * W
#             with open(config_path, "w") as config_file:
#                 hjson.dump(merged_config, config_file)
#             layout_path = get_layout_path(is_whole_block)

#             # Simulate
#             p = subprocess.run(
#                 f"""make DEBUG=ON sw \
#                     && make verify-batchnorm \
#                     && make -j traces logs/perf.csv logs/event.csv \
#                     && ../../util/trace/layout_events.py logs/event.csv ../../sw/dnn/batchnorm/{layout_path} --cfg cfg/lru.hjson -o logs/trace.csv
#                 """,
#                 shell=True,
#                 cwd=target_snitch_cluster_path,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.DEVNULL,
#             )
#             p.check_returncode()

#             # Extract cycle count
#             # what information do I want to get? overall cycles obviously. Maybe utilization of the main loop?
#             with open(
#                 target_snitch_cluster_path / "logs" / "trace.csv"
#             ) as trace_results:
#                 trace_df = pd.read_csv(trace_results)
#             # Ignore the tile size calculation - that can be assumed to stay static for a given input size
#             cycles = (trace_df["done"] - trace_df["start_main"]).iloc[0]

#             # currently section 8 is main loop
#             with open(
#                 target_snitch_cluster_path / "logs" / "perf.csv"
#             ) as raw_perf_results:
#                 raw_perf_df = pd.read_csv(raw_perf_results)

#             main_loop_mcycle_section = 2 if is_whole_block else 8
#             main_loop_fpu_occupancy = raw_perf_df.loc[
#                 0, f"{main_loop_mcycle_section}_fpss_fpu_occupancy"
#             ]
#             main_loop_total_ipc = raw_perf_df.loc[
#                 0, f"{main_loop_mcycle_section}_total_ipc"
#             ]
#             main_loop_snitch_occupancy = raw_perf_df.loc[
#                 0, f"{main_loop_mcycle_section}_snitch_occupancy"
#             ]

#             grad_ifmap_max_abs_err, grad_ifmap_max_rel_err = pd.read_csv(
#                 grad_ifmap_errors_path
#             ).max()[2:4]
#             grad_weight_max_abs_err, grad_weight_max_rel_err = pd.read_csv(
#                 grad_weight_errors_path
#             ).max()[2:4]
#             grad_bias_max_abs_err, grad_bias_max_rel_err = pd.read_csv(
#                 grad_bias_errors_path
#             ).max()[2:4]
#             try:
#                 perf_counters_raw = subprocess.check_output(
#                     r"grep -oPh 'unknown_7c4.*#; .* = \K[0-9a-fx]+' logs/trace_hart_00000000.txt",
#                     text=True,
#                     shell=True,
#                     cwd=target_snitch_cluster_path,
#                 )
#             except subprocess.CalledProcessError as e:
#                 assert e.returncode == 1  # no results
#                 perf_counters_raw = e.output

#             perf_counters_int = [
#                 int(counter_val, 0) for counter_val in perf_counters_raw.strip().split()
#             ][: len(perf_counter_labels)]
#             # pad for counters not present
#             perf_counters_int.extend(
#                 [None] * (len(perf_counter_labels) - len(perf_counters_int))
#             )

#             subprocess.check_output(
#                 f"make mcycle={main_loop_mcycle_section} dma-bound-barrier",
#                 shell=True,
#                 cwd=target_snitch_cluster_path,
#                 stderr=subprocess.DEVNULL,
#             )
#             barrier_times = pd.read_csv(
#                 target_snitch_cluster_path
#                 / "logs"
#                 / f"barrier-timings-for-mcycle-{main_loop_mcycle_section}.csv"
#             )
#             if impl == "MULTICORE_OPT":
#                 # 0th one matters - first load
#                 # 1st one matters - first tile load in
#                 # for looped, 2nd one doesn't matter because it is info.
#                 # for looped, 3rd, 5th, and so on matter
#                 # for non looped, 2nd one also doesn't matter because dma immediately waits
#                 # so the potentially dma-bound ones are 0, 1, odd, and last barrier
#                 total_time_waiting_in_barrier = sum(
#                     (barrier_times["core_8"] - barrier_times["core_0"])
#                     .clip(0, None)
#                     .iloc[
#                         [
#                             i
#                             for i in barrier_times.index
#                             if i in (0, 1, barrier_times.index[-1]) or i % 2 == 1
#                         ]
#                     ]
#                 )
#             else:
#                 total_time_waiting_in_barrier = sum(
#                     (barrier_times["core_8"] - barrier_times["core_0"]).clip(0, None)
#                 )

#             data.append(
#                 (
#                     prec,
#                     impl,
#                     C,
#                     H,
#                     W,
#                     TILE_CI,
#                     total_size,
#                     cycles,
#                     main_loop_fpu_occupancy,
#                     main_loop_total_ipc,
#                     main_loop_snitch_occupancy,
#                     grad_ifmap_max_abs_err,
#                     grad_ifmap_max_rel_err,
#                     grad_weight_max_abs_err,
#                     grad_weight_max_rel_err,
#                     grad_bias_max_abs_err,
#                     grad_bias_max_rel_err,
#                     total_time_waiting_in_barrier,
#                     *perf_counters_int,
#                 )
#             )
#             subprocess.run(
#                 f"cp logs/hart_00000000_perf.json ../../sw/dnn/batchnorm/scaling_raw_results/{prec}b_{impl}_{C}_{H}_{W}_{TILE_CI}_hart_00000000_perf.json",
#                 shell=True,
#                 cwd=target_snitch_cluster_path,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.STDOUT,
#             ).check_returncode()
#             subprocess.run(
#                 f"cp logs/trace_hart_00000000.txt ../../sw/dnn/batchnorm/scaling_raw_results/{prec}b_{impl}_{C}_{H}_{W}_{TILE_CI}_trace_hart_00000000.txt",
#                 shell=True,
#                 cwd=target_snitch_cluster_path,
#                 stdout=subprocess.DEVNULL,
#                 stderr=subprocess.STDOUT,
#             ).check_returncode()
#     except Exception as e:
#         traceback.print_exc()
#     except KeyboardInterrupt:
#         write_results(data, scaling_results_df, args.whole_block)
#         exit(130)

#     write_results(data, scaling_results_df, args.whole_block)


if __name__ == "__main__":
    main()
