const fs = require("fs");

// assume current dir is target/snitch_cluster
const files = fs.readdirSync("logs");

const benchmarks = files.filter((file) => file.endsWith(".json")).slice(0, -1);
console.log(benchmarks);

let cycles_min = 99999999;
let cycles_avg = 0;
let cycles_max = 0;

let fpu_occ_min = 1;
let fpu_occ_avg = 0;
let fpu_occ_max = 0;

let ipc_min = 1;
let ipc_avg = 0;
let ipc_max = 0;

let snitch_occ_min = 1;
let snitch_occ_avg = 0;
let snitch_occ_max = 0;

let n = 0;

let h = 4;
let w = 4;
let d = 4;
const results = [];

for (let i = 1; i < 3; i += 2) {
  cycles_avg = 0;
  fpu_occ_avg = 0;
  ipc_avg = 0;
  snitch_occ_avg = 0;
  for (const benchmark of benchmarks) {

    let data = JSON.parse(fs.readFileSync(`logs/${benchmark}`));
    if (data.length < 3) continue;

    data = data[i];

    cycles_avg += data.cycles;
    fpu_occ_avg += data.fpss_fpu_occupancy;
    ipc_avg += data.total_ipc;
    snitch_occ_avg += data.snitch_occupancy;
  }

  cycles_avg /= 8;
  fpu_occ_avg /= 8;
  ipc_avg /= 8;
  snitch_occ_avg /= 8;

  // dim,impl,H,W,D,num_data_points,stride,dilation,padding,kernel_size,kernel_cycles,main_loop_snitch_occupancy,fpss_fpu_occupancy,total_ipc,icache_miss,tcdm_congestion,time_spent_waiting_in_barrier,optimal_fmax,comments
  results.push(["3D", "Multicore Optimized", h, w, d, h*w*d, "1x1x1", 1, 0, "4x4x4", cycles_avg, snitch_occ_avg, fpu_occ_avg, ipc_avg]);

  // if (n % 2 == 0) w *= 2;
  // else h *= 2;
  // h *= 2
  if (n % 3 == 0) d *= 2;
  else if (n % 3 == 1) w *= 2;
  else h *= 2;
  ++n;

  // const res_avg = {
  //   cycles_avg,
  //   fpu_occ_avg,
  //   ipc_avg,
  //   snitch_occ_avg
  // };
  // console.log(res_avg);
}

console.log(results.map((s) => s.join(",")).join("\n"));

// for (const benchmark of benchmarks) {
//   let data = JSON.parse(fs.readFileSync(`logs/${benchmark}`));
//   if (data.length < 3) continue;
//   ++n;
//   for (let q = 1; q < data.length; q += 2) {

//   }

//   // cycles_min = Math.min(cycles_min, data.cycles);
//   cycles_avg += data.cycles;
//   // cycles_max = Math.max(cycles_max, data.cycles);

//   // fpu_occ_min = Math.min(fpu_occ_min, data.fpss_fpu_occupancy);
//   fpu_occ_avg += data.fpss_fpu_occupancy;
//   // fpu_occ_max = Math.max(fpu_occ_max, data.fpss_fpu_occupancy);

//   // ipc_min = Math.min(ipc_min, data.total_ipc);
//   ipc_avg += data.total_ipc;
//   // ipc_max = Math.max(ipc_max, data.total_ipc);

//   // snitch_occ_min = Math.min(snitch_occ_min, data.snitch_occupancy);
//   snitch_occ_avg += data.snitch_occupancy;
//   // snitch_occ_max = Math.max(snitch_occ_max, data.snitch_occupancy);

//   // dim,impl,H,W,D,num_data_points,stride,kernel_size,kernel_cycles,main_loop_snitch_occupancy,fpss_fpu_occupancy,total_ipc,icache_miss,tcdm_congestion,time_spent_waiting_in_barrier
//   results.push(["2D", "Multicore Optimized", h, w, 1, h * w, 2, 9, cycles_avg, snitch_occ_avg, fpu_occ_avg, ipc_avg]);

//   if (n % 2 == 1) w *= 2;
//   else h *= 2;
// }

// cycles_avg /= n;
// fpu_occ_avg /= n;
// ipc_avg /= n;
// snitch_occ_avg /= n;

// const res_min = {
//   cycles_min,
//   fpu_occ_min,
//   ipc_min,
//   snitch_occ_min
// };

// const res_avg = {
//   cycles_avg,
//   fpu_occ_avg,
//   ipc_avg,
//   snitch_occ_avg
// };

// const res_max = {
//   cycles_max,
//   fpu_occ_max,
//   ipc_max,
//   snitch_occ_max
// };

// console.log(res_min);
// console.log(res_avg);
// console.log(res_max);
