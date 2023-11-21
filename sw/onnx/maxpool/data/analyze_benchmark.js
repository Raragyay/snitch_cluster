const fs = require("fs");

// assume current dir is target/snitch_cluster
const files = fs.readdirSync("logs");

const benchmarks = files.filter((file) => file.endsWith(".json")).slice(0, -1);

let cycles = 0;
let util = 0;
let relUtil = 0;
for (const benchmark of benchmarks) {
  const data = JSON.parse(fs.readFileSync(`logs/${benchmark}`));

  cycles += data[1].cycles;
  util += data[1].fpss_fpu_occupancy;
  relUtil += data[1].fpss_fpu_rel_occupancy;
  
}

console.log(`cycles avg: ${cycles / 8}
fpu util avg: ${util / 8}
fpu rel util avg: ${relUtil / 8}`);
