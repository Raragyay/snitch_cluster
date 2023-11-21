const fs = require("fs");

// assume current dir is target/snitch_cluster
const files = fs.readdirSync("logs");

const benchmarks = files.filter((file) => file.endsWith(".json")).slice(0, -1);

let cycles = 0;
let util = 0;
let relUtil = 0;
let counter = 0;
for (const benchmark of benchmarks) {
  const data = JSON.parse(fs.readFileSync(`logs/${benchmark}`));
  
  for (let i = 1; i < data.length; i += 2) {
    cycles += data[i].cycles;
    util += data[i].fpss_fpu_occupancy;
    relUtil += data[i].fpss_fpu_rel_occupancy;
    ++counter;
  }
}

console.log(`cycles avg: ${cycles / 8}
fpu util avg: ${util / counter}
fpu rel util avg: ${relUtil / counter}`);
