const fs = require("fs");

const data = fs.readFileSync("logs/barrier-timings-for-mcycle-1.csv").toString();

const lines = data.split("\n").slice(1, -1);

let cores = [0, 0, 0, 0, 0, 0, 0, 0];

for (const line of lines) {
  const each = line.split(",").map((e) => parseInt(e));
  console.log(each);
  for (let i = 0; i < 8; ++i) {
    const e = each[i];
    if (!isNaN(e)) {
      if (each[8] > e) {
        cores[i] += each[8] - e;
      }
    }
    else {
      cores[i] = -1;
    }
  }
}

cores = cores.filter((c) => c !== -1);

console.log(cores.reduce((a, b) => Math.min(a, b)));
console.log(cores.reduce((a, b) => a + b) / cores.length);
console.log(cores.reduce((a, b) => Math.max(a, b)));
