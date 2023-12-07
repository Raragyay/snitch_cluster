const fs = require("fs");

// assume current dir is target/snitch_cluster
const files = fs.readdirSync("logs");

const benchmarks = files.filter((file) => file.endsWith(".json")).slice(0, -1);

const datapoints = {};

let key = 8;
for (let i = 0; i < 10; ++i) {
  datapoints[key] = {
    one: {
      dma: [],
      compute: []
    },
    two: {
      dma: [],
      compute: []
    },
    three: {
      dma: [],
      compute: []
    }
  };
  key *= 2;
}

for (const benchmark of benchmarks) {
  const data = JSON.parse(fs.readFileSync(`logs/${benchmark}`));

  let cnt = 1;
  key = 8;
  for (let i = 0; i < 9; ++i) {
    datapoints[key].one.dma.push(data[cnt++]);
    datapoints[key].one.compute.push(data[cnt++]);
    datapoints[key].one.dma.push(data[cnt++]);

    datapoints[key].two.dma.push(data[cnt++]);
    datapoints[key].two.compute.push(data[cnt++]);
    datapoints[key].two.dma.push(data[cnt++]);
    
    datapoints[key].three.dma.push(data[cnt++]);
    datapoints[key].three.compute.push(data[cnt++]);
    datapoints[key].three.dma.push(data[cnt++]);
    ++cnt;
    key *= 2;
  }
}

fs.writeFileSync("results.json", JSON.stringify(datapoints));