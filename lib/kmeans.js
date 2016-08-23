/**
 * Created by joonkukang on 2014. 1. 16..
 */
var math = require('./utils').math;
Kmeans = module.exports;

Kmeans.cluster = function(options) {
  var data = options['data'];
  var k = options['k'];
  var distance = getDistanceFunction(options['distance']);
  var epochs = options['epochs'];
  var init_using_data = options['init_using_data'];
  if(typeof init_using_data === "undefined");
    init_using_data = true;
  var means = getRandomMeans(data,k, init_using_data);

  var epoch, i, j, l;
  var clusters = [];
  for(i=0 ; i<k ; i++)
    clusters.push([]);

  for(epoch=0 ; epoch<epochs ; epoch++) {
    clusters = [];
    for(i=0 ; i<k ; i++)
      clusters.push([]);

    // Find which centroid is the closest for each row
    for(i=0 ; i<data.length ; i++) {
      var bestmatch = 0;
      for(j=0 ; j<k ; j++) {
        if(distance(means[j],data[i]) < distance(means[bestmatch],data[i])) bestmatch = j;
      }
      clusters[bestmatch].push(i);
    }

    // Move the centroids to the average of their members
    for(i=0 ; i<k ; i++) {
      var avgs = [];
      for(j=0 ; j<data[0].length ; j++)
        avgs.push(0.0);
      if(clusters[i].length > 0) {
        for(j=0 ; j<clusters[i].length ; j++) {
          for(l=0 ; l<data[0].length ; l++) {
            avgs[l] += data[clusters[i][j]][l];
          }
        }
        for(j=0 ; j<data[0].length ; j++) {
          avgs[j] /= clusters[i].length;
        }
        means[i] = avgs;
      }
    }
  }
  return {
    clusters : clusters,
    means : means
  };
}

/**
 * prediction clusters to be allocated from the data already obtained
 * @param {object} opts
 * @property {number} opts.k 
 * @property {number} opts.epochs 
 * @property {object} opts.distance 
 * @property {array} opts.data 
 * @property {array} opts.sum 
 * @property {array} opts.avg 
 * @property {array} opts.min 
 * @property {array} opts.max 
 * @property {array} opts.means 
 * @return {number} clusters index
 *
 * @example
 * var opts = {
    k: 10, 
    epochs: 100, 
    distance: { type: 'euclidean'},
    data: [50, 1],
    sum: [504877,12688],
    avg: [30.04326093424576,0.7550133888723595],
    min: [0,0],
    max: [120,1],
    means: [[-0.1105677763827696,-0.7550133888723795],
    [-0.05411805800310872,0.24498661112762743],
    [0.046194295749118486,0.24498661112762335],
    [-0.2367285231996391,0.24498661112764097],
    [-0.013700819485776377,0.2449866111276253],
    [0.6730872046329195,0.2449866111276402],
    [-0.00786423634615692,-0.7550133888723796],
    [0.14650355289139733,0.24498661112763215],
    [0.1889880837639144,-0.7550133888723681],
    [-0.09064699333048662,0.24498661112763037]]
  }

  var result = ml.kmeans.predictCluster(opts);
 */
Kmeans.predictCluster = function (opts) {
  var distance = getDistanceFunction(opts['distance']);
  var match = 0;
  var minDistance = null;

  opts.data = normalization(opts)

  for (i = 0 ; i < opts.k ; i++) {
    if(!minDistance || distance(opts.means[i], opts.data) < minDistance) {
      match = i;
      minDistance = distance(opts.means[i], opts.data);
    }
  }
  return match;
}

/**
 * data normalizations
 * @param  {array} data
 * @return {object}
 *
 * @example
 * var normalizations = ml.kmeans.normalizations(data)

    var result = ml.kmeans.cluster({
      data: normalizations.data, k: 10, epochs: 100, distance: { type: 'euclidean'}
    })
 */
Kmeans.normalizations = function (data) {
  var sum = []
  var avg = []
  var min = []
  var max = []
  var results = []

  var length = data[0].length

  for (var i = 0; i < data.length; i++) {
    var value = data[i]

    for (var j = 0; j < value.length; j++) {
      if (sum.length <= j) {
        sum.push(value[j])
      } else {
        sum[j] += value[j]
      }

      if (i === 0) {
        min[j] = 0
        max[j] = 0
      }
      if (min[j] > value[j]) {
        min[j] = value[j]
      }
      if (max[j] < value[j]) {
        max[j] = value[j]
      }
    }
  }

  for (var i = 0; i < length; i++) {
    avg.push(sum[i] / data.length)
  }

  for (var i = 0; i < data.length; i++) {
    var value = data[i]

    results.push([])
    for (var j = 0; j < value.length; j++) {
      results[i][j] = (data[i][j] - avg[j]) / (max[j] - min[j])
    }
  }

  return {
    data: results,
    min: min,
    max: max,
    sum: sum,
    avg: avg
  }
}

/**
 * normalization data for prodectCluster
 * @param  {object} opts
 * @property {array} opts.data
 * @property {array} opts.avg
 * @property {array} opts.min
 * @property {array} opts.max
 * @return {array}
 */
var normalization = function (opts) {
  var options = opts || {}

  for (var i = 0; i < options.data.length; i++) {
    options.data[i] = (options.data[i] - options.avg[i]) / (options.max[i] - options.min[i])
  }

  return options.data
}

var getRandomMeans = function(data,k, init_using_data) {
  var clusters = [];
  if(init_using_data) {
    var cluster_index = math.range(data.length);
    cluster_index = math.shuffle(cluster_index);
    for(i=0 ; i<k ; i++) {
      clusters.push(data[cluster_index[i]]);
    }
  } else {
    var i,j;
    var ranges = [];
    for(i=0 ; i<data[0].length ; i++) {
      var min = data[0][i] , max = data[0][i];
      for(j=0 ; j<data.length ; j++) {
        if(data[j][i] < min) min = data[j][i];
        if(data[j][i] > max) max = data[j][i];
      }
      ranges.push([min,max]);
    }
    for(i=0 ; i<k ; i++) {
      var cluster = [];
      for(j=0 ; j<data[0].length;j++) {
        cluster.push(Math.random() * (ranges[j][1] - ranges[j][0]) + ranges[j][0]);
      }
      clusters.push(cluster);
    }
  }
  return clusters;
}


function getDistanceFunction(options) {
  if(typeof options === 'undefined') {
    return math.euclidean;
  } else if (typeof options === 'function') {
    return options;
  } else if (options['type'] === 'euclidean') {
    return math.euclidean;
  } else if (options['type'] === 'pearson') {
    return math.pearson;
  }
}
