// To learn more about this code. Read our tutorial at
// https://deeplearnjs.org/docs/tutorials/ml_beginners.html

console.log('Hello from TypeScript');

import * as dl from 'deeplearn';

// Step 1. Set up variables.
const a = dl.variable(dl.scalar(Math.random()));
const b = dl.variable(dl.scalar(Math.random()));
const c = dl.variable(dl.scalar(Math.random()));


// Step 2. Create an optimizer, we will use this later
const learningRate: number = 0.01;
const optimizer = dl.train.sgd(learningRate);

// Step 3. Write our training process functions.
function predict(input) {
  // y = a * x ^ 2 + b * x + c
  return dl.tidy(() => {
    const x = dl.scalar(input);

    const ax2 = a.mul(x.square());
    const bx = b.mul(x);
    const y = ax2.add(bx).add(c);

    return y;
  });
}

function loss(prediction, actual) {
  // Having a good error metric is key for training a machine learning model
  const error = dl.scalar(actual).sub(prediction).square();
  return error;
}

async function train(xs, ys, numIterations, done) {
  let currentIteration = 0;

  for (let iter = 0; iter < numIterations; iter++) {
    for (let i = 0; i < xs.length; i++) {
      optimizer.minimize(() => {
        const pred = predict(xs[i]);
        const predLoss = loss(pred, ys[i]);

        return predLoss;
      });
    }

    // Use dl.nextFrame to not block the browser.
    await dl.nextFrame();
  }

  done();
}


function test(xs, ys) {
  dl.tidy(() => {
    const predictedYs = xs.map(predict);
    console.log('Expected', ys);
    console.log('Got', predictedYs.map((p) => p.dataSync()[0]));
  })
}


const data = {
  xs: [0, 1, 2, 3],
  ys: [1.1, 5.9, 16.8, 33.9]
};

// Lets see how it does before training.
console.log('Before training: using random coefficients')
test(data.xs, data.ys);
train(data.xs, data.ys, 50, () => {
  console.log(
      `After training: a=${a.dataSync()}, b=${b.dataSync()}, c=${c.dataSync()}`)
  test(data.xs, data.ys);
});