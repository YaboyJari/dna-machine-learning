
require('@tensorflow/tfjs-node');
const express = require('express');
const { startTraining } = require('./model');
const PORT = 3000;

(async () => {
  console.log('Starting server...');
  const app = express();
  await app.listen(PORT);
  console.log(`Server started. Listening on port ${PORT}`);
  await startTraining();
  console.log('Training done!');
})();
