const tf = require('@tensorflow/tfjs');
const {
    getData,
    getDataDNA,
    buildDataAndGetVoc,
    getVoc,
} = require('./get-data');

let myModel
const learningRate = 0.005;
const epochs = 50;
const batchSize = 10000;
const validationSplit = 0.2;

labelArray = ['G protein coupled receptors', 'Tyrosine kinase', 'Tyrosine phosophatase', 'Synthetase', 'Synthase', 'Ion Channel', 'Transcription factor']

const createModel = (learningRate) => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        units: 7,
        inputShape: [232414],
        useBias: true,
        activation: 'relu',
    }));
    model.add(tf.layers.dropout({
        rate: 0.2,
    }));
    model.add(tf.layers.dense({
        units: 7,
        activation: 'softmax',
    }));
    model.compile({
        optimizer: tf.train.adam(learningRate),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy'],
    });

    return model;
};

const trainModel = async (model, train_features, train_label, epochs, batchSize = null, validationSplit = 0.1) => {
    return await model.fit(train_features, train_label, {
        batchSize,
        epochs,
        shuffle: true,
        validationSplit,
    });
};

const castTensorToArray = (data) => {
    return Array.from(data.dataSync());
}

const getPercentArray = (predictions, labels) => {
    const allPredicted = new Array(7).fill(0);
    const rightPredicted = new Array(7).fill(0);
    predictions.forEach((data, index) => {
        allPredicted[data]++;
        if (data === labels[index]) {
            rightPredicted[data]++;
        };
    });
    return allPredicted.map((item, index) => {
        return {
            label: labelArray[index],
            percentage: Math.round((rightPredicted[index] / item) * 100, 2) + '%'
        };
    });
};

const runTestData = async (model, data, voc, stringToLog) => {
    const tensorData = await getDataDNA(data, voc);
    const testLabels = tensorData.y_labels;
    const predictions = model.predict(tensorData.x_features, {
        batchSize: batchSize
    }).argMax(-1);
    labelData = castTensorToArray(testLabels);
    predictionData = castTensorToArray(predictions);
    const percentArray = getPercentArray(predictionData, labelData);
    console.log(stringToLog);
    console.log(percentArray);
}

exports.startTraining = async () => {
    let model;
    let data = await getData('human_data.txt');
    const dataAndVoc = buildDataAndGetVoc(data);
    data = dataAndVoc.data;
    const voc = getVoc(dataAndVoc.vocData, 4);
    try {
        model = await tf.loadLayersModel('file://model/model.json');
        let monkeyData = await getData('chimp_data.txt');
        monkeyData = buildDataAndGetVoc(monkeyData).data;
        await runTestData(model, monkeyData, voc, 'Chimp Predictions.');
        let dogData = await getData('dog_data.txt');
        dogData = buildDataAndGetVoc(dogData).data;
        await runTestData(model, dogData, voc, 'Dog Predictions.');
    } catch (err) {
        if (!model) {
            const tensorData = await getDataDNA(data, voc);

            myModel = createModel(learningRate);
        
            await trainModel(myModel, tensorData.x_features, tensorData.y_labels,
                epochs, batchSize, validationSplit);
        
            await myModel.save('file://model');
            console.log('Model trained!');
        }
        
    };
};