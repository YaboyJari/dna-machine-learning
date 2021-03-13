const fs = require('fs');
const path = require('path');
const csv = require('fast-csv');
const tf = require('@tensorflow/tfjs');

const getData = async (fileName) => {
    let promise = new Promise(function (resolve, reject) {
        const features = [];
        const labels = [];
        fs.createReadStream(path.resolve(__dirname, 'files', fileName))
            .pipe(csv.parse({
                headers: true,
                delimiter: '\t'
            }))
            .on('error', error => console.error(error))
            .on('data', row => {
                features.push(row.sequence);
                labels.push(row.class);
            })
            .on('end', async (rowCount) => {
                console.log(`Parsed ${rowCount} rows`);
                resolve({
                    features,
                    labels,
                });
            });
    });

    return promise;
};

const getKmer = (sequence, size = 6) => {
    const kmerSequence = [];
    for (x = 0; x < sequence.length - size + 1; x++) {
        kmerSequence.push(sequence.substring(x, x + size).toLowerCase());
    };
    return kmerSequence;
};

const getArrayCombination = (wordArray, n) => {
    const splitArray = wordArray.split(' ');
    const allCombinations = [];
    for(x = 0; x < splitArray.length - n + 1; x++) {
        const elements = splitArray.slice(x, x + n).join(' ');
        allCombinations.push(elements);
    };
    return allCombinations;
};

const getVoc = (wordArray, numberOfWords) => {
    const wordGroups = [];
    wordArray.forEach(sentence => {
        const allWords = getArrayCombination(sentence, numberOfWords);
        wordGroups.push(allWords);
    });

    const vocSet = new Set(wordGroups.flat());

    const vocArray = Array.from(vocSet).sort();

    console.log('Made voc with ' + vocArray.length + ' words');
    return vocArray;
}

const intersectArrayIndex = (sentenceKmer, voc) => {
    let sortedSentence = sentenceKmer.concat().sort();
    let sortedVoc = voc.concat().sort();
    let indexCounter = [];
    let sentenceIndex = 0;
    let vocIndex = 0;

    while (sentenceIndex < sentenceKmer.length
           && vocIndex < voc.length)
    {
        if (sortedSentence[sentenceIndex] === sortedVoc[vocIndex]) {
            indexCounter.push(vocIndex);
            sentenceIndex++;
            vocIndex++;
        }
        else if(sortedSentence[sentenceIndex] < sortedVoc[vocIndex]) {
            sentenceIndex++;
        }
        else {
            vocIndex++;
        }
    }
    return indexCounter;
}

const searchForWordsAndReturnIndex = (features, voc) => {
    features = features.map(feature => {
        return getArrayCombination(feature, 4).sort().flat();
    });

    console.log('Array combinations calculated!');

    const indexesArray = [];

    features.forEach(feature => {
        const indexes = intersectArrayIndex(feature, voc);
        indexesArray.push(indexes);
    });

    console.log('Indexes found!')

    console.log('One-hot encoding done!');
    
    return indexesArray;
};

const setSparseToDenseTensor = (featureIndexes, voc) => {
    const indicesArray = [];
    const valueArray = [];

    featureIndexes.forEach((feature, index) => {
        feature.forEach(data => {
            indicesArray.push([index, data]);
            valueArray.push(1);
        })
    });

    const values = tf.tensor1d(valueArray, 'float32');
    const shape = [featureIndexes.length, voc.length];
    if (!indicesArray || !valueArray) {
        return new Error('Fault with indices or values');
    };
    return tf.sparseToDense(indicesArray, values, shape);
};

const transferToTensorData = (features, labels, voc) => {
    return tf.tidy(() => {
        try {
            features = setSparseToDenseTensor(features, voc);
            labels = tf.tensor(labels);
            return {
                'x_features': features,
                'y_labels': labels,
            };
        } catch (err) {
            return err;
        };
    });
};

const buildDataAndGetVoc = (data) => {
    const vocData = [];
    data.features.forEach((sequence, index) => {
        const kmerSequence = getKmer(sequence).join(' ');
        vocData.push(kmerSequence);
        data.features[index] = kmerSequence;
    });
    return {
        data,
        vocData,
    }
}

const getDataDNA = async (data, voc) => {
    intLabels = data.labels.map(Number);
    const indexesOfFeatures = searchForWordsAndReturnIndex(data.features, voc);
    return transferToTensorData(indexesOfFeatures, intLabels, voc);
}

module.exports = {
    getDataDNA,
    getData,
    getVoc,
    buildDataAndGetVoc,
}