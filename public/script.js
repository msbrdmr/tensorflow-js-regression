console.log('Hello TensorFlow');
let dataURL = "https://storage.googleapis.com/tfjs-tutorials/carsData.json"

// Function to get the data.

const visor = tfvis.visor();
visor.toggleFullScreen();
let visorOpened = false;

function toggleVisor() {
    if (!visorOpened) {
        // Open the Visor
        tfvis.visor().open();
        visorOpened = true;
    } else {
        // Close the Visor
        tfvis.visor().close();
        visorOpened = false;
    }
}
async function getData() {
    const response = await fetch(dataURL)
    const data = await response.json()
    // This data have name, acceleration, cylinders and so on. I only need the mpg and hp. Soi I am mappint this data to get a new array
    const mappedData = data.map(data => {
        return {
            // For each object, only take hp and mpg
            hp: data.Horsepower,
            mpg: data.Miles_per_Gallon
        }
    })
    // Thenü if any of the data have a null for either hp or mpg, simply remove.
    const cleanedData = mappedData.filter(data => (data.mpg != null && data.hp != null))
    return cleanedData;
}
function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
}
/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_ and _normalizing_ the data
 * MPG on the y-axis.
 */
function prepareData(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    // Executes the provided function fn and after it is executed, cleans up all intermediate tensors allocated by fn except those returned by fn
    // Step 1. Shuffle the data
    return tf.tidy(() => {
        tf.util.shuffle(data); //shuffles the array and returnes it

        // Step 2. Convert data to Tensor
        const inputs = data.map(d => d.x)
        const labels = data.map(d => d.y);
        //generated 2 seperate arrays


        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        //these 2 seperate arrays are then converted to 2d tensors of shape (sample_count, 1)

        //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
        //min-max scaling basically calculates the value in range 0-1 corresponding to that value. Formula is to divide (value-min) by (max-min)

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        verbose: 1,

        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance', tab: 'Model Training' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

function testModel(model, inputData, normalizationData) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [inputs, preds] = tf.tidy(() => {

        const normalized_inputs = tf.linspace(0, 1, 100).reshape([100, 1]) //generates 100 numbers between 0 and 1 FOR x axis, predicting y axis. Shape of xsNorm is [100]
        //xsNorm is then reshaped to 100,1 to match the input shape of the model

        const predictions = model.predict(normalized_inputs);

        const unnormalized_inputs = unnormalize(normalized_inputs, inputMin, inputMax);
        const unnormalized_predictions = unnormalize(predictions, labelMin, labelMax);

        return [unnormalized_inputs, unnormalized_predictions];
    });


    const predictedPoints = Array.from(inputs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const originalPoints = inputData.map(d => ({
        x: d.hp, y: d.mpg,
    }));


    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data', tab: 'Model Testing' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}
function unnormalize(tensor, min, max) {
    const unnormalized = tensor
        .mul(max.sub(min))
        .add(min);
    return unnormalized.dataSync();
}

async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map(d => ({
        x: d.hp,
        y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG', tab: 'Data'},
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // Create the model
    const model = createModel();
    const surface = { name: 'Model Summary', tab: 'Model Inspection' };
    tfvis.show.modelSummary(surface, model);

    // Convert the data to a form we can use for training.
    const tensorData = prepareData(values);
    const { inputs, labels } = tensorData;

    // Train the model
    const history = await trainModel(model, inputs, labels);

    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
}


// When the script is loaded, call the run function
document.addEventListener('DOMContentLoaded', run);