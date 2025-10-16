import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class GRUModel {
    constructor(inputShape = [12, 20]) {
        this.model = null;
        this.inputShape = inputShape;
        this.history = null;
    }

    buildModel() {
        this.model = tf.sequential();
        
        // First GRU layer
        this.model.add(tf.layers.gru({
            units: 64,
            returnSequences: true,
            inputShape: this.inputShape
        }));
        
        // Second GRU layer
        this.model.add(tf.layers.gru({
            units: 32,
            returnSequences: false
        }));
        
        // Dropout for regularization
        this.model.add(tf.layers.dropout({ rate: 0.2 }));
        
        // Output layer: 30 units for 10 stocks × 3 days
        this.model.add(tf.layers.dense({
            units: 30,
            activation: 'sigmoid'
        }));

        this.model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['binaryAccuracy']
        });

        return this.model.summary();
    }

    async train(X_train, y_train, X_test, y_test, epochs = 100, batchSize = 32) {
        this.history = await this.model.fit(X_train, y_train, {
            epochs,
            batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.binaryAccuracy}`);
                    // Update UI progress here if needed
                }
            }
        });
    }

    async predict(X) {
        return this.model.predict(X);
    }

    evaluate(X_test, y_test) {
        return this.model.evaluate(X_test, y_test);
    }

    computeStockAccuracy(predictions, yTrue, symbols) {
        const predData = predictions.arraySync();
        const trueData = yTrue.arraySync();
        const stocksCount = symbols.length;
        const daysAhead = 3;
        
        const stockAccuracies = {};
        symbols.forEach((symbol, stockIdx) => {
            let correct = 0;
            let total = 0;
            
            for (let sample = 0; sample < predData.length; sample++) {
                for (let day = 0; day < daysAhead; day++) {
                    const predIdx = stockIdx + day * stocksCount;
                    const pred = predData[sample][predIdx] > 0.5 ? 1 : 0;
                    const trueVal = trueData[sample][predIdx];
                    
                    if (pred === trueVal) correct++;
                    total++;
                }
            }
            
            stockAccuracies[symbol] = correct / total;
        });
        
        return stockAccuracies;
    }

    async saveModel() {
        const saveResult = await this.model.save('downloads://stock-gru-model');
        return saveResult;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
