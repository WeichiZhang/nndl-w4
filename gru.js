import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class GRUModel {
    constructor() {
        this.model = null;
    }

    buildModel() {
        console.log('Building model...');
        
        this.model = tf.sequential({
            layers: [
                tf.layers.gru({
                    units: 16,
                    inputShape: [12, 20],
                    returnSequences: false
                }),
                tf.layers.dense({
                    units: 30,
                    activation: 'sigmoid'
                })
            ]
        });

        this.model.compile({
            optimizer: 'adam',
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        console.log('Model built');
        return this.model;
    }

    async train(X_train, y_train, X_test, y_test, epochs = 10, batchSize = 8) {
        console.log('Starting training...');
        
        await this.model.fit(X_train, y_train, {
            epochs: epochs,
            batchSize: batchSize,
            validationData: [X_test, y_test],
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)}, accuracy=${logs.acc.toFixed(4)}`);
                }
            }
        });
        
        console.log('Training completed');
    }

    async predict(X) {
        return this.model.predict(X);
    }

    computeStockAccuracy(predictions, yTrue, symbols) {
        const predArray = predictions.arraySync();
        const trueArray = yTrue.arraySync();
        
        const accuracies = {};
        symbols.forEach((symbol, symbolIndex) => {
            let correct = 0;
            let total = 0;
            
            for (let i = 0; i < predArray.length; i++) {
                for (let day = 0; day < 3; day++) {
                    const predIdx = symbolIndex + day * symbols.length;
                    const pred = predArray[i][predIdx] > 0.5 ? 1 : 0;
                    const actual = trueArray[i][predIdx];
                    
                    if (pred === actual) correct++;
                    total++;
                }
            }
            
            accuracies[symbol] = correct / total;
        });
        
        return accuracies;
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
