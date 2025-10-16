import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class GRUModel {
    constructor(inputShape = [12, 20]) {
        this.model = null;
        this.inputShape = inputShape;
        this.history = null;
    }

    buildModel() {
        try {
            this.model = tf.sequential();
            
            // First GRU layer
            this.model.add(tf.layers.gru({
                units: 32, // Reduced for browser performance
                returnSequences: true,
                inputShape: this.inputShape
            }));
            
            // Second GRU layer
            this.model.add(tf.layers.gru({
                units: 16, // Reduced for browser performance
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

            console.log('Model built successfully');
            return this.model.summary();
        } catch (error) {
            console.error('Model building error:', error);
            throw error;
        }
    }

    async train(X_train, y_train, X_test, y_test, epochs = 50, batchSize = 32) {
        try {
            console.log('Starting training...');
            this.history = await this.model.fit(X_train, y_train, {
                epochs,
                batchSize,
                validationData: [X_test, y_test],
                callbacks: {
                    onEpochEnd: (epoch, logs) => {
                        console.log(`Epoch ${epoch + 1}: loss = ${logs.loss.toFixed(4)}, accuracy = ${logs.binaryAccuracy.toFixed(4)}`);
                    }
                }
            });
            console.log('Training completed');
        } catch (error) {
            console.error('Training error:', error);
            throw error;
        }
    }

    async predict(X) {
        try {
            return await this.model.predict(X);
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    evaluate(X_test, y_test) {
        try {
            return this.model.evaluate(X_test, y_test);
        } catch (error) {
            console.error('Evaluation error:', error);
            throw error;
        }
    }

    computeStockAccuracy(predictions, yTrue, symbols) {
        try {
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
                
                stockAccuracies[symbol] = total > 0 ? correct / total : 0;
            });
            
            return stockAccuracies;
        } catch (error) {
            console.error('Accuracy computation error:', error);
            return {};
        }
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
    }
}
