import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { StockDataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new StockDataLoader();
        this.model = new GRUModel();
        this.trainData = null;
        this.testData = null;
        this.predictions = null;
        this.initializeUI();
    }

    initializeUI() {
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        const progressDiv = document.getElementById('progress');
        
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        trainBtn.addEventListener('click', () => this.startTraining());
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        try {
            this.updateProgress('Loading CSV file...');
            const rawData = await this.dataLoader.loadCSV(file);
            
            this.updateProgress('Preprocessing data...');
            this.trainData = await this.dataLoader.preprocessData(rawData);
            
            this.updateProgress('Data loaded successfully!');
            document.getElementById('trainBtn').disabled = false;
        } catch (error) {
            this.updateProgress(`Error: ${error.message}`);
        }
    }

    async startTraining() {
        if (!this.trainData) {
            this.updateProgress('Please load data first');
            return;
        }

        try {
            this.updateProgress('Building model...');
            this.model.buildModel();

            this.updateProgress('Starting training...');
            await this.model.train(
                this.trainData.X_train, 
                this.trainData.y_train, 
                this.trainData.X_test, 
                this.trainData.y_test,
                50,  // epochs
                32   // batchSize
            );

            this.updateProgress('Making predictions...');
            this.predictions = await this.model.predict(this.trainData.X_test);
            
            this.evaluateAndVisualize();
            
        } catch (error) {
            this.updateProgress(`Training error: ${error.message}`);
        }
    }

    evaluateAndVisualize() {
        // Compute stock accuracies
        const stockAccuracies = this.model.computeStockAccuracy(
            this.predictions, 
            this.trainData.y_test, 
            this.trainData.symbols
        );

        // Sort stocks by accuracy
        const sortedStocks = Object.entries(stockAccuracies)
            .sort(([,a], [,b]) => b - a)
            .reduce((acc, [symbol, accuracy]) => {
                acc[symbol] = accuracy;
                return acc;
            }, {});

        // Visualize results
        this.renderAccuracyChart(sortedStocks);
        this.renderPredictionTimeline(sortedStocks);
    }

    renderAccuracyChart(accuracies) {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        // Clear previous chart
        if (this.accuracyChart) {
            this.accuracyChart.destroy();
        }

        const symbols = Object.keys(accuracies);
        const accuracyValues = Object.values(accuracies);

        this.accuracyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: symbols,
                datasets: [{
                    label: 'Prediction Accuracy',
                    data: accuracyValues,
                    backgroundColor: accuracyValues.map(acc => 
                        acc > 0.5 ? 'rgba(75, 192, 192, 0.6)' : 'rgba(255, 99, 132, 0.6)'
                    ),
                    borderColor: accuracyValues.map(acc => 
                        acc > 0.5 ? 'rgb(75, 192, 192)' : 'rgb(255, 99, 132)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'Accuracy'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Stock Prediction Accuracy Ranking'
                    }
                }
            }
        });
    }

    renderPredictionTimeline(accuracies) {
        const container = document.getElementById('timelineContainer');
        container.innerHTML = '';

        Object.keys(accuracies).forEach(symbol => {
            const timelineDiv = document.createElement('div');
            timelineDiv.className = 'stock-timeline';
            timelineDiv.innerHTML = `
                <h4>${symbol} (Accuracy: ${(accuracies[symbol] * 100).toFixed(1)}%)</h4>
                <div class="timeline" id="timeline-${symbol}"></div>
            `;
            container.appendChild(timelineDiv);

            this.renderSingleTimeline(symbol, `timeline-${symbol}`);
        });
    }

    renderSingleTimeline(symbol, elementId) {
        // Simplified timeline visualization
        // In a real implementation, you would use the actual prediction results
        const timeline = document.getElementById(elementId);
        const sampleCount = Math.min(20, this.predictions.shape[0]); // Show first 20 samples
        
        for (let i = 0; i < sampleCount; i++) {
            const point = document.createElement('div');
            point.className = 'timeline-point';
            
            // Simulate correct/incorrect predictions (replace with actual logic)
            const isCorrect = Math.random() > 0.3;
            point.style.backgroundColor = isCorrect ? '#4CAF50' : '#F44336';
            point.title = `Sample ${i}: ${isCorrect ? 'Correct' : 'Wrong'}`;
            
            timeline.appendChild(point);
        }
    }

    updateProgress(message) {
        const progressDiv = document.getElementById('progress');
        progressDiv.innerHTML = message;
        console.log(message);
    }

    dispose() {
        if (this.model) {
            this.model.dispose();
        }
        if (this.trainData) {
            this.trainData.X_train.dispose();
            this.trainData.y_train.dispose();
            this.trainData.X_test.dispose();
            this.trainData.y_test.dispose();
        }
        if (this.predictions) {
            this.predictions.dispose();
        }
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new StockPredictionApp();
});
