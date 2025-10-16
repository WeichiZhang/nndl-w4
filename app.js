import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';
import { StockDataLoader } from './data-loader.js';
import { GRUModel } from './gru.js';

class StockPredictionApp {
    constructor() {
        this.dataLoader = new StockDataLoader();
        this.model = new GRUModel();
        this.trainData = null;
        this.initializeUI();
    }

    initializeUI() {
        console.log('Initializing UI...');
        
        const fileInput = document.getElementById('csvFile');
        const trainBtn = document.getElementById('trainBtn');
        
        if (!fileInput) {
            console.error('File input not found!');
            return;
        }
        if (!trainBtn) {
            console.error('Train button not found!');
            return;
        }

        fileInput.addEventListener('change', (e) => {
            console.log('File selected');
            this.handleFileUpload(e);
        });
        
        trainBtn.addEventListener('click', () => {
            console.log('Train button clicked');
            this.startTraining();
        });
        
        console.log('UI initialized successfully');
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) {
            this.updateProgress('No file selected');
            return;
        }

        console.log('Processing file:', file.name);
        this.updateProgress('Reading file...');

        try {
            // Load CSV
            const rawData = await this.dataLoader.loadCSV(file);
            console.log('Raw data loaded:', rawData.length, 'rows');
            
            // Preprocess data
            this.updateProgress('Processing data...');
            this.trainData = this.dataLoader.preprocessData(rawData);
            
            if (this.trainData && this.trainData.X_train) {
                console.log('Data processed successfully');
                this.updateProgress(`Ready! ${this.trainData.X_train.shape[0]} training samples loaded`);
                
                // Enable train button
                const trainBtn = document.getElementById('trainBtn');
                trainBtn.disabled = false;
                trainBtn.textContent = '2. Train Model (Ready)';
                console.log('Train button enabled');
            } else {
                throw new Error('Data processing failed');
            }
            
        } catch (error) {
            console.error('Error:', error);
            this.updateProgress(`Error: ${error.message}`);
        }
    }

    async startTraining() {
        if (!this.trainData) {
            this.updateProgress('Please load data first');
            return;
        }

        const trainBtn = document.getElementById('trainBtn');
        trainBtn.disabled = true;
        trainBtn.textContent = 'Training...';

        try {
            this.updateProgress('Building model...');
            this.model.buildModel();

            this.updateProgress('Training started...');
            await this.model.train(
                this.trainData.X_train, 
                this.trainData.y_train, 
                this.trainData.X_test, 
                this.trainData.y_test,
                10,  // Very few epochs for testing
                8    // Small batch size
            );

            this.updateProgress('Making predictions...');
            const predictions = await this.model.predict(this.trainData.X_test);
            
            this.updateProgress('Evaluating...');
            const accuracies = this.model.computeStockAccuracy(
                predictions, 
                this.trainData.y_test, 
                this.trainData.symbols
            );

            this.renderResults(accuracies);
            this.updateProgress('Training completed!');
            
            // Reset button
            trainBtn.disabled = false;
            trainBtn.textContent = '2. Train Model';
            
            // Clean up
            predictions.dispose();
            
        } catch (error) {
            console.error('Training failed:', error);
            this.updateProgress(`Training failed: ${error.message}`);
            trainBtn.disabled = false;
            trainBtn.textContent = '2. Train Model';
        }
    }

    renderResults(accuracies) {
        console.log('Rendering results:', accuracies);
        
        // Create simple results display
        const container = document.getElementById('timelineContainer');
        container.innerHTML = '<h3>Results</h3>';
        
        Object.entries(accuracies).forEach(([symbol, accuracy]) => {
            const div = document.createElement('div');
            div.innerHTML = `${symbol}: ${(accuracy * 100).toFixed(1)}% accuracy`;
            container.appendChild(div);
        });

        // Also update progress with summary
        const avgAccuracy = Object.values(accuracies).reduce((a, b) => a + b, 0) / Object.values(accuracies).length;
        this.updateProgress(`Average accuracy: ${(avgAccuracy * 100).toFixed(1)}%`);
    }

    updateProgress(message) {
        console.log('Progress:', message);
        const progressDiv = document.getElementById('progress');
        if (progressDiv) {
            progressDiv.textContent = message;
        }
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing app...');
    window.app = new StockPredictionApp();
});
