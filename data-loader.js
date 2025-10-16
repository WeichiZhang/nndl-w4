import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class StockDataLoader {
    constructor() {
        this.stockSymbols = [];
        this.trainTestSplit = 0.8;
    }

    async loadCSV(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    const parsed = this.parseCSV(csv);
                    resolve(parsed);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        console.log('Total lines:', lines.length);
        
        const headers = lines[0].split(',').map(h => h.trim());
        console.log('Headers:', headers);

        const rawData = [];
        for (let i = 1; i < Math.min(lines.length, 100); i++) { // Process only first 100 rows for testing
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length === headers.length) {
                const row = {};
                headers.forEach((header, idx) => {
                    row[header] = values[idx];
                });
                
                // Convert numeric values
                if (row.Open && row.Close && row.Symbol) {
                    row.Open = parseFloat(row.Open);
                    row.Close = parseFloat(row.Close);
                    if (!isNaN(row.Open) && !isNaN(row.Close)) {
                        rawData.push(row);
                    }
                }
            }
        }

        console.log(`Parsed ${rawData.length} valid rows`);
        return rawData;
    }

    preprocessData(rawData) {
        console.log('Starting preprocessing...');
        
        // Get unique symbols
        this.stockSymbols = [...new Set(rawData.map(row => row.Symbol))].slice(0, 10); // Limit to 10 stocks
        console.log('Stocks:', this.stockSymbols);

        // Group by date
        const dataByDate = {};
        rawData.forEach(row => {
            if (!dataByDate[row.Date]) {
                dataByDate[row.Date] = {};
            }
            dataByDate[row.Date][row.Symbol] = {
                Open: row.Open,
                Close: row.Close
            };
        });

        // Get sorted dates
        const dates = Object.keys(dataByDate).sort();
        console.log('Total dates:', dates.length);

        // Create simple sequences (bypass complex normalization for now)
        const sequences = [];
        const targets = [];
        
        // Use only recent data to ensure we have enough sequences
        const recentDates = dates.slice(-100); // Last 100 days
        
        for (let i = 12; i < recentDates.length - 3; i++) {
            const sequence = [];
            let valid = true;
            
            // Create input sequence (last 12 days)
            for (let j = i - 12; j < i; j++) {
                const date = recentDates[j];
                const features = [];
                
                for (const symbol of this.stockSymbols) {
                    const data = dataByDate[date]?.[symbol];
                    if (data) {
                        // Simple normalization: divide by 1000 to get values ~0-1
                        features.push(data.Open / 1000, data.Close / 1000);
                    } else {
                        features.push(0, 0);
                        valid = false;
                    }
                }
                sequence.push(features);
            }

            // Create target (next 3 days)
            if (valid) {
                const target = [];
                const currentDate = recentDates[i];
                const currentPrices = {};
                
                // Get current prices for comparison
                for (const symbol of this.stockSymbols) {
                    currentPrices[symbol] = dataByDate[currentDate]?.[symbol]?.Close || 0;
                }

                // Check next 3 days
                for (let offset = 1; offset <= 3; offset++) {
                    const futureDate = recentDates[i + offset];
                    for (const symbol of this.stockSymbols) {
                        const futurePrice = dataByDate[futureDate]?.[symbol]?.Close || 0;
                        const currentPrice = currentPrices[symbol];
                        target.push(futurePrice > currentPrice ? 1 : 0);
                    }
                }

                sequences.push(sequence);
                targets.push(target);
            }
        }

        console.log(`Created ${sequences.length} sequences`);

        // Split data
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        const X_train = sequences.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const y_test = targets.slice(splitIndex);

        console.log(`Training: ${X_train.length}, Test: ${X_test.length}`);

        return {
            X_train: tf.tensor(X_train, [X_train.length, 12, 20]),
            y_train: tf.tensor(y_train, [y_train.length, 30]),
            X_test: tf.tensor(X_test, [X_test.length, 12, 20]),
            y_test: tf.tensor(y_test, [y_test.length, 30]),
            symbols: this.stockSymbols
        };
    }
}
