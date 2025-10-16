import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js';

export class StockDataLoader {
    constructor() {
        this.stockSymbols = [];
        this.dateIndex = [];
        this.normalizedData = null;
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
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',').map(h => h.trim());
        
        console.log('CSV Headers:', headers);
        
        const rawData = [];
        for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(',').map(v => v.trim());
            if (values.length !== headers.length) {
                console.warn(`Skipping row ${i}: column count mismatch`);
                continue;
            }
            
            const row = {};
            headers.forEach((header, idx) => {
                row[header] = values[idx];
            });
            rawData.push(row);
        }
        
        console.log(`Parsed ${rawData.length} rows`);
        return rawData;
    }

    preprocessData(rawData) {
        try {
            // Extract unique symbols and dates
            this.stockSymbols = [...new Set(rawData.map(row => row.Symbol))].sort();
            const allDates = [...new Set(rawData.map(row => row.Date))].sort();
            
            console.log(`Found ${this.stockSymbols.length} stocks:`, this.stockSymbols);
            console.log(`Found ${allDates.length} dates`);

            // Pivot data: dates × (symbols × features)
            const pivotedData = {};
            allDates.forEach(date => {
                pivotedData[date] = {};
                this.stockSymbols.forEach(symbol => {
                    const row = rawData.find(r => r.Date === date && r.Symbol === symbol);
                    if (row) {
                        pivotedData[date][symbol] = {
                            Open: parseFloat(row.Open),
                            Close: parseFloat(row.Close)
                        };
                    }
                });
            });

            // Normalize per stock
            this.normalizedData = this.minMaxNormalize(pivotedData, allDates);
            this.dateIndex = allDates;
            
            return this.createSequences();
        } catch (error) {
            console.error('Preprocessing error:', error);
            throw error;
        }
    }

    minMaxNormalize(pivotedData, dates) {
        const normalized = {};
        const stockStats = {};

        // Calculate min/max per stock
        this.stockSymbols.forEach(symbol => {
            const opens = dates.map(date => pivotedData[date]?.[symbol]?.Open).filter(v => v !== undefined);
            const closes = dates.map(date => pivotedData[date]?.[symbol]?.Close).filter(v => v !== undefined);
            
            if (opens.length === 0 || closes.length === 0) {
                console.warn(`No data found for stock ${symbol}`);
                stockStats[symbol] = { openMin: 0, openMax: 1, closeMin: 0, closeMax: 1 };
                return;
            }
            
            stockStats[symbol] = {
                openMin: Math.min(...opens),
                openMax: Math.max(...opens),
                closeMin: Math.min(...closes),
                closeMax: Math.max(...closes)
            };
        });

        // Apply normalization
        dates.forEach(date => {
            normalized[date] = {};
            this.stockSymbols.forEach(symbol => {
                const data = pivotedData[date]?.[symbol];
                if (data) {
                    const stats = stockStats[symbol];
                    // Avoid division by zero
                    const openRange = stats.openMax - stats.openMin || 1;
                    const closeRange = stats.closeMax - stats.closeMin || 1;
                    
                    normalized[date][symbol] = {
                        Open: (data.Open - stats.openMin) / openRange,
                        Close: (data.Close - stats.closeMin) / closeRange
                    };
                }
            });
        });

        return normalized;
    }

    createSequences() {
        const sequences = [];
        const targets = [];
        const sequenceLength = 12;
        const predictionHorizon = 3;

        console.log('Creating sequences...');

        for (let i = 0; i < this.dateIndex.length - sequenceLength - predictionHorizon; i++) {
            const sequenceStart = i;
            const sequenceEnd = i + sequenceLength;
            const targetDate = this.dateIndex[sequenceEnd];

            // Input sequence: 12 days × 20 features (10 stocks × [Open, Close])
            const sequence = [];
            let validSequence = true;
            
            for (let j = sequenceStart; j < sequenceEnd; j++) {
                const date = this.dateIndex[j];
                const features = [];
                
                this.stockSymbols.forEach(symbol => {
                    const stockData = this.normalizedData[date]?.[symbol];
                    if (stockData) {
                        features.push(stockData.Open, stockData.Close);
                    } else {
                        features.push(0, 0); // Padding for missing data
                        validSequence = false;
                    }
                });
                sequence.push(features);
            }

            // Target: 30 binary values (10 stocks × 3 days)
            const target = [];
            const baseClosePrices = {};
            
            this.stockSymbols.forEach(symbol => {
                baseClosePrices[symbol] = this.normalizedData[targetDate]?.[symbol]?.Close || 0;
            });

            let validTarget = true;
            for (let offset = 1; offset <= predictionHorizon; offset++) {
                const futureDate = this.dateIndex[sequenceEnd + offset];
                this.stockSymbols.forEach(symbol => {
                    const futureClose = this.normalizedData[futureDate]?.[symbol]?.Close;
                    const baseClose = baseClosePrices[symbol];
                    if (futureClose !== undefined && baseClose !== undefined) {
                        target.push(futureClose > baseClose ? 1 : 0);
                    } else {
                        target.push(0);
                        validTarget = false;
                    }
                });
            }

            if (validSequence && validTarget && sequence.length === sequenceLength && target.length === 30) {
                sequences.push(sequence);
                targets.push(target);
            }
        }

        console.log(`Created ${sequences.length} sequences`);

        // Split chronologically
        const splitIndex = Math.floor(sequences.length * this.trainTestSplit);
        const X_train = sequences.slice(0, splitIndex);
        const X_test = sequences.slice(splitIndex);
        const y_train = targets.slice(0, splitIndex);
        const y_test = targets.slice(splitIndex);

        console.log(`Training samples: ${X_train.length}, Test samples: ${X_test.length}`);

        // Use tf.tensor with explicit shapes - this is the critical fix
        return {
            X_train: tf.tensor(X_train, [X_train.length, 12, 20]),
            y_train: tf.tensor(y_train, [y_train.length, 30]),
            X_test: tf.tensor(X_test, [X_test.length, 12, 20]),
            y_test: tf.tensor(y_test, [y_test.length, 30]),
            symbols: this.stockSymbols
        };
    }
}
