from flask import Flask, request, jsonify
import warnings
import yfinance as yf
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
from ml_pipeline.utils import train_test_split, split_sequence, process_and_split_multivariate_data
from ml_pipeline.train import train_rnn_model, train_lstm_model, train_multivariate_lstm
from flask_cors import CORS
from projectpro import model_snapshot, checkpoint

# Suppress warnings
warnings.filterwarnings('ignore')

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Override Yahoo Finance's download method
yf.pdr_override()

@app.route('/')
def index():
    return "Welcome to the Stock Prediction API. Use the /predict endpoint to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the content type is application/json
        if request.content_type != 'application/json':
            return jsonify({'error': 'Content-Type must be application/json'}), 415

        # Get stock ticker and reference date from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        stock_ticker = data.get('stock_ticker')
        reference_date_str = data.get('reference_date')
        if not stock_ticker or not reference_date_str:
            return jsonify({'error': 'Stock ticker and reference date are required'}), 400
        
        reference_date = datetime.strptime(reference_date_str, '%Y-%m-%d')
        
        # Load historical stock price data
        dataset = pdr.get_data_yahoo(stock_ticker, start='2012-01-01', end=reference_date)
        if dataset.empty:
            return jsonify({'error': f'No data found for stock ticker: {stock_ticker}'}), 404
            
        checkpoint('34db30')
        print("Data Loaded")

        # Set the start and end years for data splitting
        tstart = 2016
        tend = 2023

        # Split the dataset into training and test sets
        training_set, test_set = train_test_split(dataset, tstart, tend)

        # Scale dataset values using Min-Max scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set = training_set.reshape(-1, 1)
        training_set_scaled = sc.fit_transform(training_set)

        # Create overlapping window batches
        n_steps = 1
        features = 1
        X_train, y_train = split_sequence(training_set_scaled, n_steps)

        # Reshape X_train for model compatibility
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)

        # Train the RNN model and save it
        model_rnn = train_rnn_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path="output/model_rnn.h5")
        model_snapshot("34db30")

        # Train the LSTM model and save it
        model_lstm = train_lstm_model(X_train, y_train, n_steps, features, sc, test_set, dataset, epochs=10, batch_size=32, verbose=1, steps_in_future=25, save_model_path="output/model_lstm.h5")

        # Set the number of multivariate features
        mv_features = 6

        # Process and split multivariate data
        X_train_mv, y_train_mv, X_test_mv, y_test_mv, mv_sc = process_and_split_multivariate_data(dataset, tstart, tend, mv_features)

        # Train the multivariate LSTM model and save it
        model_mv = train_multivariate_lstm(X_train_mv, y_train_mv, X_test_mv, y_test_mv, mv_features, mv_sc, save_model_path="output/model_mv_lstm.h5")
        model_snapshot("34db30")

        # Generate Predictions
        predictions = {
            "rnn_predictions": sc.inverse_transform(model_rnn.predict(X_train)).tolist(),
            "lstm_predictions": sc.inverse_transform(model_lstm.predict(X_train)).tolist(),
            "multivariate_lstm_predictions": sc.inverse_transform(model_mv.predict(X_test_mv)).tolist()
        }

        # Generate sequence of future predictions
        future_steps = 25
        future_predictions = model_lstm.predict(X_train[-1].reshape(1, n_steps, features))
        for _ in range(future_steps - 1):
            future_predictions = np.append(future_predictions, model_lstm.predict(future_predictions[-1].reshape(1, n_steps, features)), axis=0)

        # Include future predictions in the predictions dictionary
        predictions["future_predictions"] = sc.inverse_transform(future_predictions).tolist()

        return jsonify({
            "status": "success",
            "stock_ticker": stock_ticker,
            "predictions": predictions
        })

    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)