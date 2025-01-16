document.getElementById('predictButton').addEventListener('click', () => {
    const stockTicker = document.getElementById('stockInput').value.trim();
    const referenceDate = document.getElementById('dateInput').value.trim();
    const loadingElement = document.getElementById('loadingElement');
    const jokeElement = document.getElementById('jokeElement');
  
    if (!stockTicker || !referenceDate) {
        alert('Please enter both a stock ticker symbol and a reference date.');
        return;
    }
  
    // Show loading element
    loadingElement.style.display = 'block';
    jokeElement.innerHTML = '';
  
    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ stock_ticker: stockTicker, reference_date: referenceDate })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        const predictions = data.predictions;
        const futurePredictions = predictions.future_predictions; // Already inverse-transformed
        
        // Format the predictions nicely
        jokeElement.innerHTML = `
            <h3>Predictions for ${stockTicker}</h3>
            <p><strong>RNN Predictions (Inverse Transformed):</strong></p>
            <p>${predictions.rnn_predictions.map(val => val[0].toFixed(2)).join(', ')}</p>
            <p><strong>LSTM Predictions (Inverse Transformed):</strong></p>
            <p>${predictions.lstm_predictions.map(val => val[0].toFixed(2)).join(', ')}</p>
            <p><strong>Multivariate LSTM Predictions (Inverse Transformed):</strong></p>
            <p>${predictions.multivariate_lstm_predictions.map(val => val[0].toFixed(2)).join(', ')}</p>
            <h3>Generated Sequence of Future Predictions (Inverse Transformed):</h3>
            <p>${futurePredictions.map(val => val[0].toFixed(2)).join(', ')}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        jokeElement.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    })
    .finally(() => {
        // Hide loading element
        loadingElement.style.display = 'none';
    });
});
