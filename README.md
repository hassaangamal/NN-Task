# Weather Prediction Model Report

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1l1JZd00nDKGyjgkqpjcxL3CjHhFRRppu?usp=sharing)

## 1. Data Preprocessing

### 1.1 Initial Data Loading and Sampling
- Load weather data from CSV file using pandas
- Sample every 6th row of data (`df[5::6]`), likely for downsampling high-frequency measurements
- Convert date-time strings to pandas datetime objects using the format '%d.%m.%Y %H:%M:%S'

### 1.2 Wind Data Cleaning
- Handle missing values in wind velocity (wv) and maximum wind velocity (max_wv)
- Replace invalid measurements (-9999.0) with 0.0
- Extract wind velocity columns for further processing

### 1.3 Wind Component Calculations
- Convert wind direction from degrees to radians (`wd_rad = df.pop('wd (deg)')*np.pi / 180`)
- Calculate wind components using trigonometry:
  * Wx = wv * cos(wd_rad) - Wind velocity in x-direction
  * Wy = wv * sin(wd_rad) - Wind velocity in y-direction
  * Similar calculations for maximum wind velocity components

### 1.4 Temporal Feature Engineering
Constants used:
- Day length: 24*60*60 seconds
- Year length: 365.2425 days

Created cyclical features:
- Day sin/cos: Captures daily patterns
- Year sin/cos: Captures seasonal patterns

These cyclical features help the model understand periodic patterns in weather data.

## 2. Data Preparation for Model Training

### 2.1 Data Standardization
- Apply StandardScaler to normalize all features
- Ensures all features are on similar scales for better model training

### 2.2 Sequence Creation
- Create sequences of 48 hours (sequence_length = 48)
- Each sequence becomes an input sample
- Target variable: Temperature ('T (degC)')
- Data structure:
  * Input: 48 timesteps of features
  * Output: Temperature at the next timestep

### 2.3 Train-Validation Split
- 80% training data, 20% validation data
- Convert numpy arrays to PyTorch tensors
- Create DataLoader objects for batch processing:
  * Batch size: 64
  * Training data is shuffled
  * Validation data maintains order

## 3. Model Architecture

### 3.1 LSTM Model Structure
```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
```

Key components:
- LSTM layers:
  * Input size: Number of features
  * Hidden size: 50 units
  * Number of layers: 2
  * batch_first=True: Expects input shape (batch, sequence, features)
- Final fully connected layer for prediction

## 4. Training Process

### 4.1 Training Configuration
- Loss function: Mean Squared Error (MSELoss)
- Optimizer: Adam with learning rate 0.001
- Number of epochs: 20

### 4.2 Training Loop
For each epoch:
1. Training phase:
   - Forward pass through model
   - Calculate loss
   - Backpropagate errors
   - Update weights
2. Validation phase:
   - Evaluate model on validation set
   - Calculate validation loss
   - Print training and validation losses


## 5. Training and Validation Loss
- **Training and Validation Losses**: The model was trained for 20 epochs. The training loss decreases steadily from around 0.0056 to 0.0050, while the validation loss fluctuates between 0.0058 and 0.0073, indicating that the model is learning to fit the training data without overfitting.
- **Convergence**: The losses seem to have converged by the end of the 20 epochs, as the difference between the training and validation losses is relatively small. This suggests that the model has reached a stable point in its learning.
- **Test Loss**: At the end, the test loss is reported as 0.0243. This is the loss on the unseen test data, which is higher than the validation loss, indicating that the model may have some difficulty generalizing to new, unseen data.

Here is the training and validation loss information presented in a table format:

| Epoch | Training Loss | Validation Loss |
| ----- | ------------- | --------------- |
| 1     | 0.005596264155308363 | 0.006127467351863544 |
| 2     | 0.005579007710553451 | 0.00590663326034518 |
| 3     | 0.005571960383860318 | 0.00596630176450615 |
| 4     | 0.005519664587482776 | 0.005946928447061409 |
| 5     | 0.005490317170988752 | 0.006153738342244688 |
| 6     | 0.005494105724509397 | 0.0058546929639343125 |
| 7     | 0.005436909778392342 | 0.005906849873111877 |
| 8     | 0.005416410155998347 | 0.006012165646247362 |
| 9     | 0.005369353797828957 | 0.006355067243008581 |
| 10    | 0.005348933902654168 | 0.005960297446270563 |
| 11    | 0.005331229006278341 | 0.006083375231052439 |
| 12    | 0.005294342993090958 | 0.005962628489264529 |
| 13    | 0.005255342092254458 | 0.005975994105518851 |
| 14    | 0.0051950720060706035 | 0.0061683882358505235 |
| 15    | 0.005154354131799701 | 0.006280846895647903 |
| 16    | 0.005125846459421852 | 0.006189750911472323 |
| 17    | 0.0051165971640267805 | 0.006093868787493982 |
| 18    | 0.0050443726917090885 | 0.006167489105567642 |
| 19    | 0.005023772273979205 | 0.006079255613088012 |
| 20    | 0.004978890723328757 | 0.0073267756599813836 |
| Test Loss | 0.024308193213472218 |
