# Credit Card Fraud Detection Using Isolation Forest

## Project Overview
Credit Card Fraud Detection is a Machine Learning project that identifies potentially fraudulent credit card transactions using the **Isolation Forest** algorithm.

The project uses an unsupervised machine learning approach to detect unusual transaction patterns and classify transactions as normal or potentially fraudulent.

## Objectives
- Detect potentially fraudulent credit card transactions.
- Understand unusual transaction patterns.
- Apply an unsupervised machine learning algorithm.
- Visualize transaction data and model predictions.
- Evaluate the fraud detection results.

## Technologies Used
- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Algorithm Used

### Isolation Forest
Isolation Forest is an unsupervised machine learning algorithm used for anomaly detection.

It identifies unusual data points by isolating them from the remaining data. Transactions that are different from normal transaction patterns can be identified as potential fraud.

## Dataset Description
The project uses the Credit Card Fraud Detection dataset.

- Total Transactions: 284,807
- Fraudulent Transactions: 492
- Normal Transactions: 284,315
- Features: 30 input features and 1 target column (`Class`)

The `Class` column is used for evaluating the model predictions and is not used as an input feature during model training.

**Note:** The dataset is not included in this GitHub repository because of its size. It is stored locally in the project folder.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── main.py
├── README.md
├── .gitignore
├── class_distribution.png
├── transaction_amount.png
├── confusion_matrix.png
├── fraud_detection.png
└── prediction_results.csv
```

## Installation

Install the required Python libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run the Project

1. Download or clone the repository.
2. Place the dataset in the required local data folder.
3. Open the project folder in Visual Studio Code.
4. Install the required libraries.
5. Run the following command in the terminal:

```bash
python main.py
```

6. The program processes the transaction data and generates visualizations and prediction results.

## Project Outputs

The project generates the following outputs:

### 1. Class Distribution
Displays the distribution of normal and fraudulent transactions.

### 2. Transaction Amount
Visualizes the transaction amount distribution.

### 3. Confusion Matrix
Shows the comparison between actual and predicted transaction classes.

### 4. Fraud Detection
Displays the fraud detection results.

### 5. Prediction Results
A CSV file containing the transaction prediction results.

## Output Files

- `class_distribution.png`
- `transaction_amount.png`
- `confusion_matrix.png`
- `fraud_detection.png`
- `prediction_results.csv`

## Conclusion
This project demonstrates how Isolation Forest can be used for detecting potentially fraudulent credit card transactions using an unsupervised machine learning approach.

It provides practical experience in data preprocessing, anomaly detection, machine learning, and data visualization.

## Future Enhancements
- Improve fraud detection accuracy.
- Explore additional anomaly detection algorithms.
- Develop a simple web application for fraud detection.
- Implement real-time transaction monitoring.

## Author
**Karthiga Suresh**

GitHub: https://github.com/karthiga-suresh
