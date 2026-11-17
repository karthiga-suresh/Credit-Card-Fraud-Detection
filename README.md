
# Credit Card Fraud Detection Using Isolation Forest

## Project Overview

Credit Card Fraud Detection is a Machine Learning project that identifies potentially fraudulent credit card transactions using the Isolation Forest algorithm.

The project uses an unsupervised anomaly detection approach to distinguish unusual transactions from normal transactions.

## Objectives

- Identify suspicious credit card transactions.
- Apply the Isolation Forest algorithm.
- Analyze transaction patterns.
- Evaluate fraud detection performance.
- Visualize model predictions and results.

## Technologies Used

- Python 3.12
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Machine Learning Algorithm

### Isolation Forest

Isolation Forest is an unsupervised machine learning algorithm used for anomaly detection.

It identifies unusual data points by isolating them from the remaining observations.

In this project:
- Normal transactions are represented by 0.
- Fraudulent transactions are represented by 1.
- Isolation Forest predicts anomalies as -1 and normal observations as 1.

## Dataset Description

Dataset: Credit Card Fraud Detection Dataset

Source: Kaggle – https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset details:
- Total transactions: 284,807
- Fraudulent transactions: 492
- Normal transactions: 284,315
- Input features: 30
- Target column: Class

The Class column is used only for evaluating the model, not for training it.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   └── creditcard.csv
│
├── outputs/
│   ├── class_distribution.png
│   ├── transaction_amount.png
│   ├── confusion_matrix.png
│   ├── fraud_detection.png
│   └── prediction_results.csv
│
├── main.py
├── README.md
└── .gitignore
```

## Installation

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Download the dataset from Kaggle.
2. Extract the ZIP file.
3. Place creditcard.csv inside the data folder.
4. Open the project in Visual Studio Code.
5. Run the following command:

```bash
python main.py
```

## Model Evaluation

The model performance is evaluated using:

- Precision
- Recall
- F1 Score
- Precision-Recall AUC
- Confusion Matrix

## Output Visualizations

The project generates the following visualizations:

1. Class Distribution
2. Transaction Amount Distribution
3. Confusion Matrix
4. Predicted Fraud Detection Distribution

## Results

The Isolation Forest model identifies anomalous transactions and evaluates its predictions against the actual transaction labels.

Actual performance values are generated when the project is executed.

## Conclusion

This project demonstrates how unsupervised machine learning can be applied to credit card fraud detection.

The Isolation Forest algorithm helps identify unusual transaction patterns and provides a foundation for developing intelligent fraud detection systems.

## Future Enhancements

- Develop a real-time fraud detection system.
- Explore additional anomaly detection algorithms.
- Improve fraud detection performance.
- Build a web-based interface for predictions.

## Author

Karthiga Suresh