# Filling-Null-Values-in-Datasets
# Data Imputation API

This project provides a Flask-based API for performing data imputation on uploaded CSV files. It supports multiple imputation methods, including K-Nearest Neighbors (KNN) and Multiple Imputation by Chained Equations (MICE).

Takes in dataset with null values apllies KNN and MICE methods and fills the null values of the datasrts accordingly.

## Features

- **Custom KNN Imputation**: A manual implementation of KNN imputation.
- **Fast KNN Imputation**: Uses `sklearn`'s `KNNImputer` for efficient KNN-based imputation.
- **MICE Imputation**: Implements MICE using linear regression for filling missing values.
- **File Upload and Processing**: Upload a CSV file, and the API processes it to fill missing values.
- **File Download**: Download the processed files with imputed values.

## Requirements

- Python 3.7+
- Flask
- Flask-CORS
- pandas
- numpy
- scikit-learn

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/data-imputation-api.git
   cd data-imputation-api
