import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from feature_engineering import feature_engineering

if __name__ == "__main__":
    model = joblib.load('../models/loan_default_model.pkl')
    df = pd.read_csv('../data/Loan_Default.csv')
    df = df.dropna()
    df = feature_engineering(df)

    X = df.drop('Loan_Default', axis=1)
    y = df['Loan_Default']

    y_pred = model.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))
