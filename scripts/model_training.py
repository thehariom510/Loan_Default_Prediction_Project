from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from feature_engineering import feature_engineering

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, '../models/loan_default_model.pkl')
    print("Model training completed and saved.")

if __name__ == "__main__":
    df = pd.read_csv('../data/Loan_Default.csv')
    df = df.dropna()
    df = feature_engineering(df)

    X = df.drop('Loan_Default', axis=1)  # Assume 'Loan_Default' is target column
    y = df['Loan_Default']

    train_model(X, y)
