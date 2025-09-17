import pandas as pd

def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    # Example cleaning steps
    df = df.dropna()
    return df

if __name__ == "__main__":
    data = load_and_clean_data('../data/Loan_Default.csv')
    print(data.head())
