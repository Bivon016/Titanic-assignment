

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def clean_data(input_path, output_path):
    # Load data
    df = pd.read_csv(input_path)

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # Extract Deck safely
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].str[0]
        df['Deck'] = df['Deck'].fillna('Unknown')
        df = df.drop(columns=['Cabin'])
    else:
        df['Deck'] = 'Unknown'

    # Remove duplicates
    df = df.drop_duplicates()

    # Log transform Fare
    df['Fare'] = np.log1p(df['Fare'])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Data cleaned and saved to {output_path}")
    return df

if __name__ == "__main__":
    clean_data("../data/train.csv", "../data/train_cleaned.csv")