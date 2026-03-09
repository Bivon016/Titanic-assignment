

import pandas as pd
import numpy as np

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Family features
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    rare_titles = ['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # Age groups
    bins = [0,12,19,59,100]
    labels = ['Child','Teen','Adult','Senior']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)

    # Fare per person
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # One-hot encode categorical variables
    categorical_cols = ['Sex','Embarked','Title','Deck','AgeGroup']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Save feature-engineered dataset
    df.to_csv(output_path, index=False)
    print(f"Features engineered and saved to {output_path}")
    return df

if __name__ == "__main__":
    engineer_features("../data/train_cleaned.csv", "../data/train_features.csv")