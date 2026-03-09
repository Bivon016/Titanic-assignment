

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def select_features(input_path):
    df = pd.read_csv(input_path)

    # Drop non-predictive columns
    X = df.drop(columns=['Survived','Name','Ticket','PassengerId'], errors='ignore')
    y = df['Survived']

    # Feature importance using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Show top 10
    print("Top 10 important features:\n", importances.head(10))

    # Plot
    importances.head(10).plot(kind='barh', figsize=(8,6), color='skyblue')
    plt.title("Top 10 Feature Importances")
    plt.gca().invert_yaxis()
    plt.show()

    return importances

if __name__ == "__main__":
    select_features("../data/train_features.csv")