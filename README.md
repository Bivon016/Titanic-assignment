
This project analyzes the Titanic dataset to explore factors that influenced passenger survival. The goal is to prepare the dataset for predictive modeling by performing:

Data cleaning
 Feature engineering
 Feature selection

The analysis follows the requirements of AI Assignment and is organized as a structured GitHub project with notebooks and modular scripts.

The dataset used is Titanic: Machine Learning from Disaster, which contains passenger information such as age, gender, class, and ticket fare.



# Project Structure


titanic_assignment/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── Titanic_Feature_Engineering.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   └── feature_selection.py
│
├── README.md
└── requirements.txt


 Part 1: Data Cleaning

Several preprocessing steps were applied to improve data quality.

    Missing Value Handling

Missing values were identified in the dataset:

* **Age** contained missing values → filled using the **median age**
* **Embarked** contained a small number of missing values → filled using the **most frequent value (mode)**
* **Cabin** had many missing values → the **deck letter was extracted**, and missing values were labeled as **Unknown**

### Outlier Handling

Outliers were inspected in numerical columns:

* **Fare** showed a strong right skew
* A **log transformation** was applied to reduce skewness and improve distribution

### Data Consistency

Data consistency checks included:

* Verifying categorical values such as **Sex (male/female)**
* Removing **duplicate rows** if present

### Output

The cleaned dataset is saved as:

```
data/train_cleaned.csv
```

---

# Part 2: Feature Engineering

New features were created to capture meaningful patterns in the data.

### Derived Features

The following features were engineered:

| Feature           | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| **FamilySize**    | Total family members traveling together (SibSp + Parch + 1) |
| **IsAlone**       | Indicates if a passenger traveled alone                     |
| **Title**         | Extracted from the Name column (Mr, Mrs, Miss, etc.)        |
| **Deck**          | Extracted from the Cabin column                             |
| **AgeGroup**      | Categorized age into Child, Teen, Adult, Senior             |
| **FarePerPerson** | Ticket fare divided by family size                          |

These features help capture **social structure, family relationships, and passenger status**.

### Categorical Encoding

Categorical variables were encoded using **one-hot encoding**:

* Sex
* Embarked
* Title
* Deck
* AgeGroup

### Feature Transformations

To improve data distribution:

* **Fare** was **log-transformed** to reduce skewness.

---

# Part 3: Feature Selection

Feature selection was performed to identify the most important predictors of survival.

### Correlation Analysis

Correlation analysis was used to detect redundant or highly correlated features.

### Feature Importance

A **Random Forest model** was trained to rank feature importance.

The most important features included:

* Age
* Fare
* FarePerPerson
* Sex
* Title
* Passenger Class
* FamilySize

These features provide strong predictive power for survival outcomes.

---

# Key Findings

Several insights were discovered during the analysis:

* **Gender was a strong predictor of survival**, with females more likely to survive.
* **Passenger class influenced survival probability**, with first-class passengers having higher survival rates.
* **Family structure mattered**, as passengers traveling alone had different survival patterns compared to families.
* **Ticket fare and social titles** helped capture socioeconomic status.

---

# How to Run the Project

### 1 Install dependencies

```
pip install -r requirements.txt
```

### 2 Run Data Cleaning

```
python scripts/data_cleaning.py
```

This produces:

```
data/train_cleaned.csv
```

### 3 Run Feature Engineering

```
python scripts/feature_engineering.py
```

This produces:

```
data/train_features.csv
```

### 4 Run Feature Selection

```
python scripts/feature_selection.py
```

This script displays the **most important features** using a Random Forest model.

---

# Notebook

Exploratory analysis and visualizations are available in:

```
notebooks/Titanic_Feature_Engineering.ipynb
```

---

# Requirements

Main Python libraries used:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

Install them using:


pip install -r requirements.txt

# Conclusion

Through systematic **data cleaning, feature engineering, and feature selection**, the Titanic dataset was transformed into a structured dataset suitable for predictive modeling. The engineered features and selected variables reveal key factors that influenced survival outcomes.
