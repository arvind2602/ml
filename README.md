# 🧊 Titanic Dataset Analysis and Preprocessing

## 📌 Objective

By the end of this notebook, you will be able to:

* Load and inspect a dataset
* Handle missing and categorical data
* Engineer meaningful features
* Visualize relationships using plots
* Prepare data for machine learning

---

## 1️⃣ Loading the Dataset

> ✅ **Goal**: Import the Titanic dataset using `pandas`. If the URL fails, fall back to Seaborn’s built-in Titanic dataset.

```python
import pandas as pd

# Try loading from GitHub
try:
    df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}")
    
    # Fallback option
    !pip install seaborn -q
    import seaborn as sns
    df = sns.load_dataset('titanic')
    print("✅ Loaded dataset from seaborn as fallback.")
```

---

## 2️⃣ Initial Data Exploration

> ✅ **Goal**: Get an overview of the dataset—its structure, basic statistics, and data types.

### 🔍 Step 1: Peek at the data

```python
print(df.head())  # First 5 rows
```

### 📃 Step 2: Data summary

```python
df.info()  # Column types and null counts
```

### 📊 Step 3: Descriptive statistics

```python
df.describe()  # Only numeric columns
```

---

## 🧠 Understanding the Features

| Column      | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| PassengerId | Unique ID for each passenger (drop later)                            |
| Survived    | Target variable (0 = No, 1 = Yes)                                    |
| Pclass      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                             |
| Name        | Full name (used for title extraction)                                |
| Sex         | Gender                                                               |
| Age         | Age in years                                                         |
| SibSp       | # of siblings/spouses aboard                                         |
| Parch       | # of parents/children aboard                                         |
| Ticket      | Ticket number (not very useful)                                      |
| Fare        | Fare paid                                                            |
| Cabin       | Cabin number (mostly missing)                                        |
| Embarked    | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 3️⃣ Handling Missing Values

> ✅ **Goal**: Detect and fix missing values.

### 🔍 Step 1: Check for missing values

```python
df.isnull().sum()
```

### 🛠️ Step 2: Apply strategies

| Column   | Action                         |
| -------- | ------------------------------ |
| Cabin    | Drop (too many missing values) |
| Age      | Fill with median (numeric)     |
| Embarked | Fill with mode (categorical)   |

```python
df.drop('Cabin', axis=1, inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

### ✅ Step 3: Recheck missing values

```python
df.isnull().sum()
```

---

## 4️⃣ Converting Categorical Data

> ✅ **Goal**: Convert non-numeric columns into numeric format using one-hot encoding.

### 🧮 One-hot encode: `Sex` and `Embarked`

```python
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
```

### 🧮 One-hot encode: `Pclass`

```python
df = pd.get_dummies(df, columns=['Pclass'], prefix='Pclass', drop_first=True)
```

---

## 5️⃣ Feature Engineering

> ✅ **Goal**: Create new features to reveal hidden patterns.

---

### 🎩 Extract Titles from Names

> Many names contain titles (e.g., Mr, Mrs). These are informative!

```python
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

#### 🎯 Group Rare Titles

```python
df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
```

#### 🔢 Encode Titles

```python
df = pd.get_dummies(df, columns=['Title'], prefix='Title', drop_first=True)
df.drop('Name', axis=1, inplace=True)
```

---

### 👨‍👩‍👧 Create Family Size Feature

```python
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df.drop(['SibSp', 'Parch', 'FamilySize'], axis=1, inplace=True)
```

---

## 6️⃣ Final Preprocessed Dataset

> ✅ **Goal**: Review your cleaned and feature-engineered dataset.

```python
print(df.head())
df.info()
```

---

## 7️⃣ Data Visualization

> ✅ **Goal**: Use plots to gain insight from your dataset.

### 🧰 Install and import libraries

```python
!pip install matplotlib seaborn -q
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
```

---

### 📊 1. Survival Count

```python
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()
```

---

### 📊 2. Survival by Gender

```python
sns.countplot(x='Sex_male', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()
```

---

### 📊 3. Survival by Passenger Class

```python
temp_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')[['Pclass', 'Survived']]
sns.barplot(x='Pclass', y='Survived', data=temp_df, errorbar=None)
plt.title('Survival Rate by Class')
plt.show()
```

---

### 📊 4. Age Distribution by Survival

```python
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival')
plt.show()
```

---

### 📊 5. Fare Distribution by Survival

```python
sns.histplot(data=df, x='Fare', hue='Survived', kde=True, bins=30)
plt.title('Fare Distribution by Survival')
plt.xlim(0, 300)
plt.show()
```

---

### 📊 6. Survival by Embarked Port

```python
temp_df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')[['Embarked', 'Survived']].dropna()
sns.barplot(x='Embarked', y='Survived', data=temp_df, errorbar=None)
plt.title('Survival by Embarkation Port')
plt.show()
```

---

### 📊 7. Survival by Alone Status

```python
sns.barplot(x='IsAlone', y='Survived', data=df, errorbar=None)
plt.title('Survival by Alone Status')
plt.xticks([0, 1], ['Not Alone', 'Alone'])
plt.show()
```

---

### 📊 8. Correlation Heatmap

```python
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True)[['Survived']].sort_values(by='Survived', ascending=False), 
            annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Feature Correlation with Survival')
plt.show()
```

---
