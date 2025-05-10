import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("/content/healthcare_dataset.csv.zip") 
print("Dataset Loaded Successfully.\n")


print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)


for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\nDescriptive Statistics:")
print(df.describe())


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

if 'target' in df.columns:
    print("\nRunning machine learning classification...")

    X = df.drop('target', axis=1)
    y = df['target']

    
    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("\nNo 'target' column found. Skipping machine learning section.")

print("First 5 rows:")
print(df.head())


print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())

df.fillna(df.median(numeric_only=True), inplace=True)


for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


print("\nDescriptive Statistics:")
print(df.describe())


plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

if 'target' in df.columns:
    print("\nRunning machine learning classification...")

    X = df.drop('target', axis=1)
    y = df['target']


    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
else:
    print("\nNo 'target' column found. Skipping machine learning section.")