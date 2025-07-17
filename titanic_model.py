#Task 1
#Titanic Survival Prediction--

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv("train.csv")  # Ensure train.csv is in your project directory


df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


df.fillna({
    'Age': df['Age'].mean(),
    'Embarked': df['Embarked'].mode()[0]
}, inplace=True)


df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 4))
survived_counts = df['Survived'].value_counts().sort_index()
plt.bar(['Not Survived', 'Survived'], survived_counts, color=['lightcoral', 'lightgreen'])
plt.title("Survival Counts")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


X = df.drop('Survived', axis=1)
y = df['Survived']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

#
y_pred = model.predict(X_test_scaled)


print("Classification Report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


coefficients = model.coef_[0]
feature_names = X.columns
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients}).sort_values(by='Coefficient')

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df, color='skyblue')
plt.title("Feature Importance (Logistic Regression Coefficients)")
plt.tight_layout()
plt.show()


