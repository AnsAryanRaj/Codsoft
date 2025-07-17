#Task 4
#Sales Prediction Using Python--

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


df = pd.read_csv("advertising.csv")
print("Dataset Loaded:\n", df.head())


print("\nDataset Info:")
print(df.describe())


sns.set(style="whitegrid")


df.hist(figsize=(10, 6), edgecolor='black')
plt.suptitle("Histogram of Features", fontsize=16)
plt.tight_layout()
plt.show()


sns.pairplot(df, height=3)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()


plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


features = ['TV', 'Radio', 'Newspaper']
for feature in features:
    sns.lmplot(data=df, x=feature, y='Sales', height=5, aspect=1.2, line_kws={'color': 'red'})
    plt.title(f"Sales vs {feature}")
    plt.show()


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=20, color='purple')
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
