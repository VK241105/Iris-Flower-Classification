# Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load the Iris dataset
iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

# Convert numerical labels to flower names
y = y.map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Combine features and target for exploration
df = X.copy()
df["species"] = y

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nClass distribution:")
print(df["species"].value_counts())


# 2. Visualize the dataset
sns.pairplot(df, hue="species")
plt.suptitle("Iris Flower Dataset", y=1.02)
plt.show()


# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Train K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train_scaled, y_train)


# 6. Make predictions
y_pred = model.predict(X_test_scaled)


# 7. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Setosa", "Versicolor", "Virginica"],
    yticklabels=["Setosa", "Versicolor", "Virginica"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 9. Test with a new flower
new_flower = [[
    5.1,  # sepal length
    3.5,  # sepal width
    1.4,  # petal length
    0.2   # petal width
]]

new_flower_scaled = scaler.transform(new_flower)

prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction:", prediction[0])