import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.describe())
#print(df.isnull().sum())

species_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

df["species"] = df["species"].map(species_names)

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species"
)

plt.title("Petal Length vs Petal Width")
#plt.show()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual:", y_test)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()
#plt.show()

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_accuracy * 100, "%")

scaled_knn = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=3)
)

scaled_knn.fit(X_train, y_train)

scaled_pred = scaled_knn.predict(X_test)

scaled_accuracy = accuracy_score(y_test, scaled_pred)

print("Scaled KNN Accuracy:", scaled_accuracy * 100, "%")

models = {
    "Logistic Regression": model,
    "Original KNN": knn_model,
    "Scaled KNN": scaled_knn
}

for name, classifier in models.items():
    scores = cross_val_score(
        classifier,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print(name, "scores:", scores)
    print("Mean accuracy:", scores.mean() * 100, "%")