import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dados = pd.read_csv("Iris.csv")

X = dados.drop(columns=["Id", "Species"])
y = dados["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"✅ Acurácia do modelo: {acuracia*100:.2f}%\n")

print("Relatório de Classificação (por espécie):")
print(classification_report(y_test, y_pred, target_names=modelo.classes_))

cm = confusion_matrix(y_test, y_pred, labels=modelo.classes_)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",  
            xticklabels=modelo.classes_,
            yticklabels=modelo.classes_)
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.show()
