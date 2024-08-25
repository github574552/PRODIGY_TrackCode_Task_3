import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

 
bank_data = pd.read_csv('bank.csv', sep=';')
print(bank_data.head())
 
bank_data = pd.get_dummies(bank_data, drop_first=True)
 

 
X = bank_data.drop('y_yes', axis=1)
y = bank_data['y_yes']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
 
plt.figure(figsize=(10, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()
