from data_prep import prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix,
                             precision_score,
                             recall_score,
                             f1_score)


## model_1 : LogisticRegression
## Model 2: K-Nearest Neighbor (KNN)
## Model 3: Decision Tree Classifier

X_train, X_test, y_train, y_test, scaler = prepare_data()

print("Logistic regression model ")
model_1 = LogisticRegression()
model_1.fit(X_train,y_train)
y_pred_1 = model_1.predict(X_test)

cm_1 = confusion_matrix(y_test , y_pred_1)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_1).ravel().tolist()
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)
print(cm_1)

print("precision logistic =",precision_score(y_test, y_pred_1))
print("recall logistic =",recall_score(y_test, y_pred_1))
print("f1 logistic =",f1_score(y_test, y_pred_1))

#_____________________________________________________________
print("-"*100)

print("KNN model")
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred_knn).ravel().tolist()
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


print("precision KNN =",precision_score(y_test, y_pred_knn))
print("recall KNN =",recall_score(y_test, y_pred_knn))
print("f1 KNN =",f1_score(y_test, y_pred_knn))

#_____________________________________________________________
print("-"*100)

print("Decision tree model")
tree = DecisionTreeClassifier(max_depth=5,
                              random_state=42)

tree.fit(X_train,y_train)
y_pred_tree = tree.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred_tree).ravel().tolist()
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


print("precision Decision tree =",precision_score(y_test, y_pred_tree))
print("recall Decision tree =",recall_score(y_test, y_pred_tree))
print("f1 Decision tree =",f1_score(y_test, y_pred_tree))

