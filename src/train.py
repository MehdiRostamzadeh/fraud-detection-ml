from data_prep import prepare_data
import pandas as pd
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
logistic = LogisticRegression()

# logistic.fit(X_train,y_train)
# y_pred_logistic = logistic.predict(X_test)

# cm_logistic = confusion_matrix(y_test , y_pred_logistic)
# tn, fp, fn, tp = confusion_matrix(y_test,y_pred_logistic).ravel().tolist()
# print(cm_logistic)
# print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)

# print("precision logistic =",precision_score(y_test, y_pred_logistic))
# print("recall logistic =",recall_score(y_test, y_pred_logistic))
# print("f1 logistic =",f1_score(y_test, y_pred_logistic))

#____________________________________________________________________________
print("-"*100)

print("KNN model")
knn = KNeighborsClassifier(n_neighbors=5)

# knn.fit(X_train,y_train)
# y_pred_knn = knn.predict(X_test)

# cm_knn = confusion_matrix(y_test,y_pred_knn)
# tn, fp, fn, tp = confusion_matrix(y_test,y_pred_knn).ravel().tolist()
# print(cm_knn)
# print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


# print("precision KNN =",precision_score(y_test, y_pred_knn))
# print("recall KNN =",recall_score(y_test, y_pred_knn))
# print("f1 KNN =",f1_score(y_test, y_pred_knn))

#_____________________________________________________________
print("-"*100)

print("Decision tree model")
tree = DecisionTreeClassifier(max_depth=5,
                              random_state=42)

# tree.fit(X_train,y_train)
# y_pred_tree = tree.predict(X_test)

# cm_tree = confusion_matrix(y_test,y_pred_tree)

# tn, fp, fn, tp = confusion_matrix(y_test,y_pred_tree).ravel().tolist()
# print(cm_tree)
# print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


# print("precision Decision tree =",precision_score(y_test, y_pred_tree))
# print("recall Decision tree =",recall_score(y_test, y_pred_tree))
# print("f1 Decision tree =",f1_score(y_test, y_pred_tree))

#____________________________________________________________________________________
print("-"*100)


print("======> Cross validation <======")

from sklearn.model_selection import StratifiedKFold,cross_validate

cv = StratifiedKFold(n_splits=5,
                     shuffle=True,
                     random_state=42)

logistic_cv =cross_validate(logistic,
                            X_train,
                            y_train,
                            cv = cv,
                            scoring=["precision","recall","f1"])

print("Logostic Regression")
print("Precision:", logistic_cv["test_precision"])
print("Recall:", logistic_cv["test_recall"])
print("F1:", logistic_cv["test_f1"])

# print("Mean Precision:", logistic_cv["test_precision"].mean())
# print("Mean Recall:", logistic_cv["test_recall"].mean())
# print("Mean F1:", logistic_cv["test_f1"].mean())


Mean_Precision_logistic = logistic_cv["test_precision"].mean()
Mean_Recall_logistic = logistic_cv["test_recall"].mean()
Mean_F1_logistic = logistic_cv["test_f1"].mean()

print("Mean Precision logistic :", Mean_Precision_logistic)
print("Mean Recall logistic :", Mean_Recall_logistic)
print("Mean F1 logistic :", Mean_F1_logistic)

#________________________________________________________________________
print("-"*100)


knn_cv = cross_validate(knn,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=["precision", "recall", "f1"]
                        )

print("KNN")
Mean_Precision_knn= knn_cv["test_precision"].mean()
Mean_Recall_knn = knn_cv["test_recall"].mean()
Mean_F1_knn = knn_cv["test_f1"].mean()

print("Mean Precision KNN :", Mean_Precision_knn)
print("Mean Recall KNN :", Mean_Recall_knn)
print("Mean F1 KNN :", Mean_F1_knn)
#__________________________________________________________________________
print("-"*100)

tree_cv = cross_validate(tree,
                         X_train,
                         y_train,
                         cv=cv,
                         scoring=["precision", "recall", "f1"]
                         )


print("Decision Tree")
Mean_Precision_tree = tree_cv["test_precision"].mean()
Mean_Recall_tree = tree_cv["test_recall"].mean()
Mean_F1_tree = tree_cv["test_f1"].mean()


print("Mean Precision Decision Tree :", Mean_Precision_tree)
print("Mean Recall Decision Tree :", Mean_Recall_tree)
print("Mean F1 Decision Tree :", Mean_F1_tree)

#____________________________________________________________________________

print("***** Report Result *****")
report_result_cv = pd.DataFrame({
    "Model :" : ["Logistic Regression","KNN","Decision Tree"],
    "Mean Precision" :[Mean_Precision_logistic,Mean_Precision_knn,Mean_Precision_tree],
    "Mean Recall" : [Mean_Recall_logistic,Mean_Recall_knn,Mean_Recall_tree],
    "Mean F1" : [Mean_F1_logistic,Mean_F1_knn,Mean_F1_tree]
})

print(report_result_cv)