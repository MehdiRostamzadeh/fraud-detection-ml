import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data_prep import prepare_data

## model_1 : LogisticRegression
## Model 2: K-Nearest Neighbor (KNN)
## Model 3: Decision Tree Classifier

X_train, X_test, y_train, y_test, scaler , X_train_raw , X_test_raw = prepare_data()

print("Logistic regression model ")
logistic = LogisticRegression()

logistic.fit(X_train,y_train)
y_pred_logistic = logistic.predict(X_test)

cm_logistic = confusion_matrix(y_test , y_pred_logistic)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_logistic).ravel().tolist()
print(cm_logistic)
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)

print("precision logistic =",precision_score(y_test, y_pred_logistic))
print("recall logistic =",recall_score(y_test, y_pred_logistic))
print("f1 logistic =",f1_score(y_test, y_pred_logistic))

#____________________________________________________________________________
print("-"*100)

print("KNN model")
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

cm_knn = confusion_matrix(y_test,y_pred_knn)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_knn).ravel().tolist()
print(cm_knn)
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


precision_knn = precision_score(y_test, y_pred_knn)
recall_knn =recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)


print("precision KNN : ", precision_knn)
print("Recall KNN : ", recall_knn)
print("F1 KNN : ", f1_knn)


#_____________________________________________________________
print("-"*100)

print("Decision tree model")
tree = DecisionTreeClassifier(max_depth=5,
                              random_state=42)

tree.fit(X_train,y_train)
y_pred_tree = tree.predict(X_test)

cm_tree = confusion_matrix(y_test,y_pred_tree)

tn, fp, fn, tp = confusion_matrix(y_test,y_pred_tree).ravel().tolist()
print(cm_tree)
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)


print("precision Decision tree =",precision_score(y_test, y_pred_tree))
print("recall Decision tree =",recall_score(y_test, y_pred_tree))
print("f1 Decision tree =",f1_score(y_test, y_pred_tree))

# #____________________________________________________________________________________
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


print("***** Report Result *****")
report_result_cv = pd.DataFrame({
    "Model :" : ["Logistic Regression","KNN","Decision Tree"],
    "Mean Precision" :[Mean_Precision_logistic,Mean_Precision_knn,Mean_Precision_tree],
    "Mean Recall" : [Mean_Recall_logistic,Mean_Recall_knn,Mean_Recall_tree],
    "Mean F1" : [Mean_F1_logistic,Mean_F1_knn,Mean_F1_tree]
})

print(report_result_cv)

#___________________________________________________________________________________________
print("-"*100)

# Experiment 1 : KNN

print("KNN Unscaled")

knn_unscaled = KNeighborsClassifier(n_neighbors=5)

knn_unscaled.fit(X_train_raw,y_train)
y_pred_knn_unscaled = knn_unscaled.predict(X_test_raw)


precision_knn_unscaled = precision_score(y_test, y_pred_knn_unscaled)
recall_knn_unscaled =recall_score(y_test, y_pred_knn_unscaled)
f1_knn_unscaled = f1_score(y_test, y_pred_knn_unscaled)

cm_knn_unscaled = confusion_matrix(y_test,y_pred_knn_unscaled)
tn, fp, fn, tp = confusion_matrix(y_test,y_pred_knn_unscaled).ravel().tolist()
print(cm_knn_unscaled)
print("TN =",tn,"FP =",fp,"FN =",fn,"TP =", tp)

print("precision KNN unscaled : ", precision_knn_unscaled)
print("Recall KNN unscaled: ", recall_knn_unscaled)
print("F1 KNN unscaled : ", f1_knn_unscaled)


print("***** KNN scaled Vs Unscaled *****")
report_knn = pd.DataFrame({
    "Model :" : ["Without Scaling","With Scaling"],
    "Precision" :[precision_knn_unscaled,precision_knn],
    "Recall" : [recall_knn_unscaled,recall_knn],
    "F1" : [f1_knn_unscaled,f1_knn]
})

print(report_knn)

#____________________________________________________________________________
print("-"*100)

# Experiment 2 : Hyperparameter

#Option 1 :KNN

print(" ====== KNN Hyper ====== ")
k_value = [1 ,5, 20 ]

for k in k_value:
    knn_EX = KNeighborsClassifier(n_neighbors=k)
    knn_EX.fit(X_train,y_train)
    y_pred_knn_EX = knn_EX.predict(X_test)
    
    precision_knn_EX = precision_score(y_test, y_pred_knn_EX)
    recall_knn_EX =recall_score(y_test, y_pred_knn_EX)
    f1_knn_EX = f1_score(y_test, y_pred_knn_EX)

    print(f"precision KNN {k} :  {precision_knn_EX}",)
    print(f"Recall KNN {k} :  {recall_knn_EX}")
    print(f"F1 KNN  {k} : {f1_knn_EX}")
    print("-"*40)
    
    
#_____________________________________________________________________________
print("-"*100)
# Option 2 : Decison Tree

print(" ====== Decision Tree Hyper ====== ")
depth_value = [None ,2 ,5 ,10]

for depth in depth_value:
    tree_EX = DecisionTreeClassifier(max_depth=depth,
                                random_state=42)

    tree_EX.fit(X_train,y_train)
    y_pred_tree_EX = tree_EX.predict(X_test)

    precision_tree_EX = precision_score(y_test, y_pred_tree_EX)
    recall_tree_EX = recall_score(y_test, y_pred_tree_EX)
    f1_tree_EX = f1_score(y_test, y_pred_tree_EX)
    
    print(f"Max depth ==> {depth}")
    print(f"precision Decision Tree :  {precision_tree_EX}",)
    print(f"Recall Decision Tree  :  {recall_tree_EX}")
    print(f"F1 Decision Tree : {f1_tree_EX}")
    print("-"*40)
    
#______________________________________________________________________________
print("-"*100)  

# Experiment 3 : Threshold
    
y_prob = logistic.predict_proba(X_test)[:, 1]
    
thresholds =[0.3 , 0.5 , 0.7]
    
for threshold in thresholds :
        
    y_pred_threshold = (y_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    f1 = f1_score(y_test, y_pred_threshold)

    print(f"Threshold : {threshold}")
    print(f"Precision :{precision}")
    print(f"Recall : {recall}")
    print(f"F1 : {f1}")
    print("-"*40)
    
# threshold_report = pd.DataFrame({
#     "Threshold" : [ 0.3 , 0.5 , 0,7],
    
# })
# print(threshold_report)

#________________________________________________________________________________

# model Selection 

import joblib

joblib.dump(knn, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")