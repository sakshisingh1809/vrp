import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from loadtrainingdata import loadexceldata


def normalizedata(df):
    # Declare feature vector and target variable
    y = df["S10"]
    X = df.drop(["S10"], axis=1)

    #  Split data into separate training and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=109
    )
    print("Train data shape: ", X_train.shape, y_train.shape)
    print("Test data shape: ", X_test.shape, y_test.shape)

    # Feature Scaling
    """ cols = X_train.columns
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])
    X_train.describe() """

    return X_train, X_test, y_train, y_test


def svm(k, c, X_train, y_train, X_test, y_test):
    svc = SVC(kernel=k, C=c)  # instantiate classifier with default hyperparameters
    svc.fit(X_train, y_train)  # fit classifier to training set
    y_pred = svc.predict(X_test)  # make predictions on test set
    acc = accuracy_score(y_test, y_pred)
    return y_pred, acc


def svmkfold(n, k, X, y):
    kfold = KFold(n_splits=n, shuffle=True, random_state=0)
    svc = SVC(kernel=k)
    scores = cross_val_score(svc, X, y, cv=kfold)
    return scores


"""def confusion_matrix(y_test, y_pred):
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion matrix\n")
    print("True Positives(TP) = ", cm[0, 0])
    print("True Negatives(TN) = ", cm[1, 1])
    print("False Positives(FP) = ", cm[0, 1])
    print("False Negatives(FN) = ", cm[1, 0])

    cm_matrix = pd.DataFrame(
        data=cm,
        columns=["Actual Positive:1", "Actual Negative:0"],
        index=["Predict Positive:1", "Predict Negative:0"],
    ) 



def classification_report(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0, 0]
    TN = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    print("Classification accuracy : {0:0.4f}".format(classification_accuracy))
    print("Classification error : {0:0.4f}".format(classification_error))
    print("Precision : {0:0.4f}".format(precision))
    print("Recall or Sensitivity : {0:0.4f}".format(recall))
 """


def svmmodel():
    path = "C:/Users/sakshi.singh/OneDrive - LichtBlick SE/Documents/VRP + GA + ML/vrp/vrp/featuresdata/"
    filename = "SolutionsGA.xlsx"
    df = loadexceldata(path, filename)
    solutions = df.drop(["name", "S2"], axis=1)  # dropping string columns
    X_train, X_test, y_train, y_test = normalizedata(solutions)
    null_acc = y_test.value_counts()
    print("Null accuracy: {0:0.4f}".format((null_acc[0] / (null_acc[0] + null_acc[1]))))

    y_pred1, acc1 = svm("rbf", 1, X_train, y_train, X_test, y_test)
    print("Accuracy with default hyperparameters: {0:0.4f}".format(acc1))

    y_pred2, acc2 = svm("linear", 100, X_train, y_train, X_test, y_test)
    print("Accuracy score with linear kernel: {0:0.4f}".format(acc2))

    y = solutions["S10"]
    X = solutions.drop(["S10"], axis=1)

    scorerbf = svmkfold(5, "rbf", X, y)
    scores = svmkfold(5, "linear", X, y)
    # print("Stratified cross-validation scores with linear kernel:\n\n{}".format(scores))
    print(
        "Average Stratified cross-validation scores with linear kernel:{:.4f}".format(
            scores.mean()
        )
    )
    print(
        "Average stratified cross-validation score with rbf kernel:{:.4f}".format(
            scorerbf.mean()
        )
    )

    # y_pred3, acc3 = svm("poly", 1, X_train, y_train, X_test, y_test)
    # print("Accuracy score with polynomial kernel: {0:0.4f}".format(acc3))

    # y_pred4, acc4 = svm("sigmoid", 1, X_train, y_train, X_test, y_test)
    # print("Accuracy score with sigmoid kernel: {0:0.4f}".format(acc4))

    # since linear kernel gives the best accuracy, we will explore it further
    # confusion_matrix(y_test, y_pred2)
    # classification_report(y_test, y_pred2)


svmmodel()
