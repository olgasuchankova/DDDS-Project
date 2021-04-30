import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix


def classify(X_train, X_test, y_train, y_test, classifierName="SVM",do_pca=False):    
    # create model using different classifiers
    # K-Nearest Neighbors
    if classifierName == "KNN":
        gsmodel = knnClassify()
    # Support Vector Machine
    if classifierName == "SVM":
        gsmodel = svmClassify(do_pca)
    # Logisitc Regression
    if classifierName == "LR":
        gsmodel = LRClassify()

    gsmodel.fit(X_train, y_train)

    return evaluatePerformance(X_train, X_test, y_train, y_test, gsmodel, classifierName)


def knnClassify(do_pca):
    # Train the KNN classifier
    # https://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    if do_pca == False:
       knn_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
       knn_param_grid = [{'kneighborsclassifier__n_neighbors': [2,3,4,5]}] 
    else:
       pca = PCA(0.98, svd_solver='full')
       knn_pipe = make_pipeline(StandardScaler(),pca, KNeighborsClassifier(n_neighbors=5))
       knn_param_grid = [{'kneighborsclassifier__n_neighbors': [2,3,4,5],
                        'pca__n_components': [2,3,4,5,6]}]

    gs_knn = GridSearchCV(estimator=knn_pipe,
                          param_grid=knn_param_grid,
                          scoring='accuracy',
                          refit=True,
                          cv=5, verbose=0)

    return gs_knn


def svmClassify(do_pca):
    # Train the SVM classifier
    if do_pca == False:
        svm_pipe = make_pipeline(StandardScaler(), SVC(probability=True))
        svm_param_grid = [{'svc__C': [0.1, 1, 10],
                       'svc__kernel': ['poly', 'rbf', 'sigmoid', 'linear']}]
    else:
        pca = PCA(0.98, svd_solver='full')
        svm_pipe = make_pipeline(StandardScaler(),pca, SVC(probability=True))
        svm_param_grid = [{'svc__C': [0.1, 1, 10],
                       'svc__kernel': ['poly', 'rbf', 'sigmoid', 'linear'],
                       'pca__n_components': [2,3,4,5,6]}]

    gs_SVM = GridSearchCV(estimator=svm_pipe,
                          param_grid=svm_param_grid,
                          scoring='accuracy',
                          n_jobs=-1,
                          refit=True,
                          cv=3, verbose=0)

    return gs_SVM


def LRClassify(do_pca):
    # Train Logistic Regression classifier
    if do_pca == False:
        lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        lr_param_grid = [{'logisticregression__C': [1e-2, 1e-1, 1, 2],
                      'logisticregression__solver': ['liblinear', 'newton-cg', 'lbfgs']}]
    else:
        pca = PCA(0.98, svd_solver='full')
        lr_pipe = make_pipeline(StandardScaler(),pca, LogisticRegression(max_iter=1000))
        lr_param_grid = [{'logisticregression__C': [1e-2, 1e-1, 1, 2],
                      'logisticregression__solver': ['liblinear', 'newton-cg', 'lbfgs'],
                      'pca__n_components': [2,3,4,5,6]}]

    gs_LR = GridSearchCV(estimator=lr_pipe,
                         param_grid=lr_param_grid,
                         scoring='accuracy',
                         refit=True,
                         n_jobs=-1,
                         cv=3, verbose=0)

    return gs_LR


def evaluatePerformance(X_train, X_test, y_train, y_test, gridsearchmodel, classifiername,
                        CM = True, CV = True):

    # best parameters
    print(gridsearchmodel.best_params_)

    # y_prob = gridsearchmodel.predict_proba(X_test)
    y_pred = gridsearchmodel.predict(X_test)

    # accuracy
    train_score = gridsearchmodel.score(X_train, y_train)
    test_score = gridsearchmodel.score(X_test, y_test)
    print(classifiername, "Training Accuracy:", train_score)
    print(classifiername, "Test Accuracy:", test_score)
<<<<<<< HEAD
    
    if CM == True:
        get_confusion_matrix(X_test, y_test, gridsearchmodel)
        
    if CV == True:
        get_cv_scores(X_train, y_train, gridsearchmodel,classifiername)
        
=======

    # if cvscores == True:
    #     scores = cross_val_score(gridsearchmodel, X_train, y_train, cv=5, scoring='accuracy')
    #     print(classifiername, "Mean Accuracy: {:f}".format(np.mean(scores)))
    #     print(classifiername, "Stdev of Accuracy: {:f}".format(np.std(scores)))
    #
    # if CFM == True:
    #     # confusion matrix
    #     disp = plot_confusion_matrix(model,X_test,y_test,cmap=plt.cm.Blues,normalize='true')
    #     disp.ax_.set_title("Normalized confusion matrix")
    #     plt.show()

>>>>>>> 77708cf7c8a399faf890dc91a06df1592ed523aa
    return train_score, test_score, y_pred, gridsearchmodel

def get_confusion_matrix(X_test, y_test, gridsearchmodel):
    disp = plot_confusion_matrix(gridsearchmodel,X_test,y_test,cmap=plt.cm.Blues,normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    plt.show()
    
def get_cv_scores(X_train, y_train, gridsearchmodel,classifiername):
    scores = cross_val_score(gridsearchmodel, X_train, y_train, cv=5, scoring='accuracy')
    print(" ")
    print(classifiername, "Mean Accuracy: {:f}".format(np.mean(scores)))
    print(classifiername, "Stdev of Accuracy: {:f}".format(np.std(scores)))
    
