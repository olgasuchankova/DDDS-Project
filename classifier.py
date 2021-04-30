import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix  # , plot_confusion_matrix
from sklearn import svm
from sklearn import linear_model as lm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import pickle


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


def evaluatePerformance(X_train, X_test, y_train, y_test, gridsearchmodel,
                        classifiername,cvscores=True,CFM=True):

    # best parameters
    print(gridsearchmodel.best_params_)

    # y_prob = gridsearchmodel.predict_proba(X_test)
    y_pred = gridsearchmodel.predict(X_test)

    # accuracy
    train_score = gridsearchmodel.score(X_train, y_train)
    test_score = gridsearchmodel.score(X_test, y_test)
    print(classifiername, "Training Accuracy:", train_score)
    print(classifiername, "Test Accuracy:", test_score)

    if cvscores == True:
        scores = cross_val_score(gridsearchmodel, X_train, y_train, cv=5, scoring='accuracy')
        print(classifiername, "Mean Accuracy: {:f}".format(np.mean(scores)))
        print(classifiername, "Stdev of Accuracy: {:f}".format(np.std(scores)))

    if CFM == True:
        # confusion matrix
        disp = plot_confusion_matrix(model,X_test,y_test,cmap=plt.cm.Blues,normalize='true')
        disp.ax_.set_title("Normalized confusion matrix")
        plt.show()

    return train_score, test_score, y_pred, gridsearchmodel

# if __name__ == '__main__':
#     instrumfeats_main = np.load('data_file.npy')
#     targets_main = np.load('targets.npy')
#     classifyAudio(instrumfeats_main,targets_main,num_feats=22)
# runAllClassifiers(instrumfeats_main,targets_main)#,cvscores=True,CFM=True)