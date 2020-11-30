"""
@authors: Valentin Morel
          Iason Batzianoulis

Training two SVR modes with the recorded data to predict the gaze coordinates

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.externals 
import joblib
# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)

def removeZeros(df):
    idxNames = df[df['left_cx']==0].index
    df.drop(idxNames, inplace=True)
    idxNames = df[df['right_cx']==0].index
    df.drop(idxNames, inplace=True)
    idxNames = df[df['aruco_cx']==0].index
    df.drop(idxNames, inplace=True)
    return df

def mySVRfct(fnameRead):
    # warnings.filterwarnings("ignore", category=DeprecationWarning, module='sklearn.svm')

    data = pd.read_csv(fnameRead + '.csv')
 
    data = removeZeros(data)

    pupil_coord = data[['left_cx','right_cx', 'left_cy', 'right_cy']]

    target_world = data[['aruco_cx', 'aruco_cy']]

    
    # Stratify split of the dataset
    X_train, X_test, y_train, y_test = train_test_split(pupil_coord, target_world, test_size=0.2, random_state=1)
    print (X_train.shape , X_test.shape , y_train.shape , y_test.shape)

    
    X_train_cx = X_train[['left_cx','right_cx','left_cy','right_cy']].to_numpy()
    y_train_cx = y_train[['aruco_cx']].to_numpy()
    y_train_cx = y_train_cx.ravel()
    
    X_train_cy = X_train[['left_cx','right_cx','left_cy','right_cy']].to_numpy()
    y_train_cy = y_train[['aruco_cy']].to_numpy()
    y_train_cy = y_train_cy.ravel()
    
    X_test_cx = X_test[['left_cx','right_cx','left_cy','right_cy']]
    y_test_cx = y_test[['aruco_cx']].to_numpy()
    y_test_cx = y_test_cx.ravel()
    
    X_test_cy = X_test[['left_cx','right_cx','left_cy','right_cy']]
    y_test_cy = y_test[['aruco_cy']].to_numpy()
    y_test_cy = y_test_cy.ravel()
    
    # define SVR with rbf kernel
    mySVR = SVR(gamma="scale", kernel = 'rbf')

    # define parameters for the Gridsearch
    parameters_cx = {'C':[1150, 1000, 1100, 1500], 'epsilon':[0.5, 0.7, 0.22, 0.67]}
    parameters_cy = {'C':[1150, 1300, 1100, 1500], 'epsilon':[0.5, 0.7, 0.18, 0.67]}
    
    # Find best parameters for cx. 5-fold crossvalidation
    myclf_SVR_cx = GridSearchCV(mySVR, parameters_cx, cv=5)
    print('training SVR for the x-coordinate ...')
    myclf_SVR_cx.fit(X_train_cx, y_train_cx)

    # Find best parameters for cy. 5-fold crossvalidation
    myclf_SVR_cy = GridSearchCV(mySVR, parameters_cy, cv=5)
    print('training SVR for the y-coordinate ...')
    myclf_SVR_cy.fit(X_train_cy, y_train_cy) 
    
    # show the best estimator for cx    
    print(pd.DataFrame(myclf_SVR_cx.cv_results_)[['mean_test_score', 'std_test_score', 'params']])    
    print('Best SVR for cx: ',myclf_SVR_cx.best_estimator_)
    
    # show the best estimator for cy    
    print(pd.DataFrame(myclf_SVR_cy.cv_results_)[['mean_test_score', 'std_test_score', 'params']])
    print('Best SVR for cy: ',myclf_SVR_cy.best_estimator_)
    
    # preidtion on the TEST sets
    mypred_SVR_cx = myclf_SVR_cx.predict(X_test_cx)
    mypred_SVR_cy = myclf_SVR_cy.predict(X_test_cy)

    
    
    fig = plt.figure(figsize=(16,8))
    plt.rcParams.update({'font.size': 30})
    plt.scatter('aruco_cx','aruco_cy', data = y_test, c='blue',marker='o', label = "Ground truth", s=30) 
    plt.scatter(mypred_SVR_cx, mypred_SVR_cy, c='red',marker='o', label = "Prediction with rbf kernel", s=30, )
    plt.xlabel("px")
    plt.ylabel("py")
    plt.title("epsilon-Support Vector Regression")
    plt.legend()
    fig.savefig('gaze_model/prediction.png', dpi = 300)

    fig = plt.figure(figsize=(20,11))
    # plot_cx = y_test_cx.copy()
    # plot_cx['predictionX'] = mypred_SVR_cx
    # plot_cx.sort_index(inplace = True)
    plt.plot(y_test_cx, marker='D',c='b', label = "Ground truth", ms=10)
    plt.plot(mypred_SVR_cx, marker='o',c='r', label = "Prediction with rbf kernel", ms=10, linewidth = 2)
    plt.title("epsilon-Support Vector Regression for X coordinate", fontsize=30, fontweight="bold")
    plt.xlabel("data", fontsize=30)
    plt.ylabel("Coordinate px", fontsize=30)
    plt.legend(prop={'size': 30})
    plt.ylim([400,1400])
    plt.xlim([0,150])
    fig.savefig('gaze_model/predictionX.png', dpi = 300)

    
    
    fig = plt.figure(figsize=(20,11))
    # plot_cy = y_test_cy.copy()
    # plot_cy['predictionY'] = mypred_SVR_cy
    # plot_cy.sort_index(inplace = True)
    plt.plot(y_test_cy, marker='D',c='b', label = "Ground truth", ms=10)
    plt.plot(mypred_SVR_cy, marker='o',c='r', label = "Prediction with rbf kernel", ms=10, linewidth = 2)
    plt.title("epsilon-Support Vector Regression for Y coordinate", fontsize=30, fontweight="bold")
    plt.xlabel("data", fontsize=30)
    plt.ylabel("Coordinate py", fontsize=30)
    plt.legend(prop={'size': 30})
    plt.ylim([0,900])
    plt.xlim([0,150])
    fig.savefig('gaze_model/predictionY.png', dpi = 300)
    plt.show()
    
        
    print('MSE cx: ',mean_squared_error(y_test_cx, mypred_SVR_cx))
    print('MSE cy: ',mean_squared_error(y_test_cy, mypred_SVR_cy))
    
    # Best possible score is 1.0
    print('r2 cx: ',r2_score(y_test_cx, mypred_SVR_cx))
    print('r2 cy: ',r2_score(y_test_cy, mypred_SVR_cy))
    
    print('MAE cx: ',mean_absolute_error(y_test_cx, mypred_SVR_cx))
    print('MAE cy: ',mean_absolute_error(y_test_cy, mypred_SVR_cy))
    
    # Best possible score is 1.0
    print('Explained variance cx: ',explained_variance_score(y_test_cx, mypred_SVR_cx))
    print('Explained variance cy: ',explained_variance_score(y_test_cy, mypred_SVR_cy))
    print(type(myclf_SVR_cx))    

    # save the model 
    filename_cx = 'gaze_model/SVR_model_cx.sav'
    filename_cy = 'gaze_model/SVR_model_cy.sav'
    joblib.dump(myclf_SVR_cx, filename_cx)
    joblib.dump(myclf_SVR_cy, filename_cy)

    return(myclf_SVR_cx, myclf_SVR_cy)

if __name__ == '__main__':
    mySVRfct('data/coord')
    