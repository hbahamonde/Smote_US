# svm.py
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
import random
from numpy import mean
from statistics import pstdev
import matplotlib.pyplot as plt

# for classification problems
from sklearn import svm 
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler # scaling data
from collections import Counter

from imblearn.over_sampling import SMOTE

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn import metrics

data = pd.read_csv('./dataMergedConjoint.csv')
data = data.drop_duplicates(subset=['idnum'])

X = data[['woman', 'partyid', 'reg', 'trustfed', 'income.n','educ.n', 'polknow','vote.selling']]
#X = data[['trustfed','educ.n', 'polknow']]
y = 1*(data['socideo']<=2)
#concatenate?

#Standarization
xvalue = X.values
min_max_scaler = preprocessing.MinMaxScaler()
xscaled = min_max_scaler.fit_transform(xvalue)
X = pd.DataFrame(xscaled)

#training_pairs = pd.concat([X,y],axis=1)
nobs = len(X)
nfold = 3
lsets = int(np.ceil(nobs/nfold))
nrepeat = 13

Repeat = [np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
          np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
          np.zeros([nrepeat,nfold])]
PPV = [np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
       np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
       np.zeros([nrepeat,nfold])]
Accuracy = [np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
            np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
            np.zeros([nrepeat,nfold])]
AUC = [np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
       np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
       np.zeros([nrepeat,nfold])]
fScore = [np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
          np.zeros([nrepeat,nfold]),np.zeros([nrepeat,nfold]),
          np.zeros([nrepeat,nfold])]

for i in range(nrepeat):
    indexes = list(range(nobs))
    random.Random(i).shuffle(indexes)
    dfs = np.array_split(indexes,nfold)
    for j in range(nfold):
        
        index_bad = X.index.isin(dfs[j])
        X_test = X[index_bad]
        y_test = y[index_bad].ravel()
        X_train = X[~index_bad]
        y_train = y[~index_bad]
        #SMOTE
        oversample = SMOTE(k_neighbors=5,random_state=0)
        X_train,y_train = oversample.fit_resample(X_train,y_train)

        #clf = CalibratedClassifierCV(svm.SVC(kernel='linear',random_state=0))
        #clf = RandomForestClassifier(max_depth=2, random_state=0)
        #clf = DecisionTreeClassifier(random_state=0)
        #clf = MLPClassifier(solver='sgd',max_iter=1000,alpha=1e-5,hidden_layer_sizes=(50,50,), random_state=1)
        #clf = svm.SVC(kernel='rbf',random_state=0, probability=True)
        #clf = GaussianNB()
        clf = [svm.SVC(kernel='linear',random_state=0, probability=True),
               svm.SVC(kernel='rbf',random_state=0, probability=True),
               MLPClassifier(solver='sgd',max_iter=1000,alpha=1e-5,hidden_layer_sizes=(50,50,), random_state=1),
               GaussianNB()]
        for k in range(len(clf)):
            clf[k].fit(X_train, y_train)
            y_pred = clf[k].predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
            print(str(i)+str(j))
            Accuracy[k][(i,j)] = accuracy_score(y_test,y_pred, normalize=True)
            Repeat[k][(i,j)] = tp/(tp+fn)
            PPV[k][(i,j)] = tp/(tp+fp)
            fScore[k][(i,j)] = 2/(1/PPV[k][(i,j)]+1/Repeat[k][(i,j)])

            y_pred_proba = clf[k].predict_proba(X_test)[::,1]
            fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
            AUC[k][(i,j)] =metrics.roc_auc_score(y_test, y_pred_proba)

        # Baseline coin        
        y_pred = np.random.randint(0, 2, (len(y_test), 1)).ravel()
        tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
        Accuracy[4][(i,j)] = accuracy_score(y_test,y_pred, normalize=True)
        Repeat[4][(i,j)] = tp/(tp+fn)
        PPV[4][(i,j)] = tp/(tp+fp)
        fScore[4][(i,j)] = 2/(1/PPV[4][(i,j)]+1/Repeat[4][(i,j)])
        
file1 = open("tablaR_machinelearning_models"+str(nfold)+".txt","w+")
nfstr = repr(nfold)
medias = ["{:.2f}".format(mean(Repeat[0].ravel())),"{:.2f}".format(mean(Repeat[1].ravel())),
          "{:.2f}".format(mean(Repeat[2].ravel())),"{:.2f}".format(mean(Repeat[3].ravel()))]
stdevs = ["{:.2f}".format(pstdev(Repeat[0].ravel())),"{:.2f}".format(pstdev(Repeat[1].ravel())),
          "{:.2f}".format(pstdev(Repeat[2].ravel())),"{:.2f}".format(pstdev(Repeat[3].ravel()))]
file1.write(nfstr+' & '+medias[0]+'$\pm$'+stdevs[0]+
            ' & '+medias[1]+'$\pm$'+stdevs[1]+
            ' & '+medias[2]+'$\pm$'+stdevs[2]+
            ' & '+medias[3]+'$\pm$'+stdevs[3] )
file1.write('\n')
medias = ["{:.2f}".format(mean(PPV[0].ravel())),"{:.2f}".format(mean(PPV[1].ravel())),
          "{:.2f}".format(mean(PPV[2].ravel())),"{:.2f}".format(mean(PPV[3].ravel()))]
stdevs = ["{:.2f}".format(pstdev(PPV[0].ravel())),"{:.2f}".format(pstdev(PPV[1].ravel())),
          "{:.2f}".format(pstdev(PPV[2].ravel())),"{:.2f}".format(pstdev(PPV[3].ravel()))]
file1.write(nfstr+' & '+medias[0]+'$\pm$'+stdevs[0]+
            ' & '+medias[1]+'$\pm$'+stdevs[1]+
            ' & '+medias[2]+'$\pm$'+stdevs[2]+
            ' & '+medias[3]+'$\pm$'+stdevs[3] )
file1.write('\n')
medias = ["{:.2f}".format(mean(fScore[0].ravel())),"{:.2f}".format(mean(fScore[1].ravel())),
          "{:.2f}".format(mean(fScore[2].ravel())),"{:.2f}".format(mean(fScore[3].ravel()))]
stdevs = ["{:.2f}".format(pstdev(fScore[0].ravel())),"{:.2f}".format(pstdev(fScore[1].ravel())),
          "{:.2f}".format(pstdev(fScore[2].ravel())),"{:.2f}".format(pstdev(fScore[3].ravel()))]
file1.write(nfstr+' & '+medias[0]+'$\pm$'+stdevs[0]+
            ' & '+medias[1]+'$\pm$'+stdevs[1]+
            ' & '+medias[2]+'$\pm$'+stdevs[2]+
            ' & '+medias[3]+'$\pm$'+stdevs[3] )
file1.write('\n')
medias = ["{:.2f}".format(mean(AUC[0].ravel())),"{:.2f}".format(mean(AUC[1].ravel())),
          "{:.2f}".format(mean(AUC[2].ravel())),"{:.2f}".format(mean(AUC[3].ravel()))]
stdevs = ["{:.2f}".format(pstdev(AUC[0].ravel())),"{:.2f}".format(pstdev(AUC[1].ravel())),
          "{:.2f}".format(pstdev(AUC[2].ravel())),"{:.2f}".format(pstdev(AUC[3].ravel()))]
file1.write(nfstr+' & '+medias[0]+'$\pm$'+stdevs[0]+
            ' & '+medias[1]+'$\pm$'+stdevs[1]+
            ' & '+medias[2]+'$\pm$'+stdevs[2]+
            ' & '+medias[3]+'$\pm$'+stdevs[3] )
file1.close()

# Saving data
# R score
np.savetxt("R_lin"+str(nfold)+".csv",
           np.insert(Repeat[0].ravel(),0,nfold), delimiter=",")
np.savetxt("R_rbf"+str(nfold)+".csv",
           np.insert(Repeat[1].ravel(),0,nfold), delimiter=",")
np.savetxt("R_mlp"+str(nfold)+".csv",
           np.insert(Repeat[2].ravel(),0,nfold), delimiter=",")
np.savetxt("R_nb"+str(nfold)+".csv",
           np.insert(Repeat[3].ravel(),0,nfold), delimiter=",")
# PPV score
np.savetxt("PPV_lin"+str(nfold)+".csv",
           np.insert(PPV[0].ravel(),0,nfold), delimiter=",")
np.savetxt("PPV_rbf"+str(nfold)+".csv",
           np.insert(PPV[1].ravel(),0,nfold), delimiter=",")
np.savetxt("PPV_mlp"+str(nfold)+".csv",
           np.insert(PPV[2].ravel(),0,nfold), delimiter=",")
np.savetxt("PPV_nb"+str(nfold)+".csv",
           np.insert(PPV[3].ravel(),0,nfold), delimiter=",")
# f score
np.savetxt("fS_lin"+str(nfold)+".csv",
           np.insert(fScore[0].ravel(),0,nfold), delimiter=",")
np.savetxt("fS_rbf"+str(nfold)+".csv",
           np.insert(fScore[1].ravel(),0,nfold), delimiter=",")
np.savetxt("fS_mlp"+str(nfold)+".csv",
           np.insert(fScore[2].ravel(),0,nfold), delimiter=",")
np.savetxt("fS_nb"+str(nfold)+".csv",
           np.insert(fScore[3].ravel(),0,nfold), delimiter=",")
# AUC score
np.savetxt("AUC_lin"+str(nfold)+".csv",
           np.insert(AUC[0].ravel(),0,nfold), delimiter=",")
np.savetxt("AUC_rbf"+str(nfold)+".csv",
           np.insert(AUC[1].ravel(),0,nfold), delimiter=",")
np.savetxt("AUC_mlp"+str(nfold)+".csv",
           np.insert(AUC[2].ravel(),0,nfold), delimiter=",")
np.savetxt("AUC_nb"+str(nfold)+".csv",
           np.insert(AUC[3].ravel(),0,nfold), delimiter=",")

##print('Results Accuracy')
##print("{:.2f}".format(mean(Accuracy.ravel())))
##print("{:.2f}".format(pstdev(Accuracy.ravel())))
##print('Results Repeat')
##print("{:.2f}".format(mean(Repeat.ravel())))
##print("{:.2f}".format(pstdev(Repeat.ravel())))
##print('Results Positive Predictive Value')
##print("{:.2f}".format(mean(PPV.ravel())))
##print("{:.2f}".format(pstdev(PPV.ravel())))
##print('Results f Score')
##print("{:.2f}".format(mean(fScore.ravel())))
##print("{:.2f}".format(pstdev(fScore.ravel())))
##print('Results AUC')
##print("{:.2f}".format(mean(AUC.ravel())))
##print("{:.2f}".format(pstdev(AUC.ravel())))

##
### summarize class distribution
##counter = Counter(y)
##print(counter)
##
### Classification
##over = SMOTE(k_neighbors=7)
##
### Decision Tree Classifier
##steps = [('over', over), ('model', DecisionTreeClassifier())]
##pipeline = Pipeline(steps=steps)
##cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
##scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
##print('Mean ROC AUC: %.3f' % mean(scores))
##
### Decision Tree Classifier
##steps = [('over', over), ('model', LinearSVC())]
##pipeline = Pipeline(steps=steps)
##cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
##scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
##print('Mean ROC AUC: %.3f' % mean(scores))
##

##
##
### SVM only accepts numerical values. 
### Therefore, we will transform the categories into
### values 1 and 0.
##
### at.run
##cancannot_map = {'Citizens CANNOT run for office for the next two elections':0, 'Citizens CAN run for office for the next two elections':1}
##data['at.run'] = data['at.run'].map(cancannot_map)
### at.asso
##cancannot_map = {'Citizens CANNOT associate with others and form groups':0, 'Citizens CAN associate with others and form groups':1}
##data['at.asso'] = data['at.asso'].map(cancannot_map)
### at.press
##cancannot_map = {'Media CANNOT confront the Government':0, 'Media CAN confront the Government':1}
##data['at.press'] = data['at.press'].map(cancannot_map)
### at.presaut
##cancannot_map = {'President CANNOT rule without Congress':1, 'President CAN rule without Congress':0}
##data['at.presaut'] = data['at.presaut'].map(cancannot_map)
### at.vote
##cancannot_map = {'Citizens CANNOT vote in the next two elections':0, 'Citizens CAN vote in the next two elections':1}
##data['at.vote'] = data['at.vote'].map(cancannot_map)
##
### drop last column (extra column added by pd)
### and unnecessary first column (id)
### data.drop(data.columns[[-1 0]], axis=1, inplace=True)
### put features & outputs in different DataFrames for convenience
##Y = data.loc[:, 'selected'] # all rows of 'diagnosis' 
##X_c1 = data.iloc[range(0,11080,2),[5,6,7,8,9]]  # all feature rows candidate 1
##X_c2 = data.iloc[range(1,11080,2),[5,6,7,8,9]]  # all feature rows candidate 1
##X = X_c1.values-X_c2.values
##X = pd.DataFrame(X)
##Y_c1 = Y.iloc[range(0,11080,2)]  # all feature rows candidate 1
##Y_c2 = Y.iloc[range(1,11080,2)]  # all feature rows candidate 1
##Y = Y_c1.values-Y_c2.values
##Y = pd.DataFrame(Y)
##W = pd.DataFrame(data=None,columns=['k','w.at.run','w.at.asso','w.at.press','w.at.presaut','w.at.vote','selected','at.run','at.asso','at.press','at.presaut','at.vote'])
##
##print("training started...")
##for i in list(range(int(len(Y)/5))):
##    print(i)
##    X_train = X.iloc[5*i:5*(i+1),:]
##    #X_train = [X_train.iloc[0,:],X_train.iloc[1,:],X_train.iloc[2,:],X_train.iloc[3,:],X_train.iloc[4,:]]
##    y_train = Y.iloc[5*i:5*(i+1)]
##    if (-1 in np.array(y_train)) and (1 in np.array(y_train)):
##        #clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5, fit_intercept=False))
##        clf = LinearSVC(random_state=0, tol=1e-5, fit_intercept=False, C = 10, max_iter = 2000)
##        clf.fit(X_train, y_train.values.ravel())
##        #print(clf.decision_function(np.eye(5)))
##        w=list(clf.decision_function(np.eye(5)))
##
##        w = [i+1]+w
##        w = pd.DataFrame({'k':[w[0],w[0],w[0],w[0],w[0]],
##                          'w.at.run':[w[1],w[1],w[1],w[1],w[1]],
##                          'w.at.asso':[w[2],w[2],w[2],w[2],w[2]],
##                          'w.at.press':[w[3],w[3],w[3],w[3],w[3]],
##                          'w.at.presaut':[w[4],w[4],w[4],w[4],w[4]],
##                          'w.at.vote':[w[5],w[5],w[5],w[5],w[5]]})
##        #aux=pd.DataFrame(np.ones((5,1))*w)
##        w['selected']=y_train.values
##        w['at.run']=X_train[0].values
##        w['at.asso']=X_train[1].values
##        w['at.press']=X_train[2].values
##        w['at.presaut']=X_train[3].values
##        w['at.vote']=X_train[4].values
##        W = pd.concat([W,w])
##
##pd.DataFrame(W).to_excel(r'./File Name.xlsx', index = False)
##
##
