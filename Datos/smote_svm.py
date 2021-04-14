# svm.py
import numpy as np  # for handling multi-dimensional array operation
import pandas as pd  # for reading data from csv
from sklearn import svm # for classification problem
from sklearn.pipeline import make_pipeline # create pipeline
from sklearn.preprocessing import StandardScaler # scaling data
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
from numpy import mean
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import random
from statistics import pstdev
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('./dataMergedConjoint.csv')
data = data.drop_duplicates(subset=['idnum'])

X = data[['woman', 'socideo', 'partyid', 'reg', 'trustfed', 'income.n','educ.n', 'polknow']]
y = data['vote.selling']
#concatenate?

#Standarization
xvalue = X.values
min_max_scaler = preprocessing.MinMaxScaler()
xscaled = min_max_scaler.fit_transform(xvalue)
X = pd.DataFrame(xscaled)

#training_pairs = pd.concat([X,y],axis=1)
nobs = len(X)
nfold = 10
lsets = int(np.ceil(nobs/nfold))
nrepeat = 3
RepeatSVMLinear = np.zeros([nrepeat,nfold])

PPVSVMLinear = np.zeros([nrepeat,nfold])

for i in range(nrepeat):
    indexes = list(range(nobs))
    random.shuffle(indexes)
    dfs = np.array_split(indexes,nfold)
    for j in range(nfold):
        
        index_bad = X.index.isin(dfs[j])
        X_test = X[index_bad]
        y_test = y[index_bad]
        X_train = X[~index_bad]
        y_train = y[~index_bad]
        #SMOTE
        oversample = SMOTE(k_neighbors=7)
        X_train,y_train = oversample.fit_resample(X_train,y_train)

        linear_svc = svm.SVC(kernel='rbf',random_state=0, tol=1e-5, C = 1)
        linear_svc.fit(X_train, y_train)

        y_predict = linear_svc.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test.ravel(),y_predict).ravel()
        RepeatSVMLinear[i,j] = tp/(tp+fn)
        PPVSVMLinear[i,j] = tp/(tp+fp)

print('Linear')
print(mean(100*RepeatSVMLinear.ravel()))
print(pstdev(100*RepeatSVMLinear.ravel()))
print(mean(100*PPVSVMLinear.ravel()))
print(pstdev(100*PPVSVMLinear.ravel()))

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
