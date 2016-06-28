
### Author: Ikenna Ivenso ###

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import AdaBoostClassifier
from sklearn import cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from unbalanced_dataset.under_sampling import RandomUnderSampler

#=======================================================================

def preProcess(theFileName):
    df = pd.read_csv(str(theFileName))
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis = 1)
    labBin = sklearn.preprocessing.LabelBinarizer()
    df['y'] = labBin.fit_transform(df['y'])
    dp = pd.get_dummies(df)
    X = dp.drop('y', axis = 1) 
    y = dp[['y']]

    # get the features
    theFeatures = X.columns

    # convert the dataframes to arrays
    X = X.values
    y = y.values
    y.shape = np.shape(y)[0]

    yOrig = y[:] # need this later for plotting feature impacts

    # and carry out feature scaling
    X = StandardScaler().fit_transform(X)

    #=======================================================================

    # apply random undersampling if labels are imbalanced
    labelSkewness = 100*np.sum(y)*1./np.shape(y)[0]
    if np.min([labelSkewness, 100-labelSkewness]) < (100./3.):
        rus = RandomUnderSampler(verbose=0)
        X, y = rus.fit_sample(X, y)

    #=======================================================================

    # select optimal number of features
    thisModel = LogisticRegression(penalty='l1', C=1)
    rfecv = RFECV(estimator=thisModel, step=1, cv=StratifiedKFold(y, n_folds=3), scoring='f1')
    Xt = rfecv.fit_transform(X, y);

    optimalNumberOfFeatures = rfecv.n_features_
    introReport = ['Optimal Number of Attributes: ' + str(optimalNumberOfFeatures), 'The following attributes are the most influential to the outcome']

    #=======================================================================

    # plot number of selected features VS cross-validation scores
    plt.figure(figsize=(12, 8))

    plt.xlabel("Number of Attributes", fontsize=20)
    plt.ylabel("Score", fontsize=20)
    plt.title("Attribute Selection", fontsize=25)
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

    imgOne = 'static/thePlot.jpg'
    plt.savefig('flask_files/'+imgOne, dpi=300)
    
    #=======================================================================

    # get the feature feature importance rankings
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X,y)
    theImportances = list(model.feature_importances_)
    sortedImportances = sorted(theImportances,reverse = True)

    # ...and print the selected features along with their weights and ranks
    tableOne = []
    for ii in range(1,optimalNumberOfFeatures+1):
        tableOne.append(dict(Feature = str(theFeatures[theImportances.index(sortedImportances[ii-1])]), Weight = str(sortedImportances[ii-1]), Rank = str(ii)))

    #=======================================================================

    # plot histogram of the most important feature
    thisFeature = 0
    allThoseFeatures = dp[theFeatures[theImportances.index(sortedImportances[thisFeature])]]

    plt.figure(figsize=(12, 8))
    
    combinedOutcomes = plt.hist(allThoseFeatures, bins=10)

#    plt.hist(allThoseFeatures, bins=10)
    plt.xlabel('Attribute: ' + theFeatures[theImportances.index(sortedImportances[0])], fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.title('Impact of the Most Influential Attribute', fontsize=25)

    imgTwo = 'static/theHist.jpg'
    plt.savefig('flask_files/'+imgTwo, dpi=300)

    #=======================================================================

    # plot impact of the most important feature
    positiv = allThoseFeatures[yOrig==1]
    negativ = allThoseFeatures[yOrig==0]

    plt.figure(figsize=(12, 8))
    
    negA = plt.hist(negativ,bins=combinedOutcomes[1])
    posA = plt.hist(positiv,bins=combinedOutcomes[1])
#    yUpperLimit = np.max([negA[0], posA[0]])*1.01

#    plt.subplot(1,2,1)
#    plt.hist(negativ,bins=combinedOutcomes[1])
#    plt.ylim(ymax = yUpperLimit*1.01, ymin = 0)
#    plt.xlabel(theFeatures[theImportances.index(sortedImportances[thisFeature])], fontsize=16)
#    plt.ylabel('Count', fontsize=16)
#    plt.title('Negative', fontsize=20)
#
#    plt.subplot(1,2,2)
#    plt.hist(positiv,bins=combinedOutcomes[1])
#    plt.ylim(ymax = yUpperLimit, ymin = 0)
#    plt.xlabel(theFeatures[theImportances.index(sortedImportances[thisFeature])], fontsize=16)
#    plt.title('Positive',fontsize=20)
#
#    imgThree = 'static/theNegPosHist.jpg'
#    plt.savefig('flask_files/'+imgThree, dpi=300)

    #=======================================================================
    
    a = posA[0]
    b = negA[0]
    c = combinedOutcomes[0]

    posImpact = np.divide(a,c)
    negImpact = np.divide(b,c)

    midPoints=[]
    for i in range(1,len(combinedOutcomes[1])):
        midPoints.append((combinedOutcomes[1][i] + combinedOutcomes[1][i-1])/2.)

    for i in range(len(posImpact)):
        if np.isnan(posImpact[i]):
            posImpact[i]=0
        if np.isnan(negImpact[i]):
            negImpact[i]=0

    plt.figure(figsize=(12, 8))
    plt.hold(True)
    plt.plot(midPoints, posImpact,'.', markersize=20, label='Positive')
    plt.plot(midPoints, negImpact, 'r.', markersize=20, label='Negative')
    plt.legend(prop={'size':20})
    plt.xlabel(theFeatures[theImportances.index(sortedImportances[thisFeature])], fontsize=16)
    plt.ylabel('Relative Impact', fontsize=20)
    plt.grid()

    imgThree = 'static/theNegPosHist.jpg'
    plt.savefig('flask_files/'+imgThree, dpi=300)

    #=======================================================================

    # generate plots for report (this is save to an "html" file)

    from bokeh.charts import Histogram, output_file, show, save, gridplot
    from bokeh.plotting import figure

    plotList=[]

    for i in range(optimalNumberOfFeatures):
        thisFeatureIs = theFeatures[theImportances.index(sortedImportances[i])]
        allThoseFeatures = dp[thisFeatureIs]
        combinedOutcomes = plt.hist(allThoseFeatures, bins=10)
        
        positiv = allThoseFeatures[yOrig==1]
        negativ = allThoseFeatures[yOrig==0]
        negA = plt.hist(negativ,bins=combinedOutcomes[1])
        posA = plt.hist(positiv,bins=combinedOutcomes[1])
        posImpact = np.divide(posA[0],combinedOutcomes[0])
        negImpact = np.divide(negA[0],combinedOutcomes[0])
        
        midPoints=[]
        for i in range(1,len(combinedOutcomes[1])):
            midPoints.append((combinedOutcomes[1][i] + combinedOutcomes[1][i-1])/2.)
        
        for i in range(len(posImpact)):
            if np.isnan(posImpact[i]):
                posImpact[i]=0
            if np.isnan(negImpact[i]):
                negImpact[i]=0

        hist0 = Histogram(dp, values=thisFeatureIs, color='blue', title="Impact of " + thisFeatureIs, bins=10)
        plot0 = figure()
        plot0.xaxis.axis_label = thisFeatureIs
        plot0.yaxis.axis_label = "Relative Impact"
        #     plot0.title = "Relative Impact of " + thisFeatureIs
        plot0.circle(midPoints, list(negImpact), size=10, color="red", alpha=0.9, legend='Negative')
        plot0.circle(midPoints, list(posImpact), size=10, color="green", alpha=0.9, legend='Positive')
        plotList.append([hist0,plot0])

    output_file("flask_files/static/Report.html", title = "Report")
    hist = gridplot(plotList)
    save(hist)

    #=======================================================================

    # specify the models to run tests with
    theModels = {'Logistic Regression':LogisticRegression(penalty='l1'), 'LDA':LinearDiscriminantAnalysis(), 'SVM':SVC(kernel='linear'), 'Random Forest':RandomForestClassifier(n_estimators=300)}

    # ...then display the results of the tests
    classifierComparisons=[]
    for aModel in theModels:
        model = theModels[aModel]
        results = cross_validation.cross_val_score(model, Xt, y, scoring='f1', cv=StratifiedKFold(y, n_folds=3))
        classifierComparisons.append(dict(Classifier = aModel, Score = np.max(results)))

    #=======================================================================

    # display the plots
    theJPGs = [imgOne, imgTwo, imgThree]

    #=======================================================================

    return introReport, tableOne, optimalNumberOfFeatures, classifierComparisons, theJPGs







