# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 23:57:58 2016

@author: pithapliyal
"""

"""
Created on Wed Mar 30 15:24:09 2016

@author: pithapliyal
"""

import pandas as pd
import numpy as np
import sys, os, cPickle, datetime
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.ensemble import GradientBoostingRegressor as GBM
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error as mse
#from scipy import stats
from pprint import pprint
from sklearn.cross_validation import KFold
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import MinMaxScaler

#==============================================================================
# Config
#==============================================================================
wdir = 'C:\\Users\\pithapliyal\\Documents\\personal\\'
os.chdir(wdir)

targetCol = 'price'
dropCols = []
catCols = ['key']
removeFeats = ['ID']
verbose = True
#==============================================================================
#==============================================================================

def scatterPlot(x, y, xn=None, yn=None, group=None, gn=None):
    if xn is None: xn = x.name if type(x) == pd.core.series.Series else 'x'
    if yn is None: yn = y.name if type(y) == pd.core.series.Series else 'y'
    if gn is None: gn = group.name if type(group) == pd.core.series.Series else 'group'
    if group is None: group = [1]*len(x)
    x = deepcopy(list(x))
    y = deepcopy(list(y))
    group = deepcopy(group)
    allgroups = list(set(group))
    df = pd.DataFrame(zip(x, y, group), columns=[xn, yn, gn])
    colors = cm.rainbow(np.linspace(0, 1, len(allgroups)))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for gp, c in zip(allgroups, colors):
        x = list(df[xn][df[gn] == gp])
        y = list(df[yn][df[gn] == gp])
        ax.scatter(x, y, color=c, label=gp, s=5)
    ax.set_xlabel(xn)
    ax.set_ylabel(yn)
    ax.legend(title=gn, loc="upper left", bbox_to_anchor=(1,1))
    return fig
       
def readData(fname='TransactionDataCompiledFinal.csv'):
    df = pd.read_csv(fname)
    if 'Age' not in df.columns:
        df['Age'] = map(lambda x: 2016 - x, df['Year'])
    if 'Sold Price' not in df.columns:
        df['Sold Price'] = None
    
    df['ID'] = range(df.shape[0])
    df['key'] = map(lambda x, y, z: '%s|%s|%s'%(x.upper(), y.upper(), z.upper()), df['Model'], df['Variant'], df['City'])
    df = df[['ID', 'key', 'Sold Price', 'Year', 'Ownership', 'Age', 'Out Kms']]
    df.columns = ['ID', 'key', 'price', 'year', 'owners', 'age', 'mileage']
    
    tmp = df.groupby('key', as_index=False).agg({'ID':'count'}).rename(columns={'ID':'Freq'}).sort('Freq', ascending=False)
    tmp = tmp[tmp.Freq > 15]
    df = df.merge(tmp[['key']], on='key', how='inner')
        
    # Add any pre-processing steps here
    # Convert all featuers to float
    return df

def impute(df, ref=None):
    if ref is None:
        ref = df[df.columns]
    else:
        pass
    ref = ref[[x for x in df.columns if x not in dropCols + removeFeats + catCols]]
    refMedian = ref.median().reset_index().rename(columns={'index':'feature', 0:'median'})
    refDict = dict(zip(refMedian['feature'], refMedian['median']))
    
    for feat in df.columns:
        if df[pd.isnull(df[feat])].shape[0] != 0:
            df[feat] = map(lambda x: refDict[feat] if pd.isnull(x) else x, df[feat])
        else:
            pass
    return df

def oneHot(trn, tst, tot, col, oneHotThresh):
    catFreq = trn.groupby(col, as_index=False).agg({'ID':'count'}).rename(columns={'ID':'Freq'})
    catFreq = catFreq[catFreq.Freq > oneHotThresh]
    catFreq[col] = map(lambda x: x.strip().replace(' ', ''), catFreq[col])
    catList = list(catFreq[col])
    for v in catList:
        trn[col + '_' + v] = map(lambda x: 1.0 if x.strip().replace(' ', '') == v else 0.0, trn[col])
        tst[col + '_' + v] = map(lambda x: 1.0 if x.strip().replace(' ', '') == v else 0.0, tst[col])
        tot[col + '_' + v] = map(lambda x: 1.0 if x.strip().replace(' ', '') == v else 0.0, tot[col])
    trn[col + '_OTHERS'] = map(lambda *ls: 1.0 if sum(ls) == 0 else 0.0, *[trn[col + '_' + v] for v in catList])
    tst[col + '_OTHERS'] = map(lambda *ls: 1.0 if sum(ls) == 0 else 0.0, *[tst[col + '_' + v] for v in catList])
    tot[col + '_OTHERS'] = map(lambda *ls: 1.0 if sum(ls) == 0 else 0.0, *[tot[col + '_' + v] for v in catList])
    trn = trn.drop(col, axis=1)
    tst = tst.drop(col, axis=1)
    tot = tot.drop(col, axis=1)
    return trn, tst, tot

def runKFold(model, n, X, Y):
    X = np.array(X.copy())
    Y = np.array(Y.copy())
    kf = KFold(X.shape[0], n_folds=n, shuffle= True, random_state= 123)
    rmselist1 = []
    rmselist2 = []
    preds = []
    targs = []
    idxs = []
    fold = []
    i = 0
    for train_index, test_index in kf:
        idxs.extend(test_index)
        fold.extend([i]*len(test_index))
        i += 1
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model.fit(X_train, Y_train)
        Y_hat_test = model.predict(X_test)
        preds.extend(Y_hat_test)
        targs.extend(Y_test)
        rmselist1.append(pow(mse(Y_test, Y_hat_test), 0.5))
        Yhat = model.predict(X)
        rmselist2.append(pow(mse(Y, Yhat), 0.5))
    rmse1 = np.mean(rmselist1)
    rmse2 = pow(mse(targs, preds), 0.5)
    tmp = pd.DataFrame(zip(idxs, fold, targs, preds), columns=['index', 'fold', 'target', 'predicted'])
    return rmse1, rmse2, rmselist1, rmselist2, tmp

def buildModel(model, Xtrn, Ytrn, Xtst):
    model.fit(Xtrn, Ytrn)
    Yhat_trn = model.predict(Xtrn)
    if Xtst is not None:
        Yhat_tst = model.predict(Xtst)
    else:
        Yhat_tst = None
    rmse = pow(mse(Ytrn, Yhat_trn), 0.5)
    return Yhat_trn, Yhat_tst, rmse

def extractN(perf_eval, n):
    if perf_eval.endswith('out') or perf_eval[0] == 'n' or perf_eval in ('loo'): nfold = n
    elif perf_eval in ('kfold', 'k-fold', 'cv'): nfold = 10
    else: nfold = int(float(perf_eval.replace('fold', '')))
    return nfold

def getNPArrays(trn, tst, targetCol, removeFeats, keepFeats = None):
    if keepFeats is not None:
        Xtrn = np.array(trn[keepFeats])
        featNames = trn[keepFeats].columns
    else:
        Xtrn = np.array(trn.drop([targetCol] + removeFeats, axis=1))
        featNames = trn.drop([targetCol] + removeFeats, axis=1).columns
    if tst is not None:
        if keepFeats is not None:
            Xtst = np.array(tst[keepFeats])
        else:
            Xtst = np.array(tst.drop([targetCol] + removeFeats, axis=1))
    else:
        Xtst = None
    Ytrn = np.array(trn[targetCol])
    
    return Xtrn, Xtst, Ytrn, featNames

def runForewardSelection(model, trn, targetCol, checkVars, keepVars, best_rmse, perf_all):
    #func_list = [np.log, np.exp, np.sqrt]
    best_var = ''
    for var in checkVars:
        Xtrn = np.array(trn[keepVars + [var]])
        Ytrn = np.array(trn[targetCol])
        perf = runKFold(model, 5, Xtrn, Ytrn)[0]
        perf_all = pd.concat([perf_all, pd.DataFrame([[keepVars + [var], perf]], columns=['feats', 'perf'])])
        
        if perf < best_rmse:
            best_rmse = perf
            best_var = var
        else:
            continue
    
    if best_var == '':
        return keepVars, perf_all
    else:
        keepVars = keepVars + [best_var]
        checkVars = [x for x in checkVars if x != best_var]
        return runForewardSelection(model, trn, targetCol, checkVars, keepVars, best_rmse, perf_all)

def trainModel(trn, tst, targetCol=targetCol, 
                   model_type       = '',
                   featureSelect    = '', 
                   perf_eval        = '', 
                   model_args       = {}
               ):
    featureSelect = featureSelect.lower()
    perf_eval = perf_eval.lower()
    model_type = model_type.lower()
    kfoldPreds = None
    ignoreFeats = []
    
    if model_type not in ('gaussian', 'gaussiannaivebayes'):
        trn = impute(trn, tot)
        tst = impute(tst, tot)

    
    if model_type in ('linear', 'lm', 'lr', 'linearreg', 'linearregression'):
        model = LinReg(
                        fit_intercept   = model_args['fit_intercept'],
                        normalize       = model_args['normalize']
                        )
    elif model_type in ('rf', 'randomforest', 'randomforestregression'):
        model = RF(max_features = 'sqrt', random_state = 123, oob_score=True, n_jobs = -1, 
                        n_estimators        = model_args['n_estimators'], 
                        min_samples_split   = model_args['min_samples_split'],
                        min_samples_leaf    = model_args['min_samples_leaf'],
                        max_depth           = model_args['max_depth']
                    )
    
    elif model_type in ('gbm', 'boosting', 'radientboostingmachine', 'gradientboosting'):
        model = GBM(subsample = 0.8, max_features= "sqrt", random_state = 123,
                        n_estimators        = model_args['n_estimators'],
                        min_samples_split   = model_args['min_samples_split'],
                        min_samples_leaf    = model_args['min_samples_leaf'],
                        max_depth           = model_args['max_depth']
                    )
    elif model_type in ('gaussian', 'gaussiannaivebayes'):
        model = gaussianRegression()
        ignoreFeats.extend([x for x in trn.columns if x.startswith('key')])
    else:
        raise Exception('Invalid model selection')
    
    ignoreFeats.extend(removeFeats)
    trn = trn.copy()
    tst = tst.copy()
    
    if featureSelect == 'None' or featureSelect is None or featureSelect == '':
        Xtrn, Xtst, Ytrn, featNames = getNPArrays(trn, tst, targetCol, ignoreFeats)
        feat_sel = deepcopy(featNames)
        perf_all = None

        if perf_eval == '' or perf_eval is None:
            if verbose: print "No validation."
            Yhat_trn, Yhat_tst, rmse = buildModel(model, Xtrn, Ytrn, Xtst)
            
        elif perf_eval.endswith('fold') or perf_eval.endswith('out') or perf_eval in ('loo', 'cv'):
            nfold = extractN(perf_eval, Xtrn.shape[0])
            if verbose: print "Running %d-fold validation" %nfold
                
            rmse1, rmse2, rmselist1, rmselist2, kfoldPreds = runKFold(model, nfold, Xtrn, Ytrn)
            Yhat_trn = list(kfoldPreds.sort('index')['predicted'])
            if verbose: 
                print "Average RMSE of all folds", rmse1, "\nTotal RMSE - stacked folds", rmse2
                pprint(rmselist1)
                pprint(rmselist2)
            
            _, Yhat_tst, rmsefinal = buildModel(model, Xtrn, Ytrn, Xtst)
            rmse = {'rmse1': rmse1, 'rmse2':rmse2, 'rmselist1':rmselist1, 'rmselist2':rmselist2, 'rmse': rmsefinal}
        else:
            raise Exception("PT: Unknown value for argument perf_eval: %s"%perf_eval)
            
    elif featureSelect in ('pvals', 'pval', 'p'):
        raise Exception("Removed feature selection %s from framework" %featureSelect)
        
    elif featureSelect in ('grid', 'forewardsearch', 'foreward', 'fwd', 'backward', 'backwardsearch', 'iterative'):
        if featureSelect.startswith('b'):
            raise Exception('Backward selection not supported')
        checkVars = [x for x in trn.columns if x not in [targetCol] + ignoreFeats]
        feat_sel, perf_all = runForewardSelection(model, trn, targetCol, 
                                                      checkVars = deepcopy(checkVars), 
                                                      keepVars  = [], 
                                                      best_rmse = 1000000,
                                                      perf_all = pd.DataFrame()
                                                    )
        Xtrn, Xtst, Ytrn, featNames = getNPArrays(trn, tst, targetCol, ignoreFeats, feat_sel)
        
        if perf_eval == '' or perf_eval is None:
            if verbose: print "No validation."
            Yhat_trn, Yhat_tst, rmse = buildModel(model, Xtrn, Ytrn, Xtst)
            
        elif perf_eval.endswith('fold') or perf_eval.endswith('out') or perf_eval in ('loo', 'cv'):
            nfold = extractN(perf_eval, Xtrn.shape[0])
            if verbose: print "Running %d-fold validation" %nfold
                
            rmse1, rmse2, rmselist1, rmselist2, kfoldPreds = runKFold(model, nfold, Xtrn, Ytrn)
            Yhat_trn = list(kfoldPreds.sort('index')['predicted'])
            if verbose: 
                print "Average RMSE of all folds", rmse1, "\nTotal RMSE - stacked folds", rmse2
                pprint(rmselist1)
                pprint(rmselist2)
            
            _, Yhat_tst, rmsefinal = buildModel(model, Xtrn, Ytrn, Xtst)
            rmse = {'rmse1': rmse1, 'rmse2':rmse2, 'rmselist1':rmselist1, 'rmselist2':rmselist2, 'rmse': rmsefinal}
        else:
            raise Exception("PT: Unknown value for argument perf_eval: %s"%perf_eval)
    else:
        raise Exception("PT: feature selection method %s Not implemented" %featureSelect)
    return Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds

def writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix=None, log_flag=False):
    out = tst[['ID']]
    out['Predicted'] = list(Yhat_tst)
    if log_flag:
        out['Predicted'] = map(lambda x: np.exp(x), out['Predicted'])
    if prefix is None:
        prefix=raw_input('Input the prefix for the output:')
    out.to_csv(prefix +'_' + 'logTarget'*log_flag + '_' + str(datetime.datetime.now())[:19].replace(':', '-') + '.csv', index=False)
    
    out = trn[['ID', 'price']]
    out['Predicted'] = list(Yhat_trn)
    if log_flag:
        out['Predicted'] = map(lambda x: np.exp(x), out['Predicted'])
        out['price'] = map(lambda x: np.exp(x), out['price'])
    out.to_csv(prefix +'TRAIN_' + 'logTarget'*log_flag + '_' + str(datetime.datetime.now())[:19].replace(':', '-') + '.csv', index=False)        

def plotScatterCharts(df, targOnly = True):
    pp = PdfPages('Charts\\scatter_charts_all%s.pdf' %('targOnly'*targOnly))
    df = df.copy()
    df = df[[x for x in df.columns if x not in catCols + removeFeats + dropCols]]
    for c1 in df.columns:
        for c2 in df.columns:
            if c1 >= c2 or 'Tag' in [c1, c2]: 
                continue
            if (c2 != 'price') and targOnly == True:
                continue
            print c1, c2
            fig = scatterPlot(df[c1], df[c2], group=df['Tag'])
            fig.savefig(pp, format='pdf')
    pp.close()
    try:
        pp.close()
    except:
        pass
    pp = 1
 
class gaussianRegression():
    def __init__(self):
        self.mean = []
        self.cov = []
        self.std = []
        self.const = []
        self.rho = []
        self.k = 2
        
    def fit(self, X, Y):
        self.__init__()
        Y = Y.copy()
        Y = Y.reshape(Y.shape[0], 1)
        data = np.hstack((Y, X))
        data = np.ma.array(data, mask=np.isnan(data))
        self.mean = np.array(np.ma.mean(data, axis=0))
        self.cov = np.array(np.ma.cov(data, rowvar=False))
        for i in range(data.shape[1]):
            self.std.append(pow(self.cov[i][i], 0.5))
            if i == 0:
                self.const.append(np.nan)
                self.rho.append(np.nan)
            else:
                self.const.append(1/pow(pow(2*np.pi, self.k)*np.linalg.det(self.cov[np.ix_([0, i], [0, i])]), 2))
                self.rho.append(self.cov[0][i] / pow((self.cov[0][0]*self.cov[i][i]), 0.5))
        
    def getVarProb(self, i, x, y):
        i = i + 1
        if x != x:
            pdf = 1.0
        else:
            pdf = self.const[i] * np.exp(-1.0 * (
                                                    pow((x - self.mean[i])/self.std[i], 2) + 
                                                    pow((y - self.mean[0])/self.std[0], 2) - 
                                                    2 * self.rho[i] * (x - self.mean[i]) * (y - self.mean[0]) / (self.std[i] * self.std[0])
                                                ) / (2 * (1 - pow(self.rho[i], 2)))
                                        )
        return np.log(pdf)
        
    def getOverallProb(self, y, ls):
        overallProb = sum([self.getVarProb(i, x, y) for i, x in enumerate(ls)])
        return -1.0*overallProb
        
    def getPred(self, ls):
        pred = minimize_scalar(self.getOverallProb, args=(ls))
        return pred.x
        
    def predict(self, X):
        Yhat = map(self.getPred, X)
        return Yhat   

def scaleFeatures(trn, tst, tot):
    # Implement a min_max scaler to convert all data to range 1-2
    # Handle NaNs
    return trn, tst, tot

def featureTransform(trn, tst, tot):
    # add exp/log/sqr/sqrt/inv of each (possible) variable to the data
    # Handle NaNs
    return trn, tst, tot

#==============================================================================
# Run modelling
#==============================================================================
verbose = 0
for log_flag in [False, True]:
    trn = readData('TransactionDataCompiledFinal.csv')
    tst = readData('sourcelist.csv')
    if log_flag:
        trn['price'] = map(lambda x: np.log(x), trn['price'])
    tot = pd.concat([
                        pd.concat([trn, tst]).reset_index(drop=True),
                        pd.DataFrame([['train']]*trn.shape[0] + [['test']]*tst.shape[0], columns=['Tag'])
                    ], axis=1)
    
    #trn, tst, tot = scaleFeatures(trn, tst, tot)
    #trn, tst, tot = featureTransform(trn, tst, tot)
    trn, tst, tot = oneHot(trn, tst, tot, 'key', 0)
    #plotScatterCharts(tot, True)
    #plotScatterCharts(tot, False)
    
    
    # Linear model
    # No intercept does not differ from with intercept version at all
    model_args = {'fit_intercept':True, 'normalize':True}
    Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds = trainModel(trn, tst, targetCol=targetCol, 
                                                                  model_type       = 'linear',
                                                                  featureSelect    = '', 
                                                                  perf_eval        = '10fold', 
                                                                  model_args = model_args
                                                            )
    print "Linear Regression, 10fold, with intercept, normalize, no feature selection"
    pprint(rmse)
    print "---------------------------------------"
    writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix='submissions\\lr_noFeatSel_10fold_intercept_normalize', log_flag=log_flag)
    outlist = [Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds]
    cPickle.dump(outlist, open('pickles\\lr_noFeatSel_10fold_intercept_normalize_'+'logTarget'*log_flag + '.pickle', 'w'))
    
    model_args = {'fit_intercept':True, 'normalize':True}
    Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds = trainModel(trn, tst, targetCol=targetCol, 
                                                                  model_type       = 'linear',
                                                                  featureSelect    = 'fwd', 
                                                                  perf_eval        = '10fold', 
                                                                  model_args = model_args
                                                            )
    print "Linear Regression, 10fold, with intercept, normalize, foreward feature selection"
    pprint(rmse)
    pprint(feat_sel)
    pprint(perf_all)
    print "---------------------------------------"
    writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix='submissions\\lr_fwdFeatSel_10fold_intercept_normalize', log_flag=log_flag)
    outlist = [Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds]
    cPickle.dump(outlist, open('pickles\\lr_fwdFeatSel_10fold_intercept_normalize_'+'logTarget'*log_flag + '.pickle', 'w'))
    
    '''
    model_args = {}
    Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds = trainModel(trn, tst, targetCol=targetCol, 
                                                                  model_type       = 'gaussian',
                                                                  featureSelect    = 'fwd', 
                                                                  perf_eval        = '10fold', 
                                                                  model_args = model_args
                                                            )
    print "Gaussian Regression, 10fold, without state variables, no normalization, foreward feature selection"
    pprint(rmse)
    pprint(feat_sel)
    pprint(perf_all)
    print "---------------------------------------"
    writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix='submissions\\gaussian_fwdFeatSel_3fold', log_flag=log_flag)
    outlist = [Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds]
    cPickle.dump(outlist, open('pickles\\gaussian_fwdFeatSel_10fold_'+'logTarget'*log_flag + '.pickle', 'w'))
    '''
    # GBM
    perf_list = []
    n = 10
    mss = 3
    msl = 1
    md = 1
    
    for n in [10, 20, 50, 100, 200]:
        for mss in [3, 5, 10, 20, 50]:
            for msl in [1, 2, 3, 5, 10, 15, 20]:
                for md in [1, 2, 3, 5, 7, 10, 15]:
                    model_args = {'n_estimators':n,
                                  'min_samples_split':mss,
                                  'min_samples_leaf': msl,
                                  'max_depth': md
                                  }
                    stTime = datetime.datetime.now()
                    _, _, rmse, _, _, _ = trainModel(trn, tst, targetCol=targetCol, 
                                                model_type       = 'gbm',
                                                featureSelect    = '', 
                                                perf_eval        = '5fold', 
                                                model_args       = model_args
                                            )
                    perf_list.append([n, mss, msl, md, rmse['rmse1']])
            print "Done", n, mss, "in", datetime.datetime.now() - stTime
            sys.stdout.flush()
    gbm_perf = pd.DataFrame(perf_list, columns=['n', 'mss', 'msl', 'md', 'rmse']).sort('rmse', ascending=True)
    gbm_perf.to_csv('perf_summary\\gbm_perf_' + 'logTarget'*log_flag + '.csv', index=False)
    pprint(gbm_perf.head(1))
    
    model_args = {'n_estimators' : int(gbm_perf.head(1)['n']),
                  'min_samples_split' : int(gbm_perf.head(1)['mss']),
                  'min_samples_leaf' : int(gbm_perf.head(1)['msl']),
                  'max_depth' : int(gbm_perf.head(1)['md'])
                  }
    
    Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds = trainModel(trn, tst, targetCol=targetCol,
                                                                          model_type       = 'gbm',
                                                                          featureSelect    = '', 
                                                                          perf_eval        = '10fold', 
                                                                          model_args       = model_args
                                                                          )
    print "GBM, %s trees, %s min_samples_split, %s mil_samples_leaf, %s max_depth" %(model_args['n_estimators'], model_args['min_samples_split'], model_args['min_samples_leaf'], model_args['max_depth'])
    pprint(rmse)
    print "---------------------------------------"
    writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix='submissions\\gbm_10fold_' + 
                                        '_'.join([str(model_args[x]) for x in ['n_estimators',
                                                                          'min_samples_split', 
                                                                          'min_samples_leaf', 
                                                                          'max_depth'
                                                                          ]]), log_flag=log_flag)
    outlist = [Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds]
    cPickle.dump(outlist, open('pickles\\gbm_10fold_' + 
                                        '_'.join([str(model_args[x]) for x in ['n_estimators',
                                                                          'min_samples_split', 
                                                                          'min_samples_leaf', 
                                                                          'max_depth'
                                                                          ]]) + '_' +'logTarget'*log_flag + '.pickle', 'w'))
    
    
    # RF
    perf_list = []
    for n in [10, 20, 50, 100, 200, 500]:
        for mss in [3, 5, 10, 20, 50]:
            for msl in [1, 2, 3, 5, 10, 15, 20]:
                for md in [1, 2, 3, 5, 7, 10, 15]:
                    model_args = {'n_estimators':n,
                                  'min_samples_split':mss,
                                  'min_samples_leaf': msl,
                                  'max_depth': md
                                  }
                    stTime = datetime.datetime.now()
                    _, _, rmse, _, _, _ = trainModel(trn, tst, targetCol=targetCol, 
                                                model_type       = 'rf',
                                                featureSelect    = '', 
                                                perf_eval        = '5fold', 
                                                model_args       = model_args
                                            )
                    perf_list.append([n, mss, msl, md, rmse['rmse1']])
            print "Done", n, mss, "in", datetime.datetime.now() - stTime
            sys.stdout.flush()
    rf_perf = pd.DataFrame(perf_list, columns=['n', 'mss', 'msl', 'md', 'rmse']).sort('rmse', ascending=True)
    rf_perf.to_csv('perf_summary\\rf_perf_' + 'logTarget'*log_flag + '.csv', index=False)
    pprint(rf_perf.head(1))
    
    model_args = {'n_estimators' : int(rf_perf.head(1)['n']),
                  'min_samples_split' : int(rf_perf.head(1)['mss']),
                  'min_samples_leaf' : int(rf_perf.head(1)['msl']),
                  'max_depth' : int(rf_perf.head(1)['md'])
                  }
    
    Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds = trainModel(trn, tst, targetCol=targetCol,
                                                                          model_type       = 'rf',
                                                                          featureSelect    = '', 
                                                                          perf_eval        = '10fold', 
                                                                          model_args       = model_args
                                                                          )
    print "RF, %s trees, %s min_samples_split, %s mil_samples_leaf, %s max_depth" %(model_args['n_estimators'], model_args['min_samples_split'], model_args['min_samples_leaf'], model_args['max_depth'])
    pprint(rmse)
    print "---------------------------------------"
    writeOutput(tst, Yhat_tst, trn, Yhat_trn, prefix='submissions\\rf_10fold_' + 
                                        '_'.join([str(model_args[x]) for x in ['n_estimators',
                                                                          'min_samples_split', 
                                                                          'min_samples_leaf', 
                                                                          'max_depth'
                                                                          ]]), log_flag=log_flag)
    outlist = [Yhat_trn, Yhat_tst, rmse, feat_sel, perf_all, kfoldPreds]
    cPickle.dump(outlist, open('pickles\\rf_10fold_' + 
                                        '_'.join([str(model_args[x]) for x in ['n_estimators',
                                                                          'min_samples_split', 
                                                                          'min_samples_leaf', 
                                                                          'max_depth'
                                                                          ]]) + '_' +'logTarget'*log_flag + '.pickle', 'w'))
