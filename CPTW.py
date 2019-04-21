# -*- coding: iso-8859-1 -*-

from gensim import corpora
import os
import numpy as np
import string
import codecs
import io
import gensim
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection  import train_test_split
import pickle
from sklearn import neighbors
from gensim import corpora
import gensim
import pickle
import re
import os
import numpy as np
import string
from operator import itemgetter
import codecs
import io
from sklearn.model_selection  import train_test_split
from sklearn import metrics
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import GridSearchCV
from sklearn.model_selection  import RandomizedSearchCV
from time import time
from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics

from collections import namedtuple
from collections import Counter
import multiprocessing
from scipy.sparse import csr_matrix
from scipy.sparse import diags

import spacy
import time

from sklearn import metrics
import sklearn
import pickle
from scipy import interp
from sklearn.metrics import roc_curve, auc
import numpy as np
import glob

#fileV have each file corresponding to a row
def computeW2W(w2w,fileV):
    return fileV.dot( w2w) #np.dot(fileV,w2w)

def allmeasures(preds, gt, pp):
    n_classes = len(np.unique(gt))
    gt2 = np.zeros((len(gt), n_classes))
    #print "Uniks", np.unique(gt)
    for i, elem in enumerate(gt):
        gt2[i, int(elem-1)] = 1
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(gt2[:, i], pp[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(gt2.ravel(), pp.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    rocmicro =  roc_auc["micro"]
    rocmacro = roc_auc["macro"]
    f1macro =  sklearn.metrics.f1_score(gt, preds, average="macro")
    f1micro =  sklearn.metrics.f1_score(gt, preds, average="micro")
    acc = sklearn.metrics.accuracy_score(gt, preds)

    return rocmicro, rocmacro, f1micro, f1macro, acc



def getStopWordList(pathToStopWords):
    with open(pathToStopWords) as f:
        stopwords = [word.rstrip("\n ").decode("iso-8859-1") for word in f if len(word.rstrip("\n "))>0]

    return stopwords

def tokenizeFile(filepath):
    with open(filepath) as f:
    #with codecs.open(filepath, encoding='utf-8') as f:
        words = [(re.sub('[^a-zA-Z\']', ' ', line)).lower().split() for line in f]
        #print words
    #words.flatten()
    words = [val.decode("iso-8859-1") for sublist in words for val in sublist]

    return words


def removeStopWords(wordDict, stopWords):
    toRemove = []
    for (i,w) in wordDict.items():
        if w in stopWords:
            toRemove.append(i)
    wordDict.filter_tokens(bad_ids=toRemove)
    wordDict.compactify()

#this one can load all data that is sorted by folder
def loadDataSet(path):
    counter = 1;
    filesAll = []
    labelsAll = np.zeros(0)
    for dirName in filter(os.path.isdir, [path + "/" + p for p in os.listdir(path)]):
        #print dirName
        files = [dirName + "/" +filename for filename in os.listdir(dirName)]
        labels = np.zeros(len(files))
        labels[:] = counter
        filesAll.extend(files)
        labelsAll = np.append(labelsAll,labels)
        counter += 1
    return filesAll,labelsAll

#load data with multiple dataset
def loadDataWithPreDefTest(pathTrain,pathTest):
    fileTrain, labelTrain = loadDataSet(pathTrain)
    fileTest, labeltest = loadDataSet(pathTest)

    return fileTrain, labelTrain, fileTest, labeltest


#return the train and test concateneted, and the index where the test set starts
def loadDataWithTest(pathTrain,pathTest):
    fileTrain, labelTrain = loadDataSet(pathTrain)
    fileTest, labelTest = loadDataSet(pathTest)
    files = fileTrain + fileTest
    labels = np.append(labelTrain,labelTest)
    #print len(labels)
    return files, labels, len(fileTrain)

def constructAndSave(filelist,fname):
    cdict = constructDict(filelist)
    computeWordToWordMatrix(cdict,fname)

def constructDict(fileList):
    wordDict = corpora.dictionary.Dictionary()
    for file in fileList:
        doc = tokenizeFile(file)
        wordDict.add_documents([doc])

    stopWords = getStopWordList("../stopwords")
    removeStopWords(wordDict,stopWords)
    return wordDict

def constructCorpus(fileList,ourDict):
    corpus = [ourDict.doc2bow(tokenizeFile(text)) for text in fileList]
    return corpus

def computeWordToWordMatrix(myDict, fname):
    wordVectorModel = gensim.models.KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)
    wordToWord = np.eye((len(myDict)))
    for i in np.arange(len(myDict)): #(i,w) in myDict.items():
        try:
            #wx=wordVectorModel[myDict[i]]
            for j in np.arange(i): #(ii,ww) in myDict.items():
                try:
                    #wwx=wordVectorModel[myDict[j]]
                    wordToWord[i,j] =  wordVectorModel.similarity(myDict[i],myDict[j])#np.dot(wx,wwx)
                except KeyError:
                    None
                    #nothing
        except KeyError:
            wordToWord[i,i] = 1
        print i

    a_tril = np.tril(wordToWord, k=0)
    a_diag = np.diag(np.diag(wordToWord))
    wordToWord = a_tril + a_tril.T - a_diag


    np.save(fname,wordToWord)
    with open(fname+ '.pickle', 'wb') as handle:
        pickle.dump(myDict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return wordToWord

def computeBOW(files,cDict):
    BOW = np.zeros((len(files),len(cDict)))
    c=0
    for f in files:
        BOW[c,:] = densify(cDict.doc2bow(tokenizeFile(f)),len(cDict))
        c+=1
    return BOW


def computeTFIDF(files,cDict):
    IDTIF = np.zeros((len(files),len(cDict)))
    myIDTIF=gensim.models.tfidfmodel.TfidfModel(dictionary=cDict)
    c=0
    for f in files:
        IDTIF[c,:] = densify(myIDTIF[cDict.doc2bow(tokenizeFile(f))],len(cDict))
        c+=1
    return IDTIF

def densify(vec,size):
    dense = np.zeros(size)
    for (i,v) in vec:
        dense[i] = v
    return dense

def makeBOW(fileList, corpus):
    bow = []
    for file in fileList:
        doc = tokenizeFile(file)
        x = np.zeros(len(corpus))
        xbow = corpus.doc2bow(doc)
        for (wid, count) in xbow:
            x[wid] = count
        bow.append(x)
    return np.array(bow)

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

#filepath is path of data in format used in the rest of the program
#cc is current iteration of test files, start with cc=0 when calling, function returns next cc
def orderData(filepath,cc):
    #define
    testSize = 0.2
    if type(filepath) is tuple:
        (pathTrain,pathTest) = filepath
        files,labels,index = loadDataWithTest(pathTrain,pathTest)
        files = np.array(files)

        cc = 5
    else:
        files,labels = loadDataSet(filepath)
        print "load", filepath + "/" + str(cc) + 'perm.pickle'
        with open(filepath + "/" + str(cc) + 'perm.pickle', 'rb') as handle:
               currentPerm = pickle.load(handle)
               currentPerm.astype(int)
        files = np.array(files)
        files = files[currentPerm]
        labels = labels[currentPerm]
        index = np.floor(len(labels)*(1-testSize))
        cc = cc + 1
    trainFiles = files[0:index]
    testFiles = files[index:]
    trainLabels = labels[0:index]
    testLabels = labels[index:]

    return trainFiles, testFiles, trainLabels, testLabels, cc

def run_CPWE_IDF_experiment(data, w2w, myDict, pars, savename):
    print("CPWE IDF")
    (dataset,filepath) = data
    cc = 0
    accs = []
    while cc < 5:
        trainFiles, testFiles, trainLabels, testLabels, cc = orderData(filepath,cc)
        ress = runCPWE_IDF(trainFiles, testFiles, trainLabels, testLabels, w2w, myDict)
        acc, accMeans, best_k, kks,    preds, perc_preds, testLabels = ress
        accs.append(allmeasures(preds, testLabels, perc_preds))
        print accs[-1]
        with open("results/cpwe_idf" + savename + str(cc) + '_results.pkl', 'wb') as f:
                pickle.dump(ress, f)
    print "------"
    print "mean", np.mean(accs, axis=0)

def runCPWE_IDF(trainFiles, testFiles, trainLabels, testLabels, w2wO, myDict, n_neighbors=-1, split = 0.3):
    cvs = 1
    random_states = np.random.permutation(cvs)
    files = trainFiles
    pars = np.arange(0.1,1.01,0.1)
    pars[-1] = 1.0
    kks = np.concatenate((np.array([2]),np.arange(1,20,2)))
    print "(thresholds, k's)", (pars,kks)
    accs = np.zeros((len(pars),len(kks),cvs))
    BOW = (makeBOW(files, myDict))
    w2w = np.copy(w2wO)

    #get frequencies
    myIDTIF=gensim.models.tfidfmodel.TfidfModel(dictionary=myDict,normalize=False)
    badidx = -1
    for i in xrange(len(myDict)):
        e = (myIDTIF[[(i,1)]])
        if len(e) == 0:
            print(i,e,myDict[i])
            badidx = i

    freq = myIDTIF[[(i,1) for i in xrange(len(myDict))]]
    freq = [w for (i,w) in freq]
    freq = np.array(freq)

    if badidx > 0:
        real_freq = np.zeros(len(freq)+1)
        real_freq[:badidx] = freq[:badidx]
        real_freq[badidx+1:] = freq[badidx:]
        real_freq[badidx] = 0
        freq = real_freq 

    freq = 1/(np.power(2,freq)/len(files))

    print(len(myDict), len(freq))

    cc = 0
    for splitted in pars:
        kCounter = 0
        if splitted > -0.0000001:

            w2w[w2w < splitted] = 0

            w2wN = sklearn.preprocessing.normalize(w2w, axis=0)
            w2wBOW = computeW2W(w2wN,BOW)
            w2wBOW = csr_matrix(w2wBOW)
            weights = np.dot(freq, w2wN)
            weights = np.log2(len(files)/weights)
            weights = csr_matrix(diags(weights))
            w2wtfidf =  csr_matrix(w2wBOW * weights)
            w2wtfidf = sklearn.preprocessing.normalize(w2wtfidf)

            for k in kks:
                for i in xrange(0,cvs):
                    X_train, X_test, y_train, y_test = train_test_split(w2wtfidf, trainLabels, test_size=split, random_state = random_states[i])
                    clf = neighbors.KNeighborsClassifier(k,weights='distance',n_jobs=40)#, )
                    clf.fit(X_train, y_train)

                    acct = sklearn.metrics.accuracy_score(y_test, clf.predict(X_test))
                    accs[cc,kCounter,i] = acct
                kCounter = kCounter + 1

            cc = cc + 1

    accMeans = np.mean(accs,axis=2)
    (bestSplit,best_k_index) = np.unravel_index(np.argmax(accMeans),accMeans.shape)

    files = np.concatenate((trainFiles,testFiles))
    BOW = (makeBOW(files, myDict))

    splitted = pars[bestSplit]
    best_k = kks[best_k_index]
    print "best (threshold, k)", (splitted,best_k)
    w2w = np.copy(w2wO)
    w2w[w2w < splitted] = 0
    w2w = sklearn.preprocessing.normalize(w2w, axis=0)

    w2wBOW = computeW2W(w2w,BOW)
    w2wBOW = csr_matrix(w2wBOW)

    weights = np.dot(freq, w2w)
    weights = np.log2(len(files)/weights)
    weights = diags(weights)

    w2wtfidf =  (w2wBOW * weights)#[None,:]
    w2wtfidf = sklearn.preprocessing.normalize(w2wtfidf)
    w2wtfidf = w2wtfidf.todense()

    X_train = w2wtfidf[0:len(trainFiles),:]
    X_test = w2wtfidf[len(trainFiles):,:]

    clf = neighbors.KNeighborsClassifier(best_k,weights='distance',n_jobs=40)
    clf.fit(X_train,trainLabels)
    preds = clf.predict(X_test)
    acc = sklearn.metrics.accuracy_score(testLabels, preds)
    perc_preds = clf.predict_proba(X_test)

    print "#", acc
    return acc, accMeans, best_k, kks,  preds, perc_preds, testLabels


def testFunctions():
    datasets = [('twitter',"twitter")]
    for data in datasets:
        (dataset,filepath) = data
        print(dataset)
        w2w = np.load(dataset + '.npy')
        with open(dataset + '.pickle', 'rb') as handle:
            myDict = pickle.load(handle)
        run_CPWE_IDF_experiment(data,w2w,myDict,None, dataset)

if __name__ == '__main__':
    testFunctions()
