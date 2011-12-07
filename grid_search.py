from numpy.lib.shape_base import vstack, hstack
from optparse import OptionParser
from result_util import get_fscore_output
import datetime
import pickle
from common import FILES_PATH, RESULT_FILES_PATH
import StringIO
import sys
print __doc__

from pprint import pprint
import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file
from numpy.core.numeric import array

class ob:
    def __init__(self):
        self.stream = StringIO.StringIO()
    def write(self, data):
        if not isinstance(data, unicode):
            data = data.decode('utf-8')
        self.stream.write(data.encode('utf-8'))
        self.stream.flush()
    def __getattr(self, attr):
        return getattr(self.stream, attr)
    def getvalue(self):
        return self.stream.getvalue()
    def flush(self):
        self.write(self.getvalue())

def load_data(filename, nparray=True, feature_list=None):
    """
    Returns (features [m x n], labels [m x 1])

    filename - filename to parse
    nparray - return numpy array
    feature_list - a list of feature numbers (which are used in the file) to load. If None, all the features are loaded
    """
    X = []
    Y = []
    i = 0
    f = open(filename, 'r')
    if len(feature_list) == 0: feature_list = None
    for l in f:
        i += 1
        tokens = l.split(' ')
        curx = []
        if len(tokens) > 1:
            Y.append(int(tokens[0]))
            for t in tokens[1:]:
                tokens2 = t.split(':')
                if len(tokens2) == 2:
                    if len(feature_list) and float(tokens2[0]) not in feature_list:
                        # we skip this feature 
                        continue
                    x = float(tokens2[1])
                else:
                    x = 0.0
                curx.append(x)
        X.append(curx)
    f.close()
    if nparray:
        return (array(X), array(Y))
    return (X, Y)

global_time = datetime.datetime.today()
def log(msg):
    diff = datetime.datetime.today() - global_time
    print datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'), diff, msg
################################################################################
# Loading the Digits dataset
#digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

op = OptionParser()

op.add_option('-p',
              '--prefix',
              help='training and test file prefix (without .train/.test)',
              dest='prefix',
              default=None)
op.add_option('-d',
              '--data',
              help='training and test file (will be divided into training/validation/test sets',
              dest='data',
              default=None)
op.add_option('-r',
              '--results',
              help='results file. if omitted, data filename + date will be used',
              dest='results',
              default=None)
op.add_option('-f',
              '--features',
              help='which features to use (use the number in the file)',
              dest='features',
              default=None)

(options, args) = op.parse_args()

prefix = 'tallinn_200s_title_rdist_1000'
prefix = 'data/tallinn_111_titlewc_rdist_10000'
prefix = 'data/tallinn_202s_edistsim_rdist_10000'
prefix = 'data/tallinn_200ss_title_rdist_1000'

bln_write_model = True

testX = None

feature_list = []
if options.features:
    print options.features
    feature_list = map(lambda s : int(s), options.features.split(','))
    print feature_list
    
if options.prefix:
    prefix = options.prefix
    
    #prefix = 'data/tallinn_201s_title_rdist_10000'
    X, y = load_data(prefix + '.train', feature_list=feature_list)
    #print y.shape
    #X, y = load_svmlight_file(prefix + '.train')
    testX, testY = load_data(prefix + '.test', feature_list=feature_list)
    
    X = vstack((X, testX))
    X = preprocessing.scale(X)
    print X

    #y = vstack((y, testY))
    #y = y + testY
    #y = y.transpose()
    #print y.shape, testY.shape
    y = hstack((y, testY))
    print y

if options.data:
    log('loading ' + str(options.data))
    X, y = load_data(options.data, feature_list=feature_list)
    if False:
        pos_start = 0
        pos_end = 1500
        neg_start = 2000
        neg_end = 1000000
        # tmp
        tmp = X[pos_start:pos_end, :]
        tmp = vstack((tmp, X[neg_start:neg_end]))
        X = tmp
        
        tmp = y[pos_start:pos_end]
        tmp = hstack((tmp, y[neg_start:neg_end]))
        y = tmp
    #X = X[0:2000, :]
    #y = y[0:2000]
    
#print X
#print y
#exit()
#pprint(np.array(X))
#pprint(y)
# split the dataset in two equal part respecting label proportions
train, test = iter(StratifiedKFold(y, 2)).next()

print X[train].shape, y[train].shape
log('start training')
################################################################################
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = [
    #('precision', precision_score),
    #('recall', recall_score),
    ('f1_score', f1_score),
]

for score_name, score_func in scores:
    clf = GridSearchCV(SVC(C=1, cache_size=100), tuned_parameters, score_func=score_func, verbose=2, n_jobs=8)
    clf.fit(X[train], y[train], cv=StratifiedKFold(y[train], 5))
    y_true, y_pred = y[test], clf.predict(X[test])

    sys.stdout = ob()
    log('score_name '+ str(score_name))
    print "Classification report for the best estimator: "
    print clf.best_estimator
    print "Tuned for '%s' with optimal value: %0.3f" % (
        score_name, score_func(y_true, y_pred))
    print classification_report(y_true, y_pred)
    print "Grid scores:"
    pprint(clf.grid_scores_)
    print

    if testX is not None:
        # let's use clf:
        predict_result = clf.predict(testX)
        
        i = 0
        tp = 0
        fp = 0
        fn = 0
        for p in predict_result:
            if p == testY[i]:
                if p == 1:
                    tp += 1
            else:
                # p != Y
                if p == 1:
                    fp += 1
                else:
                    fn += 1
            i += 1
        s = get_fscore_output(tp, fp, fn)
        print s
        print
    
    """
    Let's test the whole set (training + test)
    """
    predict_result = clf.predict(X)
    
    i = 0
    tp = 0
    fp = 0
    fn = 0
    for p in predict_result:
        if p == y[i]:
            if p == 1:
                tp += 1
        else:
            # p != Y
            if p == 1:
                fp += 1
            else:
                fn += 1
        i += 1
    s = get_fscore_output(tp, fp, fn)
    print s
    
    str_output = sys.stdout.getvalue()
    sys.stdout = sys.__stdout__
    
    print str_output
    
    
    if bln_write_model:
        strtime = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        mname = 'x'
        if options.prefix:
            mname = options.prefix
        elif options.data:
            dotpos = options.data.find('.')
            if dotpos > 0:
                mname = options.data[:dotpos]
        slashpos = mname.rfind('/')
        if slashpos > 0:
            mname = mname[slashpos+1:]
        fname = 'model_' + mname + '_' + str(strtime) + '.pickle'
        pickle.dump(clf.best_estimator, open(FILES_PATH + '/' + fname, 'w'))
        
        # results
        if options.results:
            fname = options.results
        else:
            fname = RESULT_FILES_PATH + '/' + 'results_' + mname + '_' + str(strtime) + '.results'
        fresults = open(fname, 'w')
        fresults.write(str_output)
        fresults.close()
    
    #return (tp, fp, fn, predict_result, clf)

log('done')
