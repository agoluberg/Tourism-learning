
def output_fscore(true_positive, false_positive, false_negative):
    tp = true_positive
    fp = false_positive
    fn = false_negative
    print get_fscore_output(tp, fp, fn)
    return True

    print 'tp', tp
    print 'fp', fp
    print 'fn', fn
    precision = (float(tp) / (tp + fp))
    print 'Precision=', precision
    
    # recall = tp / (tp + fn)
    recall = (float(tp) / (tp + fn))
    print 'Recall=', recall
    
    # f-measure = 2 * precision * recall / (precision + recall)
    print 'F-score', (2 * precision * recall / (precision + recall))
    
def get_fscore_output(tp, fp, fn, train_pos=None, train_neg=None, test_pos=None, test_neg=None, data=None):
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = (float(tp) / (tp + fp))
    recall = (float(tp) / (tp + fn))
    if precision + recall == 0:
        fscore = 0.0
    else:
        fscore = (2 * precision * recall / (precision + recall))
    s = ''
    s += 'tp=' + str(tp) + '\n'
    s += 'fp=' + str(fp) + '\n'
    s += 'fn=' + str(fn) + '\n'
    s += 'precision=' + str(precision) + '\n'
    s += 'recall=' + str(recall) + '\n'
    s += 'f-score=' + str(fscore) + '\n'
    if train_pos is not None:
        s += 'train_pos=' + str(train_pos) + '\n'
    if train_neg is not None:
        s += 'train_neg=' + str(train_neg) + '\n'
    if test_pos is not None:
        s += 'test_pos=' + str(test_pos) + '\n'
    if test_neg is not None:
        s += 'test_neg=' + str(test_neg) + '\n'
    if isinstance(data, dict):
        for k, v in data.iteritems():
            s += str(k) + '=' + str(v) + '\n'
    
    return s