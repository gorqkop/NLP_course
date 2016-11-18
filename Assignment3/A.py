from main import replace_accented
from main import parse_data
from sklearn import svm
from sklearn import neighbors
import nltk, collections, numpy
# don't change the window size
window_size = 10
# A.1
def build_s(data):
    '''
    Compute the context vector for each lexelt
    :param data: dic with the following structure:
        {
			lexelt: [(instance_id, left_context, head, right_context, sense_id), ...],
			...
        }
    :return: dic s with the following structure:
        {
			lexelt: [w1,w2,w3, ...],
			...
        }

    '''
    s = {}
    for lexelt in data:
        out = set()
        for observation in data[lexelt]:
            out.update(nltk.word_tokenize(observation[1])[-window_size:])
            out.update(nltk.word_tokenize(observation[3])[:window_size])
        s[lexelt]=list(out)

    # implement your code here
    return s


# A.1
def vectorize(data, s):
    '''
    :param data: list of instances for a given lexelt with the following structure:
        {
			[(instance_id, left_context, head, right_context, sense_id), ...]
        }
    :param s: list of words (features) for a given lexelt: [w1,w2,w3, ...]
    :return: vectors: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }
            labels: A dictionary with the following structure
            { instance_id : sense_id }

    '''
    vectors = {}
    labels = {}
    # implement your code here
    for obsr in data:
        out = []
        labels[obsr[0]]=obsr[-1]
        out.extend(nltk.word_tokenize(obsr[1])[-window_size:])
        out.extend(nltk.word_tokenize(obsr[3])[:window_size])
        dat = collections.Counter()
        dat.update(out)
        vectors[obsr[0]]=[dat[x] for x in s]
    return vectors, labels


# A.2
def classify(X_train, X_test, y_train):
    '''
    Train two classifiers on (X_train, and y_train) then predict X_test labels

    :param X_train: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param X_test: A dictionary with the following structure
            { instance_id: [w_1 count, w_2 count, ...],
            ...
            }

    :param y_train: A dictionary with the following structure
            { instance_id : sense_id }

    :return: svm_results: a list of tuples (instance_id, label) where labels are predicted by LinearSVC
             knn_results: a list of tuples (instance_id, label) where labels are predicted by KNeighborsClassifier
    '''
    # print X_train
    # print X_test

    svm_clf = svm.LinearSVC()
    knn_clf = neighbors.KNeighborsClassifier()
    X_f, X_c = [], []
    [(X_c.append(y_train[i[0]]), X_f.append(i[1])) for i in X_train.items()]

    print "Start train classifyers"
    svm_clf.fit(X_f, X_c)
    knn_clf.fit(X_f, X_c)
    print "Finish train classifyers"
    svm_results = [(i[0], svm_clf.predict(i[1])) for i in X_test.items()]
    knn_results = [(i[0], knn_clf.predict(i[1])) for i in X_test.items()]

    # T_f, T_c = [], []
    # [(T_c.append(x[0]), T_f.append(x[1])) for x in X_test]
    # svm_results = svm_clf.predict(T_f)
    # svm_results = [(T_c[i], svm_results[i]) for i in range(len(svm_results))]
    # knn_results = svm_clf.predict(T_f)
    # knn_results = [(T_c[i], knn_results[i]) for i in range(len(knn_results))]

    return svm_results, knn_results

# A.3, A.4 output
def print_results(results ,output_file):
    '''

    :param results: A dictionary with key = lexelt and value = a list of tuples (instance_id, label)
    :param output_file: file to write output

    '''

    # implement your code here
    # don't forget to remove the accent of characters using main.replace_accented(input_str)
    # you should sort results on instance_id before printing
    for lexelt, val in results.items():
        val = sorted(val, key=lambda x: x[0])
        with open(output_file, 'a') as out:
            for x in val:
                #print x[1][0]
                out.write('%s %s %s\n'%(replace_accented(lexelt), replace_accented(x[0]), x[1][0]))
# (u'begin.v.bnc.00001099', array(['369201'], dtype='|S6'))
# run part A
def run(train, test, language):#, knn_file, svm_file):
    s = build_s(train)
    svm_results = {}
    knn_results = {}
    for lexelt in s:
        X_train, y_train = vectorize(train[lexelt], s[lexelt])
        X_test, _ = vectorize(test[lexelt], s[lexelt])
        svm_results[lexelt], knn_results[lexelt] = classify(X_train, X_test, y_train)
        #print svm_results

    # print_results(svm_results, svm_file)
    # print_results(knn_results, knn_file)
    print_results(svm_results, 'SVM-%s.answer'%language)
    print_results(knn_results, 'KNN-%s.answer'%language)

# data_tr = parse_data('data/Catalan-train.xml')
# data_tst = parse_data('data/Catalan-dev.xml')
# run(data_tr, data_tst, 'Catalan')
