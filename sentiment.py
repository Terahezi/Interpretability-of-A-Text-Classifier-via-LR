#!/bin/python

from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    print("-- train data")
    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    print(len(sentiment.train_data))

    print("-- dev data")
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)
    print(len(sentiment.dev_data))
    print("-- transforming data and labels")
    from sklearn.feature_extraction.text import CountVectorizer
    sentiment.count_vect = CountVectorizer()
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    tar.close()
    return sentiment

def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
    print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
            
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
    print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()
    
    
def classifier_explanation(cls, sentiment, encoded_sentence, top_n = 5):
    '''return the coefficients and features of the classifier'''
    coefficients = cls.coef_  # parameter array 1 x Num_features
    tokens_list = sentiment.count_vect.get_feature_names() # feature list
    # n = len(tokens_list)
    
    sentence = list(sentiment.count_vect.inverse_transform(encoded_sentence)[0])
        
    # assure that the input is a list
    assert isinstance(sentence, list)
    sentence_len = len(sentence)
    
    try:
        assert sentence_len != 0
    except:
        print('Nothing is input')
        
    sentence_index = []
    for i in range(sentence_len):
        sentence_index.append(tokens_list.index(sentence[i]))
    sentence_coefficient = coefficients[0, sentence_index]
    
    assert sentence_len == len(sentence_coefficient)
        
    prediction_result_index = list(cls.predict(encoded_sentence))[0]
    list_labels = list(sentiment.target_labels)
    prediction_result = list_labels[prediction_result_index]
    prob_prediction = cls.predict_proba(encoded_sentence)[0, prediction_result_index]
    
    if prediction_result_index == 0:
        do_we_reverse = False
    else:
        do_we_reverse = True
        
    if sentence_len <= top_n:
        top_coef_index = sorted(range(sentence_len), key = lambda x: sentence_coefficient[x],\
                          reverse = do_we_reverse)
        top_words = itemgetter(*top_coef_index)(sentence)
        top_coef = sorted(sentence_coefficient, reverse = do_we_reverse)
        print('Hi, humans! The text is {} with probability {} due to these top {} relevant words {} with their contribution as {}'\
              .format(prediction_result, prob_prediction, sentence_len, top_words, top_coef))
    else:
        # only print out the top top_n words
        top_coef_index = sorted(range(sentence_len), key = lambda x: sentence_coefficient[x],\
                          reverse = do_we_reverse)[:top_n + 1]
        top_words = itemgetter(*top_coef_index)(sentence)
        top_coef = sorted(sentence_coefficient, reverse = do_we_reverse)[:top_n + 1]
        print('Hi, humans! The text is {} with probability {} due to these top {} relevant words {} with their contribution as {}'\
              .format(prediction_result, prob_prediction, top_n, top_words, top_coef))
        
    print('Let\'s see it in a bar chart!')
    plt.figure
    plt.barh(top_words, top_coef)
    if do_we_reverse:
        plt.xlim(0, np.max(coefficients))
    else:
        plt.xlim(0, np.min(coefficients))
    plt.title('Weights assigned among different features')
    plt.ylabel('token')
    plt.xlabel('weight')
        

if __name__ == "__main__":
    print("Reading data")
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    print("\nTraining classifier")
    import classify
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, C = 0.625)
    
    # make prediction on a new sample and then try to make explanations
    index = 1  # randomly select an index
    encoded_sentence = sentiment.trainX[index, :]
    # sentence = sentiment.train_data[index].split()
    
    classifier_explanation(cls, sentiment, encoded_sentence, top_n = 10)
    

