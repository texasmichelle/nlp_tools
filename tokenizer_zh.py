#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

def tokenize(text, lang_code):
    """
    >>> tokenize("ned flanders", "en")
    ['ned', 'flanders']
    """
    if (lang_code == "en"):
        return text.split()
    elif (lang_code == "zh"):
        return text.decode('gb18030').rstrip().encode('utf8').split()

def getListFromFile(filename):
    # Removing doctests until I understand the encoding discrepancies better
    #>>> getListFromFile("resources/training_small.txt")
    #丁  
    #る  ら    琍戳    と    

    training = []
    with open(filename) as f:
        for line in f:
            decoded_string = line.decode('gb18030').rstrip().encode('utf8')
            print decoded_string
            training.append(decoded_string)
        f.close()
    return training

def removeWhitespace(training):
    # Removing doctests until I understand the encoding discrepancies better
    #>>> removeWhitespace(["丁  ", "る  ら    琍戳    と    "])
    #['丁', 'るら琍戳と']

    test = []
    for s in training:
        test.append(s.replace(" ", ""))
    return test

def countNgrams(training):
    """
    Using a sliding window of 1- and 2-character sequences, construct probability counts for each
    of the ngrams we need statistics for:

      WXB XB XBY BY BYZ

    :param training: list of strings with segments
    :return: a dict whose keys are ngrams and values their prior probability in the provided training set
    """

    ngram_stats = {}

    return ngram_stats

def featurize(text, ngram_stats):
    """
    Using a sliding window, look at pairs of characters and their immediate neighbors.
    Construct a feature vector of 5 ngrams with a label. The composition of those ngrams
    is as follows:

      For consecutive characters <X,Y> in a 4-character window WXYZ, look at unigrams X and Y
      and bigrams WX, XY, YZ. Boundaries are: WXB XB XBY BY BYZ

    These 5 ngrams constitute the feature dimensions. Their value represents the probability of that
    interval being a word boundary. Those values are calculated in countNgrams and used here.

    :param text: list of strings to be segmented
    :return: list of feature vectors
    """

    features = []

    return features

def train(training, ngram_stats):
    """
    Perform logistic regression on the provided feature vector

    :param training: a list of labeled feature vectors
    :return: the resulting model to be used for prediction
    """
    # Get the data ready for training
    features = featurize(training, ngram_stats)

    # Perform logistic regression

    # Return the resulting model
    return {}

def predict(data_set, model):
    """

    :return predictions: a list of classifications corresponding with each piece of text
    in the provided file
    """
    predictions = []

    return predictions

def main():
    # Scoop out the training data
    # Whitespace included
    text_with_whitespace = getListFromFile("resources/training_small.txt")
    # Whitespace removed
    text_no_whitespace = removeWhitespace(text_with_whitespace)
    print text_with_whitespace
    print text_no_whitespace

    # Split the dataset into two pieces (60/40 or 70/30)
    # Really dumb split, need a better way
    boundary = math.floor(len(text_with_whitespace) * .6)
    training = text_with_whitespace[:boundary]
    test = text_no_whitespace[boundary:]

    # Build an ngram probability map
    ngram_stats = countNgrams(training)

    # Train a classifier
    model = train(training)

    # Predict on the test set
    predictions = predict(test)

    # Evaluate the results
    # Can we do xval somehow with scipy?




if __name__ == "__main__":
    import doctest
    doctest.testmod()

    main()

