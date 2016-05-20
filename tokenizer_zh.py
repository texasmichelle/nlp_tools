#!/usr/bin/python
# -*- coding: utf-8 -*-

import math

encoding = 'big5hkscs'

def tokenize(text, lang_code):
    # Until this section is implemented, these tests will fail
    #>>> for s in removeWhitespace(getLinesFromFile("resources/training_small.txt")):
    #    ...     for t in tokenize(s.encode(encoding), "zh"):
    #    ...         print t
    #['時間', '：']
    #['三月', '十日', '（', '星期四', '）', '上午', '十時', '。']
    """
    >>> tokenize("ned flanders", "en")
    ['ned', 'flanders']
    """

    if (lang_code == "en"):
        return text.split()
    elif (lang_code == "zh"):
        return text.decode(encoding).strip().split()

def getLinesFromFile(filename):
    # Removing doctests until I understand the encoding discrepancies better
    '''
    >>> for s in getLinesFromFile("resources/training_10.txt"):
    ...     print s.strip().encode('utf8')
    時間  ：
    三月  十日  （  星期四  ）  上午  十時  。
    地點  ：
    學術  活動  中心  一樓  簡報室  。
    主講  ：
    民族所  所長  莊英章  先生  。
    講題  ：
    閩  、  台  漢人  社會  研究  的  若干  考察  。
    李  院長  於  二月  二十六日  至  三月  十五日  赴  美  訪問  ，
    期間  將  與  在  美  院士  商討  院務  ，
    '''

    with open(filename) as fp:
        lines, errors = [], []
        for i, encoded_line in enumerate(fp):
            try:
                lines.append(encoded_line.decode(encoding).strip())
            except UnicodeDecodeError as e:
                errors.append('line %d: %s' % (i, e))

    return lines

def removeWhitespace(training):
    '''
    >>> for s in removeWhitespace(["時間  ：", "三月  十日  （  星期四  ）  上午  十時  。"]):
    ...     print s
    時間：
    三月十日（星期四）上午十時。
    '''

    test = []
    for s in training:
        test.append(s.replace(" ", ""))
    return test

def countNgrams(text, window_size = 4):
    """
    Using a sliding window of 1- and 2-character sequences, construct probability counts for each
    of the ngrams we need statistics for:

      WXB XB XBY BY BYZ

    :param text: list of strings with segments
    :param window_size: number of characters to separate into unigrams & bigrams. Default is 4.
    :return: a dict whose keys are ngrams and values their prior probability in the provided training set
    """

    # keys: ngram, including a space representing a boundary
    # values: [char_boundary_count, word_boundary_count]
    ngram_stats = {}

    # use a library for this?

    # First sequence of window_size characters
    for line in text:
        # Toss if the character count is too low
        #line_wo_whitespace = line.replace(" ", "")
        #if len(line_wo_whitespace) < window_size:
        if len(line.replace(" ", "")) < window_size:
            continue

        # i is the beginning index of our window. In other words, how far through the line we have progressed
        i = 0
        while len(line) > window_size + i:
            # this is where we store our characters
            window = []
            # j keeps track of how far to the right we had to search to find window_size characters
            j = 0

            # ignore windows that begin with whitespace. Since we discard whitespace when building our
            # window, we end up with the same window multiple times if we don't do this.
            if line[i] == " ":
                i += 1
                continue

            while len(window) < window_size and len(line) > i + j:
                # ignore whitespace
                if line[i + j] != " ":
                    window.append(line[i + j])
                j += 1

            if len(window) == window_size:
                # Identify our ngrams
                ngrams = []
                # bigram WXB
                ngrams.append("".join(window[0:2]) + " ")
                # unigram XB
                ngrams.append(window[1] + " ")
                # bigram XBY
                ngrams.append(window[1] + " " + window[2])
                # unigram BY
                ngrams.append(" " + window[2])
                # bigram BYZ
                ngrams.append(" " + "".join(window[2:4]))

                #print "ngrams:", ngrams

                for n in ngrams:
                    # if it doesn't exist
                    ngram_stats += (n <- [0, 0])

            i += 1

    print "ngram_stats:", ngram_stats

    # Now that we have a dict with all the ngrams we've seen, it's time to start counting

    return ngram_stats

def featurize(text, window_size, ngram_stats):
    """
    Using a sliding window, look at pairs of characters and their immediate neighbors.
    Construct a feature vector of 5 ngrams with a label. The composition of those ngrams
    is as follows:

      # parameterize the char window
      For consecutive characters <X,Y> in a 4-character window WXYZ, look at unigrams X and Y
      and bigrams WX, XY, YZ. Boundaries are: WXB XB XBY BY BYZ

    These 5 ngrams constitute the feature dimensions. Their value represents the probability of that
    interval being a word boundary. Those values are calculated in countNgrams and used here.

    :param text: list of strings to be segmented
    :return: list of feature vectors
    """

    features = []

    # First sequence of window_size characters
    for line in text:
        # i is the beginning index of our window. In other words, how far through the line we have progressed
        i = 0
        # Toss it if the character count is too low
        while len(line) > window_size + i:
            #window = line[i:i+window_size]

            # this is where we store our characters
            window = []
            # j keeps track of how far to the right we had to search to find window_size characters
            j = 0
            while len(window) < window_size and (len(line) > i + j):
                # ignore whitespace
                if line[i + j] != "\s":
                    window.append(line[i + j])
                j += 1

            # now that we have our chars, let's keep track of them in two places.
            # In a dict, for calculating statistics


            i += 1

    return features

def train(training, window_size, ngram_stats):
    """
    Perform logistic regression on the provided feature vector

    :param training: a list of labeled feature vectors
    :return: the resulting model to be used for prediction
    """
    # Get the data ready for training
    features = featurize(training, window_size, ngram_stats)

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
    text_with_whitespace = getLinesFromFile("resources/training_10.txt")
    # Whitespace removed
    text_no_whitespace = removeWhitespace(text_with_whitespace)
    print "with: ",  text_with_whitespace
    print "without: ", text_no_whitespace

    # Split the dataset into two pieces (60/40 or 70/30)
    # Really dumb split, need a better way
    boundary = int(math.floor(len(text_with_whitespace) * .6))
    training = text_with_whitespace[:boundary]
    test = text_no_whitespace[boundary:]

    # Build an ngram probability map
    window_size = 4
    ngram_stats = countNgrams(training, window_size)

    # Train a classifier
    model = train(training, window_size, ngram_stats)

    # Predict on the test set
    predictions = predict(test, model)

    # Evaluate the results
    # Can we do xval somehow with scikitlearn?




if __name__ == "__main__":
    import doctest
    doctest.testmod()

    main()

