#!/usr/bin/python
# -*- coding: utf-8 -*-

def tokenize(text, lang_code):
    """
    >>> tokenize("ned flanders", "en")
    ['ned', 'flanders']
    >>> with open("resources/training_small.txt") as f:
    ...    for line in f:
    ...        tokenize(line, "zh")
    ['\xee\x86\xa0\xe4\xb8\x81', '\xee\x93\x8d']
    ['\xee\x97\xba\xe3\x82\x8b', '\xee\x97\xb7\xe3\x82\x89', '\xee\x93\xa3', '\xe7\x90\x8d\xe6\x88\xb3\xee\x9a\x82', '\xee\x93\xa4', '\xee\x97\xbd\xe3\x81\xa8', '\xee\x97\xb7\xee\x86\xa0', '\xee\x93\x89']
    """
    if (lang_code == "en"):
        return text.split()
    elif (lang_code == "zh"):
        return text.decode('gb18030').rstrip().encode('utf8').split()

def processFile(filename):
    """
    >>> processFile("resources/training_small.txt")
    丁  
    る  ら    琍戳    と    

    """
    with open(filename) as f:
        for line in f:
            print line.decode('gb18030').rstrip().encode('utf8')
    f.close()

def main():
    print tokenize("something else", "zh")
    processFile("resources/training_small.txt")

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    main()

