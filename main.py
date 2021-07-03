# We're going to try OOP with Python and call all our functions from other classes in this main file
import numpy as np
import spacy

#COUNT VECTORIZER FOR BOW AND SVM
import SVMTextClassifier as svmtc

#WORD VECTOR
import WordVectors as wv

#Regular Expressions
import re

if __name__ == '__main__':
    trainx = np.array(['I love the book', 'this is a great book', 'the fit is great', 'i love the shoes'])
    trainy = np.array(['book', 'book', 'clothing', 'clothing'])
    testx = np.array(['I love stories'])

    regexp = re.compile(r"ab[^\s]*cd")

    # print(re.match(regexp, "abcd")) # Matches strictly to expression pattern, None if no match found, otherwise a match object
    print(re.search(regexp, "aa ab cd dd")) # Does something, I really need to understand regex-es better

    # SpaCy tidbits
    # nlp = spacy.load("en_core_web_md") #Load up
    # trainSample1 = "cat"
    # trainSample2 = "dog"
    # trainSample3 = "I have a cat"
    # trainSample4 = "I have a dog"
    # trainSample5 = "I have a cat and a dog"
    #
    # testdoc = nlp(trainSample5)
    # print(len(testdoc.vector))
    # print(testdoc)
    # print(testdoc.vector)
    #
    # testdoc1 = nlp(trainSample3)
    # testdoc2 = nlp(trainSample5)
    # print(testdoc1.similarity(testdoc2))

    # svm_clf = svmtc.SVMTextClassifier(trainx=trainx, trainy=trainy)
    # svm_clf.train()
    # svm_clf.predict(testx)

    # wordvec = wv.WordVectors(trainx, trainy)
    # wordvec.train()
    # wordvec.predict(testx)