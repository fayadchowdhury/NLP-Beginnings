# We're going to try OOP with Python and call all our functions from other classes in this main file
import numpy as np
import spacy

#COUNT VECTORIZER FOR BOW AND SVM
import SVMTextClassifier as svmtc

#WORD VECTOR
import WordVectors as wv

#Regular Expressions
import re

#Stemming and lemmatization
import StemLem

#TextBlob GOODNESS!
import coolProcessing as cp

if __name__ == '__main__':
    trainx = np.array(['I love the book', 'this is a great book', 'the fit is great', 'i love the shoes'])
    trainy = np.array(['book', 'book', 'clothing', 'clothing'])
    testx = np.array(['I love stories'])
    big_text = 'As he crossed toward the pharmacy at the corner he involuntarily turned his head because of a burst of light that had ricocheted from his temple, and saw, with that quick smile with which we greet a rainbow or a rose, a blindingly white parallelogram of sky being unloaded from the van—a dresser with mirrors across which, as across a cinema screen, passed a flawlessly clear reflection of boughs sliding and swaying not arboreally, but with a human vacillation, produced by the nature of those who were carrying this sky, these boughs, this gliding façade.'
    wanky_happy_text = 'I really luv doggos!'
    wanky_sad_text = 'Classes suck! they are horribol!'

    # regexp = re.compile(r"ab[^\s]*cd")

    # print(re.match(regexp, "abcd")) # Matches strictly to expression pattern, None if no match found, otherwise a match object
    # print(re.search(regexp, "aa ab cd dd")) # Does something, I really need to understand regex-es better

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

    # print(len(big_text))
    # sl = StemLem.StemLem()
    # stemmed_text = sl.stem(big_text)
    # print("Stemmed text")
    # print(stemmed_text)
    # lemmatized_text = sl.lemmatize(big_text)
    # print("Lemmatized text")
    # print(lemmatized_text)
    # stopped_text = sl.removeStopwords(big_text)
    # print("Stopped text")
    # print(stopped_text)
    # stemmed_stopped_text = sl.stem(stopped_text)
    # print("Stemmed stopped text")
    # print(stemmed_stopped_text)
    # lemmatized_stopped_text = sl.lemmatize(stopped_text)
    # print("Lemmatized stopped text")
    # print(lemmatized_stopped_text)

    cp_happy = cp.coolProcessing(wanky_happy_text)
    cp_sad = cp.coolProcessing(wanky_sad_text)

    print(cp_happy.correctSpelling()) # So close
    print(cp_happy.sentiment()) # A high positive polarity score = good and vice versa
    print(cp_happy.posTags()) # Need to understand the tags a little better

    print(cp_sad.correctSpelling()) # Stays fairly wanky
    print(cp_sad.sentiment()) # Cannot detect sentiment (y tho?)
    print(cp_sad.posTags())

