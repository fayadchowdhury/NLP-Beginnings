from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

class SVMTextClassifier:

    def __init__(self, trainx, trainy):
        # vectorizer = CountVectorizer() # Non-binary (counts the number of occurrences of the word)
        # vectorizer = CountVectorizer(binary=True) # Binary (checks if the word existed or not)
        # vectorizer = CountVectorizer(ngram_range=(x,y)) # Takes between x and y words in a sequence
        self.vectorizer = CountVectorizer()
        self.clf = svm.SVC(kernel='linear') # Galli claims that linear kernels work better for text classification from prior experience
        self.trainx = trainx
        self.trainy = trainy

    def vectorize(self, text):
        vectors = self.vectorizer.fit_transform(text) # Fits a dictionary to the text
        return [vectors, self.vectorizer.get_feature_names()] # Returns the vectorized version of the text sequences and the feature names (words being searched for)

    def train(self):
        trainxVec, feature_names = self.vectorize(self.trainx)
        # print(trainxVec.toarray())
        # print(feature_names)
        self.clf.fit(trainxVec, self.trainy) # We fit the classifier to the training data vector and the corresponding training labels

    def predict(self, testX):
        result = self.clf.predict(self.vectorizer.transform(testX))
        print(result)