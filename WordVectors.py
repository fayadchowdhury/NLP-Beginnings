import spacy
from sklearn import svm

class WordVectors:
    def __init__(self, text, labels):
        self.embedder = spacy.load("en_core_web_md")
        self.text = text
        self.clf = svm.SVC(kernel='linear')
        self.labels = labels

    def embed(self):
        self.embeddings = [self.embedder(str(text)) for text in self.text]
        return self.embeddings

    def train(self):
        trainEmbeddings= self.embed()
        trainVectors = [x.vector for x in trainEmbeddings]
        self.clf.fit(trainVectors, self.labels)

    def predict(self, testx):
        testEmbedding = [self.embedder(str(text)) for text in testx]
        # testEmbedding = self.embedder(testx)
        testVectors = [x.vector for x in testEmbedding]
        # testVectors = testEmbedding.vector
        result = self.clf.predict(testVectors)
        print(result)

    # def printVec(self):
    #     for i in range(len(self.embeddings.text)):
    #         print(self.embeddings.text[i])
    #         print(self.embeddings[i].vector[:30])
