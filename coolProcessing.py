from textblob import TextBlob

class coolProcessing:

    def __init__(self, phrase):
        self.phrase = TextBlob(phrase)

    def correctSpelling(self):
        return self.phrase.correct()

    def posTags(self):
        return self.phrase.tags

    def sentiment(self):
        return self.phrase.sentiment