import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download("wordnet")
nltk.download("stopwords") # Will help filter out regular stopwords that do not contribute to textual information
nltk.download("punkt")

class StemLem:

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english') # List of stopwords in the English dictionary

    def stem(self, phrase):
        stemmed_words = []
        # The stemmer can only stem individual words
        # So we will tokenize the sentence or phrase first
        words = word_tokenize(phrase)
        for word in words:
            stemmed_words.append(self.stemmer.stem(word))

        return " ".join(stemmed_words)

    def lemmatize(self, phrase):
        lemmatized_words = []
        # The stemmer can only stem individual words
        # So we will tokenize the sentence or phrase first
        words = word_tokenize(phrase)
        for word in words:
            lemmatized_words.append(self.lemmatizer.lemmatize(word, pos='v')) # Need to specify the part of speech (pos)

        return " ".join(lemmatized_words)

    def removeStopwords(self, phrase):
        stripped_phrase = []
        words = word_tokenize(phrase)
        for word in words:
            if word not in self.stop_words:
                stripped_phrase.append(word)

        return " ".join(stripped_phrase)