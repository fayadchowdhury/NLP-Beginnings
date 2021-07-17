# NLP-Beginnings
This is where I'm going to stash my NLP knowledge and code that I gather from scouring the internet

##Bag-of-words

###Link: https://machinelearningmastery.com/gentle-introduction-bag-words-model/

Take all the different words in some text and then check sentence by sentence or text sequence by text sequence to check if the words are there (1; or count depending on binary or non-binary nature of count vectorizer) or not (0)
The output is a vector representation of the text sequences

First we FIT a dictionary to the training text and then we TRANSFORM the testing or input text to classify into the BOW representation

Sometimes, n-gram models help capture semantics better than unigram models

Limitation: If the word does not occur in the training dictionary, the model will fail

Extension -> We tried to classify sentences using an SVM (support vector machine; supervised learning model). We trained (or "fitted") it on the BOW representation of some input text and then we tried to classify some test text.

The code is mainly found in SVMTextxClassifier.py


##Random tidbits:

###Link: https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

###Count vectorisation 
Count vectorisation is a frequency based embedding of words. It counts the number of occurrences or whether a word is present or not across all documents in a corpus. The columns become the vector representations.


###TF-IDF
TF-IDF is term-frequency-inverse-document-frequency. It penalises and reduces the importance of commonly occurring words like 'is', 'are', 'the' etc. and giving more importance to more important words.

TF = no. of occurrences of that word in the document/total no. of words in the document.

IDF = log(no. of documents/no. of documents that the word has appeared in) (if a word has appeared in a lot of documents, log(~1) ~0 and vice versa).

TF-IDF score is the multiplication of the above two.


###Co-occurrence matrix

For V unique words, a V*V matrix is formed and a sliding window is slid across each term and then the occurrences of each other word is seen in the context window neighbourhood of another.
Once the matrix is formed, matrix decomposition techniques like PCA, SVD etc. are used to decompose into factors which are combined differently to form the vectors.
It preserves the semantic relationship between words.


##Word vectors:

Convert words into vectors to capture semantic meaning.
Pretrained embeddings are downloaded and words in text sequences are assigned these embeddings by default.
The entire sentence or document is then an average of the individual word vectors.

Then we use a regular linear SVM classifier to train (first fit, then predict) to predict the topic (??)

Drawbacks: If there are multiple words in the same sentence, the importance of one word may be gradually lost.
Also since word embeddings are the same, the same word used in different places to imply different meanings may not be picked up accurately by these word embeddings


##Regexes:

###Link: https://regexr.com/

Regular expressions to figure out text string patterns.

Can be used to check emails, phone numbers, passwords etc.

PUT IN A LOT MORE WORK HERE!!!


##Stemming/Lemmatization:

###Link: https://www.nltk.org/
Both are techniques to normalize text

reading -> read

books -> book

stories -> stori (stemming) or story (lemmatization)

Stemming does not guarantee the formation of a legiimate English word, lemmatization checks in the English dictionary and returns an actual English word.
This is a drawback of stemming and could sometimes lead to conflicts between two words that don't exist.
Also, punctuation gets treated as its own token.

Stopwords is the list of the most common words in the English dictionary that do not add meaning to the sentence.
They need to be removed to better capture the meaning and semantics of the sentence.

We use NLTK for these things.