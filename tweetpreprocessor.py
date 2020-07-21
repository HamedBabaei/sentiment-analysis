import preprocessor as tweet_preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet

class TweetPreprocessing:
    def __init__(self, lowerize = True, stemming=True, stopword_removal=True, 
                 punch_removal=True, emoji_removal=True, tweet_preprocessor=True):
        """
        initialize variables, and set preprocessing configuration properly.
        lowerize: for lowerizing text
        stemming: doing stemming of the words
        stopword_removal: an option for removing stopwords
        punch_removal: an option for removing punchuations
        tweet_preprocessor: an opton for doing twitter preprocessing for
                            mention, url, hashtags and ... removal
        emoji: an option for removing emojis in tweets
        """
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words('english'))
        self.table = str.maketrans('', '', string.punctuation)
        self.tokenizer = TweetTokenizer()        
        self.configuration = { 
                          1: {"permission":lowerize,           "method":self._lowerize},
                          2: {"permission":emoji_removal,      "method":self._emoji_removal,},
                          3: {"permission":tweet_preprocessor, "method":self._tweet_preprocessor},
                          4: {"permission":True,               "method":self._tokenize},
                          5: {"permission":punch_removal,      "method":self._punch_removal,},
                          6: {"permission":stopword_removal,   "method":self._stopword_removal,},
                          7: {"permission":stemming,           "method":self._stemming,}
                          }

    def clean(self, X, return_type="list", verbose=False):
        """
        X: is a sentence or document, or ... which are only plain text not dict or list
        return_type: 'list', 'text' ::: default is list of words
        """
        if verbose:
            print("ORG:", X)
        if len(X) > 1:
            for config_no in range(1, len(self.configuration)+1):
                X = self.configuration[config_no]['method'](X, self.configuration[config_no]['permission'])

            if verbose:
                print("CLEAN:", self._combine_words(X))
                print("...................")

            return  X if return_type=='list' else self._combine_words(X)
        else:
            return X
    
    def _combine_words(self, X):
        return ' '.join(X).strip()
    
    def _lowerize(self, X, permission):
        """input/output is only plain text"""
        return X.lower() if permission else X
    
    def _tweet_preprocessor(self, X, permission):
        """input/output is only plain text"""
        return tweet_preprocessing.clean(X)  if permission else X

    def _tokenize(self, X, permission):
        """
        input: is only plain text
        output: is list of words (tokens)
        """
        return self.tokenizer.tokenize(X) if permission else X

    def _punch_removal(self, X, permission):
        """
        input: a list of words (tokens)
        output: a list of words without any punchuations
        """
        """input/output is only plain text"""
        if not permission:
            return X
        chars_lsts = """`~!@#$%^&*()+=}{|\'";:><[]*,.:"_"""
        words = ' '.join(X)
        for char in chars_lsts:
            words = words.replace(char, "")
        return words.split()
    
    def _emoji_removal(self, X, permission):
        """
        input: a text
        output: a text without any emoji
        """
        return X.encode('ascii', 'ignore').decode('ascii') if permission else X

    def _stopword_removal(self, X, permission):
        """
        input: list of tokens
        output: list of tokens without stopwords
        """
        return [token for token in X if token.lower() not in self.stop_words] if permission else X

    def _stemming(self, X, permission):
        """
        input: list of tokens
        output: list of tokens with their stems
        """
        return [self.stemmer.stem(token) for token in X] if permission else X
