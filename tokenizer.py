from nltk.data import load
from nltk import tokenize
from os.path import isfile
from string import punctuation
import re

class Tokenizer():

    def __init__(self, text_obj, language='english'):
        self.text_obj = text_obj
        self.language = language
        self.__raw_text()
        self.sent_tokenizer()
        self.word_tokenizer()

    def __raw_text(self):
        if isfile(self.text_obj):
            with open(self.text_obj) as f:
                self.raw_text = f.read()
        else:
            self.raw_text = self.text_obj

    def sent_tokenizer(self):
        tokens_sent = load('tokenizers/punkt/%s.pickle' % self.language)
        self.sent_tokens = tokens_sent.tokenize(self.raw_text)


    def word_tokenizer(self):
        tmp_tokens = tokenize.word_tokenize(
        self.raw_text, language=self.language)
        self.word_tokens = [i.lower() for i in tmp_tokens if i not in punctuation]
