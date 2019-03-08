from collections import defaultdict
from languageModel import LanguageModel
from unigram import Unigram
from bigram import Bigram
import random
import bisect

'''
Jiaming He
'''
class BigramInterpolation(LanguageModel):


    def __init__(self):
        self.unigram = Unigram()
        self.bigram = Bigram()
        self.coef = 0.5
        print("W(bigram):W(unigram) coefficient is 1 :", self.coef)

    
    def train(self, trainingSentences):
        self.unigram.train(trainingSentences)
        self.bigram.train(trainingSentences)
    
    def getWordProbability(self, sentence, index):
        coef = self.coef
        x = 1 / (1+coef)

        if index == len(sentence):
            word = LanguageModel.STOP
            prev_word = sentence[-1]
        elif index == 0:
            word = sentence[0]
            prev_word = LanguageModel.START
        else:
            word = sentence[index]
            prev_word = sentence[index-1]

        if prev_word not in self.bigram.probCounter:
            prev_word = LanguageModel.UNK

        if self.bigram.probCounter[prev_word][word] == 0:
            return x*coef*self.unigram.getWordProbability(sentence, index)
        else:
            return x*self.bigram.getWordProbability(sentence, index) + x*coef*self.unigram.getWordProbability(sentence, index)

    def getVocabulary(self, context):

        next_posb_word = []
        # append all possible word except START in self.total
        for next_word in self.bigram.total:
            if next_word != LanguageModel.START:
                next_posb_word.append(next_word)
        # append STOP manually since there is no STOP in self.total
        next_posb_word.append(LanguageModel.STOP)

        return next_posb_word

    def generateWord(self, context):

        return self.bigram.generateWord(context)

    def generateSentence(self):
        result = []
        # limit sentence length to 20
        for i in range(20):
            word = LanguageModel.UNK
            while word == LanguageModel.UNK:
                # make sure word != UNK
                word = self.generateWord(result)
            result.append(word)
            if word == LanguageModel.STOP:
                break
        return result
