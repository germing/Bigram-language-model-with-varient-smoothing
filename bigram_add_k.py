from collections import defaultdict, Counter
from languageModel import LanguageModel
import random
import bisect

'''
Jiaming He
'''
class BigramAddK(LanguageModel):



    def __init__(self):
        # P(word2|word1) = self.probCounter[word1][word2]
        self.probCounter = defaultdict(lambda: defaultdict(float))
        self.rand = random.Random()
        self.KV = 0
        self.k = 1
        print("Current value of K is", self.k)
    
    def train(self, trainingSentences):
        self.accu = defaultdict(list)
        self.total = defaultdict(int)

        # build up self.total
        for sentence in trainingSentences:
            self.total[LanguageModel.START] += 1
            for word in sentence:
                self.total[word] += 1
        # all tokens except </s> can be followed by UNK
        for token in self.total:
            self.total[token] += 1
        # UNK can be followed by any tokens except <s>, also itself and STOP
        self.total[LanguageModel.UNK] = len(self.total) + 1

        # adding KV to each word count in self.total
        self.KV = self.k*len(self.total)
        for token in self.total:
           self.total[token] += self.KV

        # initialize the first layer of self.probCounter
        for word in list(self.total):
            self.probCounter[word] = defaultdict(float)

        # build up the second layer of self.probCounter
        for sentence in trainingSentences:
            sentence.insert(0, LanguageModel.START)
            sentence.append(LanguageModel.STOP)
            word_pair_counter = Counter(zip(sentence, sentence[1:]))
            for pairs in list(word_pair_counter):
                self.probCounter[pairs[0]][pairs[1]] += word_pair_counter[pairs]

        for token in self.total:
            # add count of UNK+token to probCounter
            if token != LanguageModel.START:
                self.probCounter[LanguageModel.UNK][token] = 1
            # add count of token+UNK to probCounter
            self.probCounter[token][LanguageModel.UNK] = 1
        # Since there is no STOP in self.total, add the case of UNK+STOP manually
        self.probCounter[LanguageModel.UNK][LanguageModel.STOP] = 1

        # build up self.accu
        for word in self.probCounter:
            self.accu[word] = []
            for next_word in self.probCounter[word]:
                self.accu[word].append(self.probCounter[word][next_word]
                                 if len(self.accu[word]) == 0
                                 else self.accu[word][-1] + self.probCounter[word][next_word])
                # calculate the condition probability
                # self.probCounter[word][next] = P(next|word) = c(word, next)/c(word)
                self.probCounter[word][next_word] = self.probCounter[word][next_word] / self.total[word]

    def getWordProbability(self, sentence, index):
        if index == len(sentence):
            word = LanguageModel.STOP
            prev_word = sentence[-1]
        elif index == 0:
            word = sentence[0]
            prev_word = LanguageModel.START
        else:
            word = sentence[index]
            prev_word = sentence[index-1]

        if prev_word not in self.probCounter:
            prev_word = LanguageModel.UNK

        if self.probCounter[prev_word][word] == 0:
            return self.k / (self.total[prev_word])
        else:
            return self.probCounter[prev_word][word] + (self.k / (self.total[prev_word]))

    def getVocabulary(self, context):

        next_posb_word = []
        # append all possible word except START in self.total
        for next_word in self.total:
            if next_word != LanguageModel.START:
                next_posb_word.append(next_word)
        # append STOP manually since there is no STOP in self.total
        next_posb_word.append(LanguageModel.STOP)

        return next_posb_word

    def generateWord(self, context):

        if len(context) == 0:
            word = LanguageModel.START
        else:
            word = context[-1]
            if word not in self.total:
                word = LanguageModel.UNK

        i = self.rand.randint(0, int(self.total[word]) - 1)
        # if the random number falls into spin wheel with corresponding words
        if i < (self.total[word] - self.KV - 1):
            index = bisect.bisect(self.accu[word], i)
            return list(self.probCounter[word])[index]
        # if the random number fall into empty space in the spin wheel
        else:
            other_word = [token for token in (set(self.total) - set(self.probCounter[word]))]
            return random.choice(other_word)
        
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
