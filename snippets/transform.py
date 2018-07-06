import os
from functools import reduce
from operator import add
import numpy as np

import gensim
from sklearn.base import BaseEstimator, TransformerMixin
from gensim.matutils import sparse2full

from snippets.reader import JapaneseCorpusReader


class JapaneseTextNormalizer(BaseEstimator, TransformerMixin):
    """Janomeで変換したwordの変換を行う

    """

    STOP_WORD_OF_POS = ['助詞', '助動詞', '記号']

    def __init__(self):
        pass

    def is_stopword(self, token, stop_word_of_pos=STOP_WORD_OF_POS):
        """特定の品詞に属するwordをstopwordと判定する"""
        part_of_speech = set(token.part_of_speech.split(','))

        return not part_of_speech.isdisjoint(stop_word_of_pos)

    def normalize(self, token):
        """janomeのTokenの集合を、原型の単語に変換する"""

        if not self.is_stopword(token):
            return token.base_form

    def fit(self, X, y=None):
        return self

    def transform(self, corpus):
        """
        corpusReaderのtokenからnormalizeした単語のlistを返す
        :param corpus: JapaneseCorpusReader
        :return: list(list(str))
        """
        transed = [list(filter(None, [self.normalize(word) for word in sent])) for sent in corpus.sents()]

        return transed


class GensimVectorizer(BaseEstimator, TransformerMixin):
    """
    ScikitLearnにはword2vecのような変換が存在しないので、Gensimより作成する
    学習済みモデルはこちらのを使う
    http://aial.shiroyagi.co.jp/2017/02/japanese-word2vec-model-builder/
    """
    WORD_DIM = 50

    def __init__(self, path=None):
        self.path = path
        self.id2word = None

        self._load()

    def _load(self):
        if os.path.exists(self.path):
            # 学習済みモデルのload
            self.id2word = gensim.models.Word2Vec.load(self.path)
        else:
            pass

    def fit(self, sent):
        """

        :param sent:
        :return:
        """

        return self

    def transform(self, sents):
        """
        現状Doc2Vecの学習済みモデルが存在しないので、word2vecで変換した単語単位のモデルの平均ベクトルを取る
        :param sents:
        :return:
        """
        for sent in sents:
            wordvecs = [self._word2vec(word) for word in sent]

            sentvec = self._mean(wordvecs)
        
            yield sentvec

    def _word2vec(self, word):
        try:
            word_vec = self.id2word[word]
            if len(word_vec) == self.WORD_DIM:
                return word_vec
            else:
                return np.zeros(self.WORD_DIM)

        except KeyError:
            return np.zeros(self.WORD_DIM)

    def _mean(self, wordvecs):
        try:
            sentvec = (reduce(add, wordvecs)) / len(wordvecs)

            return sentvec

        except TypeError:
            pass


if __name__ == '__main__':
    jap = JapaneseCorpusReader(root='./../dataset/corpas', encoding='shift_jis')
    jap_trans = JapaneseTextNormalizer()
    traned = jap_trans.transform(corpus=jap)
    model = GensimVectorizer('./../dataset/latest-ja-word2vec-gensim-model/word2vec.gensim.model')
    data = model.transform(traned)
    [vec for vec in data]
