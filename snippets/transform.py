from sklearn.base import BaseEstimator, TransformerMixin


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
        :return: list(str)
        """
        transed = [self.normalize(word) for word in corpus.words()]

        return list(filter(None, transed))
