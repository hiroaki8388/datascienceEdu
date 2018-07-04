from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer


def freq_vectorize(copus):
    """
    coupusごとに単語の頻度分布をとることでベクトルに変換する
    :param copus: list(str)
    :return:
    """
    features = defaultdict(int)
    for token in copus:
        features[token] = +1

    return features


def one_hot_vectorize(corpus):
    """
    one-hotによりcorpusをベクトルに変換する
    :param corpus: list(str)
    :return:
    """

    return {
        token: True
        for token in corpus
    }
