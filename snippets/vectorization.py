# from collections import defaultdict
#
# from nltk import TextCollection
# from sklearn.feature_extraction.text import CountVectorizer
#
#
# def freq_vectorize(sent):
#     """
#     sentenceごとに単語の頻度分布をとることでベクトルに変換する
#     :param sent: list(str)
#     :return:
#     """
#     features = defaultdict(int)
#     for token in sent:
#         features[token] = +1
#
#     return features
#
#
# def one_hot_vectorize(sent):
#     """
#     one-hotによりsentenceをベクトルに変換する
#     :param sent: list(str)
#     :return:
#     """
#
#     return {
#         token: True
#         for token in sent
#     }
#
#
# def tf_idf_vectorize(sents):
#     """
#     tf_idfによりsentssをベクトルに変換する
#     :param corpus: list(list(str))
#     :return:
#     """
#     # 変換するために、全ての単語のlistを生成
#     words = sum(sents, [])
#     word_collection = TextCollection(words)
#
#     for sent in sents:
#         yield {
#             word: word_collection.tf_idf(word, sent)
#             for word in sent
#         }
