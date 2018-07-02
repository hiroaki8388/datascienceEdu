from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize.regexp import RegexpTokenizer
from janome.tokenizer import Tokenizer as JanomeTokenizer
import codecs

# root配下に存在するdocsでcorpusの対象とするPATTERN
DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.txt'

SENT_PATTERN = u'[^　「」！？。]*[！？。]'


class JapaneseCorpusReader(PlaintextCorpusReader):

    def __init__(self, root, fields=DOC_PATTERN, sent_pattern=SENT_PATTERN, encoding='utf8', **kargs):
        """
        :param root: corpusが入っているdir
        :param fields: 対象となるcorpus
        :param encoding:
        """

        PlaintextCorpusReader.__init__(self, root, fields,
                                       word_tokenizer=JanomeTokenizer(udic_enc=encoding),
                                       sent_tokenizer=RegexpTokenizer(sent_pattern),
                                       encoding=encoding)

    def docs(self, fileid=None):
        """

        :param fileid:
        :return:
        """
        for path, enc in self.abspaths(fileid, include_encoding=True):
            with codecs.open(path, 'r', encoding=enc) as f:
                yield f.read()
