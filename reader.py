from nltk.corpus.reader import PlaintextCorpusReader
from nltk.tokenize.regexp import RegexpTokenizer
import codecs

DOC_PATTERN = r'(?!\.)[a-z_\s]+/[a-f0-9]+\.txt'
SENT_PATTERN = u'[^　「」！？。]*[！？。]'


class JapaneseCorpusReader(PlaintextCorpusReader):

    def __init__(self, root, fields=DOC_PATTERN, encoding='utf8', **kargs):
        '''
        :param root: corpusが入っているdir
        :param fields: 対象となるcorpus
        :param encoding:
        '''

        PlaintextCorpusReader.__init__(self, root, fields, sent_tokenizer=
        RegexpTokenizer(SENT_PATTERN), encoding=encoding)

    def docs(self, fileid=None):
        for path, enc in self.abspaths(fileid, include_encoding=True):
            with codecs.open(path, 'r', encoding=enc) as f:
                yield f.read()
