# Corpusのカスタマイズ

- Corpusとは
  - 対象としている言語のドキュメントの集合
  - これらをどのように分析するかが主の目的
 
- NLTKにおけるCorpusReader
  - ドキュメントの、読み込み、ストリーム(逐次処理)、フィルター、データラングのためのインターフェース
  - メモリに乗らないような大量のドキュメントをさばくことができる
  - 66種類存在する
  - さまざまなCorpusを取得可能

```
TaggedCorpusReader
A reader for simple part-of-speech tagged corpora, where sentences are on their own line and tokens are delimited with their tag.

BracketParseCorpusReader
A reader for corpora that consist of parenthesis-delineated parse trees.

ChunkedCorpusReader
A reader for chunked (and optionally tagged) corpora formatted with parentheses.

TwitterCorpusReader
A reader for corpora that consist of tweets that have been serialized into line-delimited JSON.

WordListCorpusReader
List of words, one per line. Blank lines are ignored.

XMLCorpusReader
A reader for corpora whose documents are XML files.

CategorizedCorpusReader
A mixin for corpus readers whose documents are organized by category.`
```

