# Tiny Transformer Summarizer 仕様書

## 今回作るものの概要

XL-Sum Japanese を使って、ニュース本文からタイトルを生成する小さな Transformer を作る。

目的は高性能な要約モデルを作ることではなく、Transformer の基本部品がコード上でどうつながるかを理解すること。

最初は 1 ファイルで動く最小構成から始め、100〜1000 件程度の小さいデータで動作確認する。

## 使用するライブラリ

- `torch`: モデル実装、loss 計算、学習
- `datasets`: XL-Sum Japanese の読み込み
- `tqdm`: 学習ログ表示
- `fastapi`: 生成 API
- `uvicorn`: API サーバー起動

最初は事前学習済みモデルは使わない。トークナイザーも理解しやすさを優先して、文字レベルの簡単なものから始める。

## データの形式

使用データセット: `csebuetnlp/xlsum` の `japanese`

主に使う列:

- `text`: 入力本文
- `title`: 最初の生成対象

最初のタスク:

```text
入力: text
出力: title
```

バッチの基本形式:

```text
src_ids:    [B, S]
tgt_input:  [B, T]
tgt_labels: [B, T]
```

## モデル構成

小さな Encoder-Decoder Transformer を実装する。

主な部品:

- Token Embedding
- Positional Encoding
- Encoder
- Decoder
- Masked Self-Attention
- Source-Target Attention
- Feed Forward
- Skip Connection
- LayerNorm
- 出力 Linear 層

モデルの入出力:

```text
入力:
  src_ids:   [B, S]
  tgt_input: [B, T]

出力:
  logits:    [B, T, vocab_size]
```

## UI / API 構成

学習済みモデルを使って、ブラウザから任意の本文に対するタイトル生成を試せるようにする。

HTML から PyTorch モデルを直接実行するのではなく、Python 側に生成 API を用意する。

```text
web/index.html  ->  app.py  ->  checkpoint  ->  generated title
```

主な役割:

- `web/index.html`: 本文入力、生成ボタン、生成結果表示
- `app.py`: checkpoint を読み込み、入力本文からタイトルを生成する API
- `checkpoints/*.pt`: 学習済みモデルの重み、tokenizer、設定を保存したファイル

API の基本形式:

```text
POST /generate
入力: text
出力: generated_title
```

## 最初の成果物

```text
tiny_transformer_summarizer.py
```

まずはこの 1 ファイルで、データ読み込みから生成までを通す。

## 追加の成果物

```text
app.py
web/index.html
Makefile
```

学習済み checkpoint を使って、ローカルブラウザからタイトル生成を試せるようにする。
