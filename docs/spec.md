# Tiny Transformer Summarizer 仕様書

## 今回作るものの概要

XL-Sum Japanese を使って、ニュース本文からタイトルまたは短い要約を生成する小さな Transformer を作る。

目的は高性能な要約モデルを作ることではなく、Transformer の基本部品がコード上でどうつながるかを理解すること。

最初は 1 ファイルで動く最小構成から始め、100〜1000 件程度の小さいデータで動作確認する。

## 使用するライブラリ

- `torch`: モデル実装、loss 計算、学習
- `datasets`: XL-Sum Japanese の読み込み
- `tqdm`: 学習ログ表示

最初は事前学習済みモデルは使わない。トークナイザーも理解しやすさを優先して、文字レベルの簡単なものから始める。

## データの形式

使用データセット: `csebuetnlp/xlsum` の `japanese`

主に使う列:

- `text`: 入力本文
- `title`: 最初の生成対象
- `summary`: 余裕があれば生成対象に切り替える

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

## 学習の流れ

1. 本文とタイトルを token ID に変換する
2. decoder 入力と正解ラベルを 1 token ずらして作る
3. モデルで `logits` を出す
4. `CrossEntropyLoss` で次トークン予測の loss を計算する
5. backpropagation で重みを更新する

学習時はまず 1 バッチの forward と loss 計算が動くことを優先する。

## 推論の流れ

1. 本文を token ID に変換する
2. `<BOS>` から生成を開始する
3. decoder の最後の出力から次 token を選ぶ
4. greedy decoding で 1 token ずつ追加する
5. `<EOS>` または最大長に達したら終了する
6. token ID を文字列に戻す

## 実装ステップ一覧

1. データセットの読み込みと中身確認
2. トークナイザーの準備
3. Dataset / DataLoader の作成
4. 小さい Encoder-Decoder Transformer の実装
5. 1 バッチだけ順伝播して shape 確認
6. loss 計算
7. 小規模学習ループ
8. Greedy decoding による生成
9. 余裕があれば top-k / top-p / temperature を追加

## 実装方針

- 初心者が読めるように、過度に抽象化しない
- 最初は 1 ファイルで動く構成にする
- CPU でも小規模に試せる設定にする
- 主要なテンソルには shape コメントを書く
- 必要に応じて途中の shape やサンプル出力を print できるようにする
- 各実装ブロックで、Transformer のどの概念に対応するかを短く説明する

## 初学者がつまずきやすいポイント

- `tgt_input` と `tgt_labels` の 1 token ずらし
- decoder の未来を見ないための mask
- padding を attention や loss で扱う方法
- `CrossEntropyLoss` に渡す前の shape 変換
- 日本語をどう token 化するか
- 小さいデータでは生成品質が低くても自然なこと

## 最初の成果物

```text
tiny_transformer_summarizer.py
```

まずはこの 1 ファイルで、データ読み込みから生成までを通す。
