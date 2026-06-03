# 実装手順書

## 方針

最初は `tiny_transformer_summarizer.py` の 1 ファイルで実装する。

まず小さいデータで forward / loss / 生成まで動くことを確認する。その後、GPU を使ってデータ量を増やし、ある程度タイトルらしい出力が出るところまで学習する。

モデル設定は次を基本にする。

```text
d_model: 512
num_heads: 8
num_encoder_layers: 6
num_decoder_layers: 6
d_ff: 2048
dropout: 0.1
```

## Step 1: データセットの読み込み

`datasets` を使って XL-Sum Japanese を読み込む。

最初は動作確認用に少量だけ使い、あとで学習用の件数を増やす。

```text
debug: 100〜1000 件
train: 数千〜数万件
valid: 500〜1000 件
```

確認すること:

- データ件数
- `text` の中身
- `title` の中身
- 1 件あたりの本文の長さ

## Step 2: トークナイザーの準備

最初は文字レベルの tokenizer を作る。

特殊トークン:

- `<PAD>`
- `<BOS>`
- `<EOS>`
- `<UNK>`

作る関数:

- `build_vocab`
- `encode`
- `decode`

確認すること:

- 語彙数
- 文字列を ID に変換できること
- ID を文字列に戻せること

## Step 3: Dataset / DataLoader の作成

`text` を `src_ids` に、`title` を `tgt_input` と `tgt_labels` に変換する。

```text
src_ids:    [B, S]
tgt_input:  [B, T]
tgt_labels: [B, T]
```

確認すること:

- 1 バッチ取り出せること
- 各 tensor の shape が想定通りであること
- 系列長をそろえるための `<PAD>` が入っていること
- `tgt_input` と `tgt_labels` が 1 token ずれていること

## Step 4: GPU 対応

`torch.cuda.is_available()` を確認し、GPU が使える場合は `cuda` を使う。

確認すること:

- model が device に乗っていること
- batch の tensor も同じ device に乗っていること
- forward / loss / backward が GPU 上で動くこと

## Step 5: Transformer 部品の実装

実装する部品:

- Token Embedding
- Positional Encoding
- Multi-Head Attention: 8 heads
- Feed Forward
- Encoder Layer
- Decoder Layer
- Encoder: Encoder Layer を 6 層重ねる
- Decoder: Decoder Layer を 6 層重ねる
- Transformer 本体

attention 用の mask もここで実装する。

- padding mask: `<PAD>` の位置を attention が見ないようにする
- causal mask: decoder が未来の token を見ないようにする

mask を使う場所:

- Encoder self-attention: source padding mask
- Decoder masked self-attention: target padding mask + causal mask
- Decoder source-target attention: source padding mask

各 `forward` には、入力と出力の shape コメントを書く。

## Step 6: 1 バッチで順伝播

DataLoader から 1 バッチだけ取り出して、モデルに通す。

```text
src_ids:   [B, S]
tgt_input: [B, T]
logits:    [B, T, vocab_size]
```

ここではまだ学習しない。まず forward が通ることを確認する。

## Step 7: loss 計算

`logits` と `tgt_labels` から `CrossEntropyLoss` を計算する。

`<PAD>` は loss に入れない。`ignore_index=pad_id` を使う。

確認すること:

- loss が scalar になること
- loss が `nan` にならないこと
- reshape 後の shape が合っていること

## Step 8: 小規模学習ループ

まずは debug 用データで短く学習する。

```text
examples: 100〜1000
epochs: 1〜3
batch_size: 8〜32
```

確認すること:

- loss が表示されること
- backward が通ること
- optimizer が更新されること
- 途中で簡単な生成を試せること

## Step 9: GPU で本番寄りの学習

debug が通ったら、データ量を増やして GPU で学習する。

```text
examples: 10000〜50000
epochs: 3〜10
batch_size: GPU メモリに合わせて調整
d_model: 512
num_heads: 8
num_encoder_layers: 6
num_decoder_layers: 6
```

確認すること:

- train loss が下がること
- valid loss も確認すること
- 数件の生成結果を定期的に見ること
- checkpoint を保存できること

## Step 10: Greedy Decoding

学習後、本文からタイトルを生成する。

流れ:

1. 本文を `src_ids` に変換する
2. `<BOS>` から生成を始める
3. 次 token を 1 つずつ選ぶ
4. `<EOS>` または最大長で止める
5. token ID を文字列に戻す

確認すること:

- 生成処理が最後まで動くこと
- 空文字だけにならないこと
- 生成結果と正解タイトルを並べて見られること

## Step 11: 余裕があれば生成方法を追加

Greedy Decoding が動いたあと、必要なら生成方法を増やす。

- temperature
- top-k
- top-p

## 完了条件

- 1 バッチの forward が動く
- loss 計算と短い学習が動く
- attention 用の padding mask と causal mask が使われている
- GPU で数千件以上の学習が動く
- valid loss を確認できる
- checkpoint を保存して再利用できる
- 本文からタイトル文字列を生成できる
