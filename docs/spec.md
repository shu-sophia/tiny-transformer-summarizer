# Tiny Transformer Summarizer 仕様書

このプロジェクトでは、XL-Sum Japanese のニュース本文から「タイトル」または「短い要約」を生成する小さな Encoder-Decoder Transformer を作ります。

目的は高性能な要約モデルを作ることではなく、Transformer の部品がどの順番でつながり、どんなテンソル shape で流れていくのかを、コードを読みながら追えるようにすることです。

最初は 1 ファイルで動く最小構成から始めます。100〜1000 件くらいの小さいデータで、1 バッチの順伝播、loss 計算、短時間の学習、greedy decoding による生成までを通します。

## 今回作るものの概要

作るものは「本文を入力すると、タイトルっぽい短い文を出す」学習用の Transformer です。

最初の目標は次の通りです。

- XL-Sum Japanese を少量だけ読み込む
- 本文 `text` を入力、タイトル `title` を出力として使う
- 文字レベルのシンプルなトークナイザーから始める
- Dataset / DataLoader でミニバッチを作る
- 小さな Encoder-Decoder Transformer を PyTorch で実装する
- decoder が「次のトークン」を予測する形で loss を計算する
- 数分以内の小規模学習を回す
- 学習後に本文からタイトルを greedy decoding で生成する

最初は生成品質が低くても大丈夫です。むしろ、変な出力でも「なぜそうなるのか」を追えることを大事にします。

## 使用するライブラリ

最小構成では次を使います。

- `python`: 実行環境
- `torch`: Transformer、loss、学習ループ
- `datasets`: XL-Sum Japanese の読み込み
- `tqdm`: 学習の進み具合を見やすくするため。なくても動くようにしてよい
- `random`: 小さいサンプルの切り出しや seed 固定

あえて最初は Hugging Face `transformers` の事前学習済みモデルは使いません。理由は、今回は性能よりも Transformer の中身を自分で読むことが目的だからです。

トークナイザーも最初は自作の文字レベル tokenizer にします。日本語では空白区切りが効きにくいので、文字単位にすると少し長くなりますが、仕組みはかなり見通しやすくなります。

余裕が出たら SentencePiece や Hugging Face tokenizers に置き換えます。

## データの形式

使うデータセットは XL-Sum Japanese です。

Hugging Face の `csebuetnlp/xlsum` には `japanese` 設定があり、主な列は次の通りです。

- `id`: 記事 ID
- `url`: 元記事 URL
- `title`: 記事タイトル
- `summary`: 記事要約
- `text`: 記事本文

このプロジェクトでは、まず次のペアを使います。

```text
入力:  text   = ニュース本文
出力:  title  = 記事タイトル
```

慣れてきたら、次のように切り替えます。

```text
入力:  text     = ニュース本文
出力:  summary  = 短い要約
```

実装上は `target_field = "title"` または `target_field = "summary"` のように切り替えられるようにします。

最初に作るバッチのイメージはこうです。

```text
src_ids:      [batch_size, src_len]
tgt_input:    [batch_size, tgt_len]
tgt_labels:   [batch_size, tgt_len]
```

例として、`batch_size = 4`, `src_len = 128`, `tgt_len = 32` ならこうなります。

```text
src_ids.shape     == [4, 128]
tgt_input.shape   == [4, 32]
tgt_labels.shape  == [4, 32]
```

`src_ids` は本文、`tgt_input` は decoder に入れる途中までのタイトル、`tgt_labels` は正解として予測してほしい次トークンです。

## モデル構成

作るモデルは小さな Encoder-Decoder Transformer です。

全体の流れは次のようになります。

```text
本文 tokens
  -> token embedding
  -> positional encoding
  -> encoder
  -> encoder memory

タイトル tokens shifted right
  -> token embedding
  -> positional encoding
  -> decoder
  -> output linear
  -> vocab logits
```

テンソル shape の基本は次の通りです。

```text
src_ids:        [B, S]
tgt_input:      [B, T]

src_emb:        [B, S, d_model]
tgt_emb:        [B, T, d_model]

encoder_out:    [B, S, d_model]
decoder_out:    [B, T, d_model]
logits:         [B, T, vocab_size]
```

ここで、

- `B`: batch size
- `S`: source length、本文側の長さ
- `T`: target length、タイトルまたは要約側の長さ
- `d_model`: 埋め込みと Transformer 内部表現の次元
- `vocab_size`: 語彙数

最初の小さい設定は次を想定します。

```text
d_model = 128
num_heads = 4
num_encoder_layers = 2
num_decoder_layers = 2
d_ff = 512
dropout = 0.1
max_src_len = 128 または 256
max_tgt_len = 32 または 64
```

### Encoder

Encoder は本文を読む部分です。

1 層の Encoder block は次を持ちます。

```text
入力 x: [B, S, d_model]

1. Multi-Head Self-Attention
2. Skip Connection + LayerNorm
3. Feed Forward
4. Skip Connection + LayerNorm

出力 x: [B, S, d_model]
```

Transformer の概念としては、本文中の各トークンが本文中の他のトークンを見て、文脈込みの表現に変わるところです。

### Decoder

Decoder はタイトルを左から右へ生成する部分です。

1 層の Decoder block は次を持ちます。

```text
入力 y:              [B, T, d_model]
encoder_out:         [B, S, d_model]

1. Masked Self-Attention
2. Skip Connection + LayerNorm
3. Source-Target Attention
4. Skip Connection + LayerNorm
5. Feed Forward
6. Skip Connection + LayerNorm

出力 y:              [B, T, d_model]
```

Masked Self-Attention は、decoder が未来の正解トークンを見ないようにするための仕組みです。

Source-Target Attention は、タイトル生成中の decoder が本文側の encoder 出力を見るための仕組みです。

## 学習の流れ

学習では teacher forcing を使います。

たとえば正解タイトルが次の token ID だとします。

```text
正解: [BOS, A, B, C, EOS]
```

decoder に入れるものはこうします。

```text
tgt_input:  [BOS, A, B, C]
```

予測してほしい正解はこうです。

```text
tgt_labels: [A, B, C, EOS]
```

つまり decoder は「ここまでの正解を見て、次の token を予測する」練習をします。

モデル出力は次の shape です。

```text
logits: [B, T, vocab_size]
```

CrossEntropyLoss に渡すときは、次のように平らにします。

```text
logits_for_loss: [B * T, vocab_size]
labels_for_loss: [B * T]
```

padding 部分は loss に入れたくないので、`ignore_index=pad_id` を使います。

学習ループは最初はとても小さくします。

```text
for epoch in range(1〜3):
    for batch in train_loader:
        1. src_ids, tgt_input, tgt_labels を取り出す
        2. model(src_ids, tgt_input) で logits を得る
        3. CrossEntropyLoss を計算する
        4. loss.backward()
        5. optimizer.step()
        6. optimizer.zero_grad()
```

最初に見るべき出力は ROUGE ではなく、次のようなものです。

- 1 バッチで forward が落ちない
- `logits.shape` が `[B, T, vocab_size]` になる
- loss が `nan` にならない
- 数十 step で loss が少しでも下がる
- 生成が空文字だけにならない

## 推論の流れ

推論では正解タイトルを使いません。

本文だけを入力して、decoder は `<bos>` から 1 token ずつ生成します。

```text
1. src_text を tokenizer で src_ids にする
2. encoder で本文を読む
3. tgt_ids = [BOS] から開始する
4. model(src_ids, tgt_ids) を呼ぶ
5. 最後の位置の logits を取り出す
6. greedy decoding なら argmax で次 token を選ぶ
7. EOS が出るか max_len に達するまで繰り返す
8. token IDs を文字列に戻す
```

shape は次のように伸びていきます。

```text
step 1: tgt_ids.shape == [1, 1]
step 2: tgt_ids.shape == [1, 2]
step 3: tgt_ids.shape == [1, 3]
...
```

各 step の model 出力はこうです。

```text
logits.shape == [1, current_tgt_len, vocab_size]
next_logits.shape == [1, vocab_size]
```

## 実装ステップ一覧

### Step 1: データセットの読み込みと中身確認

入力:

```text
データセット名: csebuetnlp/xlsum
設定: japanese
split: train
件数: 最初は 100〜1000 件
```

出力:

```text
examples: list または datasets.Dataset
各 example は text/title/summary を持つ
```

確認ポイント:

- `len(dataset)` を print する
- 1 件目の `text`, `title`, `summary` を短く print する
- 本文が長すぎることを確認し、後で `max_src_len` で切る前提にする

対応する Transformer 概念:

- まだ Transformer そのものではなく、source sequence と target sequence を決める準備です。

### Step 2: トークナイザーの準備

入力:

```text
train examples の text と target_field(title または summary)
```

出力:

```text
vocab: 文字 -> ID の辞書
encode(text): 文字列 -> token IDs
decode(ids): token IDs -> 文字列
```

特殊トークン:

```text
<PAD>: padding 用
<BOS>: decoder 開始用
<EOS>: 文末用
<UNK>: 未知文字用
```

確認ポイント:

- `vocab_size` を print する
- `encode("今日は晴れ")` の結果を print する
- `decode(encode(text))` がだいたい元に戻るか見る

対応する Transformer 概念:

- Transformer は文字列を直接読めないので、まず離散 token ID に変換します。

### Step 3: Dataset / DataLoader の作成

入力:

```text
examples
Tokenizer
max_src_len
max_tgt_len
batch_size
```

出力:

```text
batch = {
    "src_ids":    [B, S],
    "tgt_input":  [B, T],
    "tgt_labels": [B, T]
}
```

確認ポイント:

- 1 バッチ取り出して shape を print する
- `src_ids[0]`, `tgt_input[0]`, `tgt_labels[0]` を decode して見る
- padding が入っているか確認する

対応する Transformer 概念:

- ミニバッチ化と padding は、複数の長さが違う系列をまとめて計算するための準備です。

### Step 4: 小さい Encoder-Decoder Transformer の実装

入力:

```text
src_ids:   [B, S]
tgt_input: [B, T]
```

出力:

```text
logits: [B, T, vocab_size]
```

実装する部品:

- TokenEmbedding
- PositionalEncoding
- MultiHeadAttention
- FeedForward
- EncoderLayer
- DecoderLayer
- Encoder
- Decoder
- TransformerSummarizer

確認ポイント:

- 各 block の forward に shape コメントを書く
- debug モードでは主要 shape を print できるようにする
- まずは小さいダミー input で動かす

対応する Transformer 概念:

- Self-Attention、Masked Self-Attention、Source-Target Attention、Feed Forward、Skip Connection、LayerNorm がここで実装されます。

### Step 5: 1 バッチだけ順伝播して shape 確認

入力:

```text
DataLoader から取り出した 1 batch
```

出力:

```text
logits: [B, T, vocab_size]
```

確認ポイント:

- `src_ids.shape`
- `tgt_input.shape`
- `tgt_labels.shape`
- `logits.shape`
- `logits[:, -1, :].shape`

対応する Transformer 概念:

- Encoder が本文を読み、Decoder が target の途中までを見て、各位置の次 token 候補を出すところです。

### Step 6: loss 計算

入力:

```text
logits:     [B, T, vocab_size]
tgt_labels: [B, T]
```

出力:

```text
loss: scalar
```

確認ポイント:

- `logits.reshape(-1, vocab_size).shape == [B*T, vocab_size]`
- `tgt_labels.reshape(-1).shape == [B*T]`
- loss が有限値か確認する
- padding 部分が `ignore_index=pad_id` で無視されているか確認する

対応する Transformer 概念:

- 次トークン予測です。decoder の各位置で「次に来る正解 token」を当てるように学習します。

### Step 7: 小規模学習ループ

入力:

```text
train_loader
model
optimizer
loss_fn
```

出力:

```text
step ごとの loss
学習後 model parameters
```

確認ポイント:

- 最初は 20〜100 step だけ回す
- loss が `nan` にならない
- `device` が CPU/GPU で揃っている
- 途中でサンプル生成を 1 件だけ試す

対応する Transformer 概念:

- 予測誤差を backpropagation で各部品に返し、Embedding、Attention、Feed Forward などの重みを更新します。

### Step 8: Greedy decoding による生成

入力:

```text
src_text: ニュース本文 1 件
```

出力:

```text
generated_title: 生成されたタイトル文字列
```

確認ポイント:

- `<BOS>` から始めているか
- `<EOS>` が出たら止まるか
- 毎 step で `tgt_ids.shape` が 1 ずつ伸びるか
- 生成結果が空文字だけになっていないか

対応する Transformer 概念:

- 学習時と違って正解 target は使わず、モデル自身の予測を次の入力に戻して文章を作ります。

### Step 9: top-k / top-p / temperature の追加は余裕があれば

入力:

```text
next_logits: [1, vocab_size]
```

出力:

```text
next_token_id: [1]
```

追加する生成方法:

- `temperature`: logits の尖り具合を変える
- `top-k`: 確率上位 k 個から選ぶ
- `top-p`: 累積確率 p までの候補から選ぶ

確認ポイント:

- greedy より出力に揺れが出るか
- temperature を上げすぎると崩れやすいことを見る
- seed を固定すると再現しやすい

対応する Transformer 概念:

- Transformer 本体というより、decoder の出力分布からどう token を選ぶかという生成戦略です。

## 初学者がつまずきやすいポイント

### 1. target のずらし方

一番つまずきやすいのは `tgt_input` と `tgt_labels` の違いです。

```text
tgt_input  = [BOS, A, B, C]
tgt_labels = [A,   B, C, EOS]
```

この 1 token ずらしが、次トークン予測の中心です。

### 2. Masked Self-Attention

decoder は未来を見てはいけません。

たとえば `B` を予測するときに `C` や `EOS` を見てしまうと、テストのカンニングになります。

そのため causal mask を使います。

```text
causal_mask.shape == [1, 1, T, T]
```

### 3. padding mask

batch 内の系列長を揃えるために `<PAD>` を入れますが、padding は本物の文章ではありません。

attention でも loss でも padding をなるべく無視します。

```text
src_pad_mask.shape == [B, 1, 1, S]
tgt_pad_mask.shape == [B, 1, 1, T]
```

### 4. CrossEntropyLoss に入れる shape

PyTorch の `CrossEntropyLoss` は、最後の語彙次元をクラスとして扱います。

そのため、次のように reshape します。

```text
logits: [B, T, V] -> [B*T, V]
labels: [B, T]    -> [B*T]
```

### 5. 日本語 tokenizer

日本語は英語のように空白で単語が分かれません。

最初は文字単位にすると、単純で動きが見やすいです。ただし sequence が長くなりやすいので、`max_src_len` を小さくして始めます。

### 6. CPU で遅い

CPU でも動くように小さくします。

最初は次くらいで十分です。

```text
num_examples = 100
batch_size = 4
d_model = 128
num_layers = 2
max_src_len = 128
max_tgt_len = 32
```

### 7. 生成が変でも焦らない

小さいデータ、小さいモデル、短い学習では、まともなタイトルが出ないことがあります。

最初の成功条件は「学習が進み、shape が合い、生成処理が最後まで動くこと」です。

## 最初の成果物

最初の実装ファイルは次の 1 ファイルにします。

```text
tiny_transformer_summarizer.py
```

この 1 ファイルの中に、次を順番に置きます。

1. import と設定値
2. seed 固定
3. データ読み込み
4. 文字レベル tokenizer
5. Dataset / DataLoader
6. mask 作成関数
7. Transformer 部品
8. 1 バッチ shape 確認
9. loss 計算
10. 小規模学習ループ
11. greedy decoding
12. main 関数

ファイルが長くなってきたら、そのあとで分割します。

## 参考

- XL-Sum dataset card: https://huggingface.co/datasets/csebuetnlp/xlsum
- XL-Sum paper: https://aclanthology.org/2021.findings-acl.413
