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

## Step 12: checkpoint 読み込み機能

保存済み checkpoint から、生成に必要な情報を復元できるようにする。

checkpoint に含めるもの:

- model の重み
- tokenizer の語彙
- model config

追加する関数:

- `load_checkpoint`
- `build_model_from_checkpoint`

確認すること:

- tokenizer の語彙を復元できること
- model config を復元できること
- `model.load_state_dict` が通ること
- 読み込んだモデルで生成できること

## Step 13: 生成 API の作成

HTML UI から呼び出すための Python API を作る。

使用ライブラリ:

- `fastapi`
- `uvicorn`

追加ファイル:

```text
app.py
```

API:

```text
POST /generate
```

入力:

```json
{
  "text": "本文"
}
```

出力:

```json
{
  "generated_title": "生成されたタイトル"
}
```

確認すること:

- サーバー起動時に checkpoint を読み込めること
- 任意の本文を POST すると生成結果が返ること
- checkpoint がない場合に分かりやすいエラーを返すこと

## Step 14: HTML UI の作成

ブラウザから生成を試せる HTML を作る。

追加ファイル:

```text
web/index.html
```

画面に置くもの:

- 本文入力用 textarea
- 生成ボタン
- 生成中の表示
- 生成結果の表示
- エラー表示

確認すること:

- 入力した本文が API に送られること
- API の結果が画面に表示されること
- 空入力の場合は送信しないこと
- API エラー時に画面で分かること

## Step 15: Makefile の作成

学習、確認、API 起動のコマンドを短く実行できるようにする。

追加ファイル:

```text
Makefile
```

想定ターゲット:

```text
install
check
train-debug
train
serve
```

Makefile には `uv` を使ったコマンドを書く。

例:

```text
make train-debug
make train
make serve
make check
```

Windows では `make` が標準で入っていない場合があるため、手順書には `uv run ...` の直接コマンドも残す。

## Step 16: UI からの動作確認

学習済み checkpoint を使い、ブラウザからタイトル生成できることを確認する。

想定コマンド:

```text
uv run python tiny_transformer_summarizer.py --train --checkpoint-path checkpoints/tiny_transformer.pt
uv run uvicorn app:app --reload
```

ブラウザで開く:

```text
http://localhost:8000
```

確認すること:

- HTML 画面が表示されること
- 任意の本文を入力できること
- 生成ボタンで API が呼ばれること
- 生成されたタイトルが画面に表示されること
- GPU が使える環境では API 側でも `cuda` を選べること

## 完了条件

- 1 バッチの forward が動く
- loss 計算と短い学習が動く
- attention 用の padding mask と causal mask が使われている
- GPU で数千件以上の学習が動く
- valid loss を確認できる
- checkpoint を保存して再利用できる
- 本文からタイトル文字列を生成できる
- API 経由で任意の本文からタイトルを生成できる
- HTML UI から生成結果を確認できる
