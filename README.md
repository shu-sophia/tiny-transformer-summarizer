# Tiny Transformer Summarizer

Transformer の仕組みを学ぶための、小さな日本語タイトル生成プロジェクトです。

XL-Sum Japanese のニュース本文 `text` から、記事タイトル `title` を生成する Encoder-Decoder Transformer を PyTorch で実装しています。

高性能なモデルを作ることよりも、データ読み込み、トークン化、DataLoader、Attention、loss、学習、生成までの流れをコードで追えることを重視しています。

## できること

- XL-Sum Japanese を読み込む
- 文字レベル tokenizer を作る
- 小さな Transformer を学習する
- 学習済み checkpoint からタイトルを生成する
- ブラウザ UI で本文を入力して生成結果を見る

## 準備

このプロジェクトは `uv` を使います。

```bash
make install
```

## まず小さく学習する

動作確認用の小さい学習です。

```bash
make train-debug
```

## UI を起動する

`make train-debug` で保存した checkpoint を使って API サーバーを起動します。

```bash
make serve-debug
```

ブラウザで開きます。

```text
http://127.0.0.1:8000
```

本文を入力して「生成」を押すと、モデルがタイトルを生成します。

## もう少し大きく学習する

GPU が使える環境では、件数を増やして学習できます。

```bash
make train DEVICE=cuda
```

大きく学習した場合は `checkpoints/tiny_transformer.pt` に保存されます。UI を起動するときは `make serve` を使います（`make serve-debug` ではなく、こちらがデフォルトでこの checkpoint を読み込みます）。

```bash
make serve
```

## よく使うコマンド

基本はこの 4 つです。

```bash
make install
make train-debug
make train
make serve-debug
make serve
make check
```

必要になったら、Makefile の変数で学習量や保存先を変えられます。

```bash
make train TRAIN_EXAMPLES=50000 EPOCHS=5 BATCH_SIZE=16 DEVICE=cuda
make serve CHECKPOINT_PATH=checkpoints/tiny_transformer.pt
```
