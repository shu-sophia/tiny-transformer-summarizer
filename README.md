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
uv sync
```

## まず小さく学習する

動作確認用の小さい学習です。

```bash
uv run python tiny_transformer_summarizer.py --train --train-examples 100 --valid-examples 20 --max-steps 20 --checkpoint-path checkpoints/debug.pt
```

## UI を起動する

学習で保存した checkpoint を指定して API サーバーを起動します。

PowerShell:

```powershell
$env:CHECKPOINT_PATH="checkpoints/debug.pt"
uv run uvicorn app:app --reload
```

ブラウザで開きます。

```text
http://127.0.0.1:8000
```

本文を入力して「生成」を押すと、モデルがタイトルを生成します。

## もう少し大きく学習する

GPU が使える環境では、件数を増やして学習できます。

```bash
uv run python tiny_transformer_summarizer.py --train --train-examples 10000 --valid-examples 1000 --epochs 3 --checkpoint-path checkpoints/tiny_transformer.pt --device auto
```

## Makefile を使う場合

`make` が使える環境なら、短いコマンドでも実行できます。

```bash
make train-debug
make serve
make check
```

Windows では `make` が入っていないことがあります。その場合は上の `uv run ...` コマンドを使ってください。
