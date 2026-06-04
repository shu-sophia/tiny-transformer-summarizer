PYTHON := uv run python
CHECKPOINT_PATH ?= checkpoints/tiny_transformer.pt
DEVICE ?= auto

.PHONY: install check train-debug train serve

install:
	uv sync

check:
	$(PYTHON) -m py_compile tiny_transformer_summarizer.py app.py

train-debug:
	$(PYTHON) tiny_transformer_summarizer.py \
		--train \
		--train-examples 100 \
		--valid-examples 20 \
		--max-src-len 128 \
		--max-tgt-len 32 \
		--batch-size 8 \
		--d-model 128 \
		--num-heads 4 \
		--num-encoder-layers 2 \
		--num-decoder-layers 2 \
		--d-ff 512 \
		--max-steps 20 \
		--checkpoint-path checkpoints/debug.pt \
		--device $(DEVICE)

train:
	$(PYTHON) tiny_transformer_summarizer.py \
		--train \
		--train-examples 10000 \
		--valid-examples 1000 \
		--epochs 3 \
		--checkpoint-path $(CHECKPOINT_PATH) \
		--device $(DEVICE)

serve:
	uv run uvicorn app:app --reload
