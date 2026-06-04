PYTHON := uv run python
CHECKPOINT_PATH ?= checkpoints/tiny_transformer.pt
DEBUG_CHECKPOINT_PATH ?= checkpoints/debug.pt
DEVICE ?= auto
TRAIN_EXAMPLES ?= 10000
VALID_EXAMPLES ?= 1000
EPOCHS ?= 3
BATCH_SIZE ?= 8
MAX_SRC_LEN ?= 256
MAX_TGT_LEN ?= 64
MAX_VOCAB_SIZE ?= 8000
D_MODEL ?= 512
NUM_HEADS ?= 8
ENCODER_LAYERS ?= 6
DECODER_LAYERS ?= 6
D_FF ?= 2048
LR ?= 3e-4
DEBUG_TRAIN_EXAMPLES ?= 100
DEBUG_VALID_EXAMPLES ?= 20
DEBUG_MAX_STEPS ?= 20
DEBUG_BATCH_SIZE ?= 8
DEBUG_MAX_SRC_LEN ?= 128
DEBUG_MAX_TGT_LEN ?= 32
DEBUG_D_MODEL ?= 128
DEBUG_NUM_HEADS ?= 4
DEBUG_ENCODER_LAYERS ?= 2
DEBUG_DECODER_LAYERS ?= 2
DEBUG_D_FF ?= 512

export CHECKPOINT_PATH
export DEVICE

.PHONY: install check train-debug train serve serve-debug

install:
	uv sync

check:
	$(PYTHON) -m py_compile tiny_transformer_summarizer.py app.py

train-debug:
	$(PYTHON) tiny_transformer_summarizer.py \
		--train \
		--train-examples $(DEBUG_TRAIN_EXAMPLES) \
		--valid-examples $(DEBUG_VALID_EXAMPLES) \
		--max-src-len $(DEBUG_MAX_SRC_LEN) \
		--max-tgt-len $(DEBUG_MAX_TGT_LEN) \
		--batch-size $(DEBUG_BATCH_SIZE) \
		--d-model $(DEBUG_D_MODEL) \
		--num-heads $(DEBUG_NUM_HEADS) \
		--num-encoder-layers $(DEBUG_ENCODER_LAYERS) \
		--num-decoder-layers $(DEBUG_DECODER_LAYERS) \
		--d-ff $(DEBUG_D_FF) \
		--max-steps $(DEBUG_MAX_STEPS) \
		--checkpoint-path $(DEBUG_CHECKPOINT_PATH) \
		--device $(DEVICE)

train:
	$(PYTHON) tiny_transformer_summarizer.py \
		--train \
		--train-examples $(TRAIN_EXAMPLES) \
		--valid-examples $(VALID_EXAMPLES) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--max-src-len $(MAX_SRC_LEN) \
		--max-tgt-len $(MAX_TGT_LEN) \
		--max-vocab-size $(MAX_VOCAB_SIZE) \
		--d-model $(D_MODEL) \
		--num-heads $(NUM_HEADS) \
		--num-encoder-layers $(ENCODER_LAYERS) \
		--num-decoder-layers $(DECODER_LAYERS) \
		--d-ff $(D_FF) \
		--lr $(LR) \
		--checkpoint-path $(CHECKPOINT_PATH) \
		--device $(DEVICE)

serve:
	$(PYTHON) -m uvicorn app:app --reload

serve-debug:
	CHECKPOINT_PATH=$(DEBUG_CHECKPOINT_PATH) $(PYTHON) -m uvicorn app:app --reload
