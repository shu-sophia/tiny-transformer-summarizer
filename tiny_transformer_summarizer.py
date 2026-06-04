import argparse
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass
class Config:
    train_examples: int = 100
    valid_examples: int = 50
    max_src_len: int = 256
    max_tgt_len: int = 64
    batch_size: int = 8
    max_vocab_size: int = 8000
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    lr: float = 3e-4
    epochs: int = 1
    max_steps: int = 0
    log_every: int = 10
    checkpoint_path: str = "checkpoints/tiny_transformer.pt"
    decode_strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    device: str = "auto"
    seed: int = 42


class CharTokenizer:
    def __init__(self, char_to_id: dict[str, int]):
        self.char_to_id = char_to_id
        self.id_to_char = {token_id: char for char, token_id in char_to_id.items()}

        self.pad_id = self.char_to_id[PAD_TOKEN]
        self.bos_id = self.char_to_id[BOS_TOKEN]
        self.eos_id = self.char_to_id[EOS_TOKEN]
        self.unk_id = self.char_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id.get(char, self.unk_id) for char in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        chars = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            chars.append(self.id_to_char.get(int(token_id), UNK_TOKEN))
        return "".join(chars)


class TitleDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer: CharTokenizer,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.data[index]
        src_ids = encode_source(
            example["text"],
            self.tokenizer,
            self.max_src_len,
        )
        tgt_input, tgt_labels = encode_target(
            example["title"],
            self.tokenizer,
            self.max_tgt_len,
        )

        return {
            # src_ids: [S]
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            # tgt_input: [T]
            "tgt_input": torch.tensor(tgt_input, dtype=torch.long),
            # tgt_labels: [T]
            "tgt_labels": torch.tensor(tgt_labels, dtype=torch.long),
        }


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.scale = math.sqrt(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: [B, L] -> embedded: [B, L, d_model]
        return self.embedding(token_ids) * self.scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, L, d_model]
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, H, L, head_dim]
        batch_size, seq_len, _d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, head_dim] -> [B, L, d_model]
        batch_size, _num_heads, seq_len, _head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query: [B, Q, d_model], key/value: [B, K, d_model]
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))

        # scores: [B, H, Q, K]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context: [B, H, Q, head_dim]
        context = torch.matmul(attn_weights, v)
        context = self.combine_heads(context)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, L, d_model]
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, S, d_model]
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, d_model], memory: [B, S, d_model]
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_out))

        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))

        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.position = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_ids: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # src_ids: [B, S] -> x: [B, S, d_model]
        x = self.token_embedding(src_ids)
        x = self.position(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, pad_id)
        self.position = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt_input: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        # tgt_input: [B, T] -> x: [B, T, d_model]
        x = self.token_embedding(tgt_input)
        x = self.position(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x


class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, config: Config):
        super().__init__()
        self.pad_id = pad_id
        max_len = max(config.max_src_len, config.max_tgt_len)

        self.encoder = Encoder(
            vocab_size,
            pad_id,
            config.d_model,
            config.num_heads,
            config.num_encoder_layers,
            config.d_ff,
            config.dropout,
            max_len,
        )
        self.decoder = Decoder(
            vocab_size,
            pad_id,
            config.d_model,
            config.num_heads,
            config.num_decoder_layers,
            config.d_ff,
            config.dropout,
            max_len,
        )
        self.output_layer = nn.Linear(config.d_model, vocab_size)

    def make_masks(
        self,
        src_ids: torch.Tensor,
        tgt_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # src_mask: [B, 1, 1, S], True means attention may look at that position.
        src_mask = create_padding_mask(src_ids, self.pad_id)

        # tgt_padding_mask: [B, 1, 1, T]
        # causal_mask:      [1, 1, T, T]
        tgt_padding_mask = create_padding_mask(tgt_input, self.pad_id)
        causal_mask = create_causal_mask(tgt_input.size(1), tgt_input.device)
        tgt_mask = tgt_padding_mask & causal_mask
        return src_mask, tgt_mask

    def forward(self, src_ids: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        # src_ids: [B, S], tgt_input: [B, T]
        src_mask, tgt_mask = self.make_masks(src_ids, tgt_input)
        memory = self.encoder(src_ids, src_mask)
        decoder_out = self.decoder(tgt_input, memory, tgt_mask, src_mask)
        # logits: [B, T, vocab_size]
        return self.output_layer(decoder_out)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preview_text(value: str, max_chars: int = 120) -> str:
    value = value.replace("\n", " ").strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def load_xlsum_japanese(config: Config):
    train_split = f"train[:{config.train_examples}]"
    valid_split = f"validation[:{config.valid_examples}]"

    train_data = load_dataset(
        "csebuetnlp/xlsum",
        "japanese",
        split=train_split,
        trust_remote_code=True,
    )
    valid_data = load_dataset(
        "csebuetnlp/xlsum",
        "japanese",
        split=valid_split,
        trust_remote_code=True,
    )

    return train_data, valid_data


def inspect_examples(train_data, valid_data) -> None:
    print("=== Dataset ===")
    print(f"train examples: {len(train_data)}")
    print(f"valid examples: {len(valid_data)}")

    sample = train_data[0]
    print("\n=== Sample ===")
    print(f"title: {preview_text(sample['title'])}")
    print(f"text:  {preview_text(sample['text'])}")
    print(f"text chars: {len(sample['text'])}")


def build_tokenizer(train_data, max_vocab_size: int) -> CharTokenizer:
    counter: Counter[str] = Counter()
    for example in train_data:
        counter.update(example["text"])
        counter.update(example["title"])

    vocab_chars = [
        char
        for char, _count in counter.most_common(max_vocab_size - len(SPECIAL_TOKENS))
    ]
    char_to_id = {token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)}
    for char in vocab_chars:
        if char not in char_to_id:
            char_to_id[char] = len(char_to_id)

    return CharTokenizer(char_to_id)


def pad_to_length(token_ids: list[int], max_len: int, pad_id: int) -> list[int]:
    if len(token_ids) >= max_len:
        return token_ids[:max_len]
    return token_ids + [pad_id] * (max_len - len(token_ids))


def encode_source(
    text: str,
    tokenizer: CharTokenizer,
    max_src_len: int,
) -> list[int]:
    # source: text -> src_ids: [S]
    body_len = max_src_len - 1
    token_ids = tokenizer.encode(text)[:body_len]
    token_ids = token_ids + [tokenizer.eos_id]
    return pad_to_length(token_ids, max_src_len, tokenizer.pad_id)


def encode_target(
    title: str,
    tokenizer: CharTokenizer,
    max_tgt_len: int,
) -> tuple[list[int], list[int]]:
    # full target: [BOS, token..., EOS, PAD...] with length T + 1
    body_len = max_tgt_len - 1
    token_ids = tokenizer.encode(title)[:body_len]
    full_ids = [tokenizer.bos_id] + token_ids + [tokenizer.eos_id]
    full_ids = pad_to_length(full_ids, max_tgt_len + 1, tokenizer.pad_id)

    # tgt_input:  [BOS, token..., EOS/PAD...] -> [T]
    # tgt_labels: [token..., EOS, PAD...]     -> [T]
    return full_ids[:-1], full_ids[1:]


def create_dataloaders(
    train_data,
    valid_data,
    tokenizer: CharTokenizer,
    config: Config,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = TitleDataset(
        train_data,
        tokenizer,
        config.max_src_len,
        config.max_tgt_len,
    )
    valid_dataset = TitleDataset(
        valid_data,
        tokenizer,
        config.max_src_len,
        config.max_tgt_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader


def inspect_tokenizer(tokenizer: CharTokenizer, train_data) -> None:
    sample = train_data[0]
    title_ids = tokenizer.encode(sample["title"])
    text_ids = tokenizer.encode(sample["text"])

    print("\n=== Tokenizer ===")
    print(f"vocab size: {tokenizer.vocab_size}")
    print(f"title ids[:20]: {title_ids[:20]}")
    print(f"decoded title: {preview_text(tokenizer.decode(title_ids))}")
    print(f"text ids[:20]: {text_ids[:20]}")


def inspect_batch(batch: dict[str, torch.Tensor], tokenizer: CharTokenizer) -> None:
    print("\n=== Batch ===")
    print(f"src_ids.shape:    {list(batch['src_ids'].shape)}")
    print(f"tgt_input.shape:  {list(batch['tgt_input'].shape)}")
    print(f"tgt_labels.shape: {list(batch['tgt_labels'].shape)}")

    src0 = batch["src_ids"][0].tolist()
    tgt_input0 = batch["tgt_input"][0].tolist()
    tgt_labels0 = batch["tgt_labels"][0].tolist()

    print(f"src decoded:       {preview_text(tokenizer.decode(src0))}")
    print(f"tgt_input decoded: {preview_text(tokenizer.decode(tgt_input0))}")
    print(f"tgt_label decoded: {preview_text(tokenizer.decode(tgt_labels0))}")


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("cuda was requested, but CUDA is not available. Falling back to cpu.")
        device_name = "cpu"
    return torch.device(device_name)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def create_padding_mask(token_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # token_ids: [B, L] -> mask: [B, 1, 1, L]
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # mask: [1, 1, T, T]
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def inspect_device(batch: dict[str, torch.Tensor], device: torch.device) -> None:
    device_batch = move_batch_to_device(batch, device)
    print("\n=== Device ===")
    print(f"selected device: {device}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"src_ids device: {device_batch['src_ids'].device}")


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def inspect_model_and_masks(
    model: TransformerSummarizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    device_batch = move_batch_to_device(batch, device)
    src_mask, tgt_mask = model.make_masks(
        device_batch["src_ids"],
        device_batch["tgt_input"],
    )

    print("\n=== Model ===")
    print(f"trainable parameters: {count_trainable_parameters(model):,}")
    print(f"src_mask.shape: {list(src_mask.shape)}")
    print(f"tgt_mask.shape: {list(tgt_mask.shape)}")


def run_forward_check(
    model: TransformerSummarizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    device_batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        logits = model(device_batch["src_ids"], device_batch["tgt_input"])

    print("\n=== Forward Check ===")
    print(f"src_ids.shape:   {list(device_batch['src_ids'].shape)}")
    print(f"tgt_input.shape: {list(device_batch['tgt_input'].shape)}")
    print(f"logits.shape:    {list(logits.shape)}")
    print(f"last logits:     {list(logits[:, -1, :].shape)}")
    return logits


def compute_loss(
    logits: torch.Tensor,
    tgt_labels: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # logits: [B, T, V] -> [B*T, V]
    # labels: [B, T]    -> [B*T]
    return loss_fn(
        logits.reshape(-1, vocab_size),
        tgt_labels.reshape(-1),
    )


def run_loss_check(
    model: TransformerSummarizer,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    pad_id: int,
) -> torch.Tensor:
    model.eval()
    device_batch = move_batch_to_device(batch, device)
    with torch.no_grad():
        logits = model(device_batch["src_ids"], device_batch["tgt_input"])
        loss = compute_loss(logits, device_batch["tgt_labels"], pad_id)

    print("\n=== Loss Check ===")
    print(f"logits for loss: {list(logits.reshape(-1, logits.size(-1)).shape)}")
    print(f"labels for loss: {list(device_batch['tgt_labels'].reshape(-1).shape)}")
    print(f"loss: {loss.item():.4f}")
    print(f"loss is finite: {torch.isfinite(loss).item()}")
    return loss


def train_model(
    model: TransformerSummarizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    pad_id: int,
    tokenizer: CharTokenizer,
    config: Config,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch}")

        for batch in progress:
            device_batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(device_batch["src_ids"], device_batch["tgt_input"])
            loss = compute_loss(logits, device_batch["tgt_labels"], pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            global_step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")

            if global_step == 1 or global_step % config.log_every == 0:
                print(f"step {global_step}: loss={loss.item():.4f}")

            if config.max_steps > 0 and global_step >= config.max_steps:
                print(f"stopped at max_steps={config.max_steps}")
                valid_loss = evaluate_loss(model, valid_loader, device, pad_id)
                print(f"valid loss: {valid_loss:.4f}")
                save_checkpoint(model, tokenizer, config)
                return

        valid_loss = evaluate_loss(model, valid_loader, device, pad_id)
        print(f"epoch {epoch} valid loss: {valid_loss:.4f}")

    save_checkpoint(model, tokenizer, config)


def evaluate_loss(
    model: TransformerSummarizer,
    data_loader: DataLoader,
    device: torch.device,
    pad_id: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            device_batch = move_batch_to_device(batch, device)
            logits = model(device_batch["src_ids"], device_batch["tgt_input"])
            loss = compute_loss(logits, device_batch["tgt_labels"], pad_id)
            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        return float("nan")
    return total_loss / total_batches


def save_checkpoint(
    model: TransformerSummarizer,
    tokenizer: CharTokenizer,
    config: Config,
) -> None:
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_id": tokenizer.char_to_id,
            "config": asdict(config),
        },
        checkpoint_path,
    )
    print(f"saved checkpoint: {checkpoint_path}")


def config_from_dict(config_dict: dict) -> Config:
    valid_keys = set(Config.__dataclass_fields__)
    filtered_config = {
        key: value for key, value in config_dict.items() if key in valid_keys
    }
    return Config(**filtered_config)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TransformerSummarizer, CharTokenizer, Config]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = config_from_dict(checkpoint["config"])
    tokenizer = CharTokenizer(checkpoint["char_to_id"])
    model = TransformerSummarizer(
        tokenizer.vocab_size,
        tokenizer.pad_id,
        config,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, config


def filter_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _indices = torch.topk(logits, top_k)
        min_value = values[-1]
        logits = logits.masked_fill(logits < min_value, -1e9)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove_mask = cumulative_probs > top_p
        remove_mask[1:] = remove_mask[:-1].clone()
        remove_mask[0] = False

        sorted_logits = sorted_logits.masked_fill(remove_mask, -1e9)
        filtered_logits = torch.full_like(logits, -1e9)
        filtered_logits.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered_logits

    return logits


def choose_next_token(next_logits: torch.Tensor, config: Config) -> int:
    if config.decode_strategy == "greedy":
        return int(torch.argmax(next_logits).item())

    filtered_logits = filter_logits(
        next_logits,
        config.temperature,
        config.top_k,
        config.top_p,
    )
    probs = torch.softmax(filtered_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def decode_title(
    model: TransformerSummarizer,
    tokenizer: CharTokenizer,
    src_text: str,
    config: Config,
    device: torch.device,
) -> str:
    model.eval()

    src_ids = encode_source(src_text, tokenizer, config.max_src_len)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    generated = [tokenizer.bos_id]

    with torch.no_grad():
        for _step in range(config.max_tgt_len):
            # tgt_tensor: [1, current_tgt_len]
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            logits = model(src_tensor, tgt_tensor)
            # next_logits: [vocab_size]
            next_logits = logits[0, -1]
            next_id = choose_next_token(next_logits, config)

            if next_id in {tokenizer.eos_id, tokenizer.pad_id}:
                break
            generated.append(next_id)

    return tokenizer.decode(generated)


def generate_sample(
    model: TransformerSummarizer,
    tokenizer: CharTokenizer,
    train_data,
    config: Config,
    device: torch.device,
) -> None:
    sample = train_data[0]
    generated = decode_title(model, tokenizer, sample["text"], config, device)

    print("\n=== Decoding ===")
    print(f"strategy:  {config.decode_strategy}")
    print(f"source:    {preview_text(sample['text'])}")
    print(f"gold:      {sample['title']}")
    print(f"generated: {generated}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-examples", type=int, default=100)
    parser.add_argument("--valid-examples", type=int, default=50)
    parser.add_argument("--max-src-len", type=int, default=256)
    parser.add_argument("--max-tgt-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-vocab-size", type=int, default=8000)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/tiny_transformer.pt",
    )
    parser.add_argument(
        "--decode-strategy",
        choices=["greedy", "sample"],
        default="greedy",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--forward-check", action="store_true")
    parser.add_argument("--loss-check", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        train_examples=args.train_examples,
        valid_examples=args.valid_examples,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        batch_size=args.batch_size,
        max_vocab_size=args.max_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        max_steps=args.max_steps,
        log_every=args.log_every,
        checkpoint_path=args.checkpoint_path,
        decode_strategy=args.decode_strategy,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device,
        seed=args.seed,
    )

    set_seed(config.seed)
    device = get_device(config.device)
    train_data, valid_data = load_xlsum_japanese(config)
    inspect_examples(train_data, valid_data)
    tokenizer = build_tokenizer(train_data, config.max_vocab_size)
    inspect_tokenizer(tokenizer, train_data)
    train_loader, _valid_loader = create_dataloaders(
        train_data,
        valid_data,
        tokenizer,
        config,
    )
    batch = next(iter(train_loader))
    inspect_batch(batch, tokenizer)
    inspect_device(batch, device)
    model = TransformerSummarizer(tokenizer.vocab_size, tokenizer.pad_id, config).to(device)
    inspect_model_and_masks(model, batch, device)
    if args.forward_check:
        run_forward_check(model, batch, device)
    if args.loss_check:
        run_loss_check(model, batch, device, tokenizer.pad_id)
    if args.train:
        train_model(model, train_loader, _valid_loader, device, tokenizer.pad_id, tokenizer, config)
    if args.generate:
        generate_sample(model, tokenizer, train_data, config, device)


if __name__ == "__main__":
    main()
