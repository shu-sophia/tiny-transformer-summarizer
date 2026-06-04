import os
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from tiny_transformer_summarizer import (
    Config,
    CharTokenizer,
    TransformerSummarizer,
    build_model_from_checkpoint,
    decode_title,
    get_device,
)


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1)


class GenerateResponse(BaseModel):
    generated_title: str


class HealthResponse(BaseModel):
    checkpoint_path: str
    device: str
    model_loaded: bool
    error: str | None = None


class ModelService:
    def __init__(self) -> None:
        self.checkpoint_path = Path(
            os.getenv("CHECKPOINT_PATH", "checkpoints/tiny_transformer.pt")
        )
        self.device = get_device(os.getenv("DEVICE", "auto"))
        self.model: TransformerSummarizer | None = None
        self.tokenizer: CharTokenizer | None = None
        self.config: Config | None = None
        self.error: str | None = None
        self.load()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None and self.config is not None

    def load(self) -> None:
        try:
            model, tokenizer, config = build_model_from_checkpoint(
                self.checkpoint_path,
                self.device,
            )
            self.model = model
            self.tokenizer = tokenizer
            self.config = config
            self.error = None
        except FileNotFoundError as exc:
            self.error = str(exc)
        except RuntimeError as exc:
            self.error = f"failed to load checkpoint: {exc}"

    def generate(self, text: str) -> str:
        if not self.model_loaded:
            raise HTTPException(
                status_code=503,
                detail=self.error or "model is not loaded",
            )
        assert self.model is not None
        assert self.tokenizer is not None
        assert self.config is not None
        with torch.inference_mode():
            return decode_title(
                self.model,
                self.tokenizer,
                text,
                self.config,
                self.device,
            )


app = FastAPI(title="Tiny Transformer Summarizer")
model_service = ModelService()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        checkpoint_path=str(model_service.checkpoint_path),
        device=str(model_service.device),
        model_loaded=model_service.model_loaded,
        error=model_service.error,
    )


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse("web/index.html")


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    generated_title = model_service.generate(text)
    return GenerateResponse(generated_title=generated_title)
