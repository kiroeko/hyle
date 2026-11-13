"""Local model text rendering, tokenization and incremental detokenization."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, Sequence


def has_images(messages):
    return any(isinstance(message.get('content'), list) and any(
        part.get('type') == 'image' for part in message['content']) for message in messages)


class DecodeSession(Protocol):
    def push(self, token: int) -> str: ...
    def finish(self) -> str: ...


class TextCodec(Protocol):
    @property
    def stop_ids(self) -> frozenset[int]: ...
    def encode_messages(self, messages: Sequence[dict[str, Any]], *, continue_final_message: bool = False) -> list[int]: ...
    def new_decoder(self) -> DecodeSession: ...


class HuggingFaceTextCodec:
    """Use existing local Hugging Face tokenizer and model-supplied Jinja.

    The caller must choose assets belonging to the loaded model. No network or
    remote model code is used. A custom template may be supplied explicitly.
    """
    def __init__(self, directory: Path, *, template: Path | None = None,
                 expected_vocabulary: int | None = None, image_processor=None):
        directory = Path(directory).resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("tokenizer path is not a directory")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_fast=True)
        if not getattr(self._tokenizer, "is_fast", False):
            raise ValueError("the local tokenizer must support incremental decoding")
        if expected_vocabulary is not None:
            vocabulary = self._tokenizer.get_vocab().values()
            if expected_vocabulary <= 0 or any(type(value) is not int or value < 0 or value >= expected_vocabulary
                                               for value in vocabulary):
                raise ValueError("tokenizer contains IDs outside the model vocabulary")
        self._template = Path(template).read_text(encoding="utf-8") if template is not None else None
        if self._template == "":
            raise ValueError("explicit chat template is empty")
        if self._template is None and not self._tokenizer.chat_template:
            raise ValueError("model has no chat template; provide one explicitly")
        stop = set()
        self._add_stop(stop, self._tokenizer.eos_token_id)
        # Model and tokenizer may name different end tokens. Read only local
        # model metadata; do not treat every special token as a stop token.
        for name in ("config.json", "generation_config.json"):
            path = directory / name
            if not path.is_file():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("model configuration must be an object")
            self._add_stop(stop, data.get("eos_token_id"))
            text_config = data.get("text_config")
            if isinstance(text_config, dict):
                self._add_stop(stop, text_config.get("eos_token_id"))
        if expected_vocabulary is not None and any(token >= expected_vocabulary for token in stop):
            raise ValueError("model end-token ID is outside the model vocabulary")
        self._stop_ids = frozenset(stop)
        self.image_processor = image_processor

    @staticmethod
    def _add_stop(target: set[int], value: Any) -> None:
        if value is None:
            return
        values = value if isinstance(value, list) else [value]
        if any(type(item) is not int or item < 0 for item in values):
            raise ValueError("model end-token metadata is invalid")
        target.update(values)

    @property
    def stop_ids(self) -> frozenset[int]:
        return self._stop_ids

    @property
    def vocabulary_size(self) -> int:
        return len(self._tokenizer)

    @property
    def preparation_tokenizer(self):
        """Cold asset preparation borrows the same local tokenizer/template."""
        return self._tokenizer

    def validate_vocabulary(self, vocabulary: int) -> None:
        if vocabulary <= 0 or any(type(index) is not int or not 0 <= index < vocabulary
                                  for index in self._tokenizer.get_vocab().values()) or any(index >= vocabulary for index in self.stop_ids):
            raise ValueError('tokenizer or end-token IDs exceed the loaded model vocabulary')

    def encode_messages(self, messages: Sequence[dict[str, Any]], *, continue_final_message: bool = False) -> list[int]:
        if not messages:
            raise ValueError("at least one message is required")
        encoded = self._tokenizer.apply_chat_template(
            list(messages), chat_template=self._template, tokenize=True,
            add_generation_prompt=not continue_final_message, continue_final_message=continue_final_message,
            return_dict=True)
        ids = encoded["input_ids"]
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError("template did not produce one token sequence")
        return ids

    def prepare_images(self, messages, *, continue_final_message=False, max_tokens=None):
        if self.image_processor is None:
            raise ValueError('no image execution recipe was prepared for this model')
        return self.image_processor.prepare(messages, template=self._template,
            continue_final_message=continue_final_message,max_tokens=max_tokens)

    def continue_messages(self, tape: Sequence[int], messages: Sequence[dict[str, Any]]) -> list[int] | None:
        # Tokenization at a text boundary need not reproduce generated IDs.
        # Prove the complete rendered prefix, then preserve its actual tokens
        # and tokenize only the new suffix. Arbitrary Jinja rewrites fall back
        # to ordinary complete rendering instead of guessing delimiters.
        rendered = self._tokenizer.apply_chat_template(list(messages), chat_template=self._template,
            tokenize=False, add_generation_prompt=True)
        prefix = self._tokenizer.decode(list(tape), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not rendered.startswith(prefix):
            return None
        return list(tape) + self._tokenizer.encode(rendered[len(prefix):], add_special_tokens=False)

    def history_message(self, ids: Sequence[int], text: str) -> dict[str, str]:
        # Keep template-significant generated markers in saved model context;
        # presentation decoding can independently omit special token strings.
        content = list(ids)
        if content and content[-1] in self.stop_ids:
            content.pop()
        return {'role': 'assistant', 'content': self._tokenizer.decode(content, skip_special_tokens=False,
                                                                     clean_up_tokenization_spaces=False)}

    def new_decoder(self) -> DecodeSession:
        return _HuggingFaceDecodeSession(self._tokenizer)


class _HuggingFaceDecodeSession:
    def __init__(self, tokenizer: Any):
        from tokenizers.decoders import DecodeStream
        self._tokenizer = tokenizer
        self._stream = DecodeStream(skip_special_tokens=True)
        self._ids: list[int] = []
        self._emitted = ""
        self._finished = False

    def push(self, token: int) -> str:
        if self._finished:
            raise RuntimeError("decoder has already finished")
        self._ids.append(token)
        chunk = self._stream.step(self._tokenizer.backend_tokenizer, token) or ""
        self._emitted += chunk
        return chunk

    def finish(self) -> str:
        if self._finished:
            raise RuntimeError("decoder has already finished")
        self._finished = True
        final = self._tokenizer.decode(self._ids, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
        if not final.startswith(self._emitted):
            raise ValueError("streaming tokenizer output differs from final decode")
        return final[len(self._emitted):]
