"""Cold application catalog for local, model-owned asset preparers."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Protocol

from .asset_store import AssetStore
from .models.qwen35.runtime_assets import matches_source, prepare_assets, validate_tokenizer, source_identity
from .models.qwen35.draft_vocabulary_preparation import write_vocabulary
from .models.qwen35.vision_preparation import prepare_vision
from .models.qwen35.vision_processing import ImageProcessor

class PreparedAssetFiles(Protocol):
    directory: Path
    created: bool
    @property
    def execution_assets(self) -> tuple[tuple[str, Path], ...]: ...
    @property
    def artifacts(self) -> tuple[tuple[str, Path], ...]: ...


@dataclass(frozen=True)
class LocalPreparationProvider:
    matches: Callable[[Path], bool]
    validate_tokenizer: Callable[..., tuple[str, int]]
    source_identity: Callable[[Path], tuple[str, int]]
    # Independent optional operations. A text-only source does not acquire
    # draft or visual requirements just by entering the application catalog.
    prepare: Callable[..., PreparedAssetFiles] | None = None
    write_vocabulary: Callable[..., None] | None = None
    prepare_vision: Callable[..., str] | None = None
    image_processor: Callable[..., object] | None = None


# Registration only; neither the request layer nor HTTP interprets a manifest.
_PREPARERS = (LocalPreparationProvider(matches=matches_source, prepare=prepare_assets,
    validate_tokenizer=validate_tokenizer, source_identity=source_identity, write_vocabulary=write_vocabulary,
    prepare_vision=prepare_vision, image_processor=ImageProcessor),)


def _select(inputs: Path) -> LocalPreparationProvider:
    selected = [provider for provider in _PREPARERS if provider.matches(inputs)]
    if len(selected) != 1:
        raise ValueError('no unique local preparer recognizes the model source')
    return selected[0]


def prepare_local_assets(inputs: Path, tokenizer_directory: Path, tokenizer, directory: Path, **recipe) -> PreparedAssetFiles:
    prepare = _select(inputs).prepare
    if prepare is None:
        raise ValueError('selected model does not support derived assets')
    return prepare(inputs, tokenizer_directory, tokenizer, AssetStore(directory), **recipe)


def validate_local_tokenizer(inputs: Path, tokenizer) -> tuple[str, int]:
    return _select(inputs).validate_tokenizer(inputs, tokenizer)


def prepared_source_identity(inputs: Path) -> tuple[str, int]:
    return _select(inputs).source_identity(inputs)


def prepare_local_vision(inputs: Path, model: Path, output: Path, *, mmproj: Path | None = None) -> str:
    prepare = _select(inputs).prepare_vision
    if prepare is None:
        raise ValueError('selected model does not support vision assets')
    return prepare(model, inputs, output, mmproj=mmproj)


def local_image_processor(inputs: Path, model: Path, tokenizer, *, context: int):
    process = _select(inputs).image_processor
    if process is None:
        raise ValueError('selected model does not support image processing')
    return process(model, tokenizer, context=context)


def write_prepared_vocabulary(inputs: Path, output, ids: list[int], recommended_drafts: int | None, *,
                              expected_source: tuple[str, int]) -> None:
    provider = _select(inputs)
    if provider.write_vocabulary is None:
        raise ValueError('selected model does not support vocabulary writing')
    identity, vocabulary = provider.source_identity(inputs)
    if (identity, vocabulary) != expected_source:
        raise ValueError('prepared source changed after calibration')
    provider.write_vocabulary(output, identity, vocabulary, ids, recommended_drafts)
