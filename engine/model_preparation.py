"""Cold application catalog for local, model-owned asset preparers."""
from pathlib import Path
from typing import Protocol

from .asset_store import AssetStore
from .models.qwen35.runtime_assets import matches_source, prepare_assets

# Registration only. Each provider validates source meanings and its own
# generated assets; request/HTTP modules never interpret a weight manifest.
_PREPARERS = ((matches_source, prepare_assets),)


class PreparedAssetFiles(Protocol):
    directory: Path
    created: bool
    @property
    def execution_assets(self) -> tuple[tuple[str, Path], ...]: ...
    @property
    def artifacts(self) -> tuple[tuple[str, Path], ...]: ...


def prepare_local_assets(inputs: Path, tokenizer_directory: Path, tokenizer, directory: Path, **recipe) -> PreparedAssetFiles:
    selected = [prepare for matches, prepare in _PREPARERS if matches(inputs)]
    if len(selected) != 1:
        raise ValueError('no unique local asset preparer recognizes the model source')
    return selected[0](inputs, tokenizer_directory, tokenizer, AssetStore(directory), **recipe)
