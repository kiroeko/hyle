"""Qwen's source-bound original-ID draft projection file format."""


def write_vocabulary(output, identity: str, vocabulary: int, ids: list[int],
                     recommended_drafts: int | None) -> None:
    recommendation = 'none' if recommended_drafts is None else str(recommended_drafts)
    output.write(f'hyle-draft-vocabulary-v2\n{identity}\n{vocabulary}\n'
                 f'recommended_drafts {recommendation}\nhead original\n')
    output.write(' '.join(map(str, ids)) + '\n')
