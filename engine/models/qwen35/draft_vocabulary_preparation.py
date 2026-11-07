"""Source-bound MTP vocabulary preparation with no model execution."""
from __future__ import annotations

from collections import Counter
import hashlib
import json

PROMPTS = (
    "Explain how ocean tides arise from gravity. Include spring and neap tides and explain why coastlines change their height.",
    "Describe how a seed becomes a flowering plant, including photosynthesis, roots, pollination and the movement of water.",
    "Give a worked example of Bayes' theorem using a medical screening test. Clearly distinguish sensitivity and posterior probability.",
    "Explain the lifecycle of a star like the Sun, from a cloud of gas to a white dwarf, for a curious teenager.",
    "Write a concise guide to measuring electrical resistance, voltage and current safely on a low voltage breadboard.",
    "Compare induction, deduction and falsification using concrete examples from scientific experiments.",
    "Write a C++20 example of a move-only RAII file descriptor wrapper. Explain ownership transfer and error handling.",
    "Write Python code that uses asyncio to fetch independent tasks with bounded concurrency, cancellation and timeouts.",
    "Explain database transaction isolation with a bank transfer example. Compare read committed and serializable isolation.",
    "Describe a reliable method to debug an intermittent memory leak in a long running service, including tools and measurements.",
    "Explain how HTTP caching works, including ETag, Cache-Control, conditional requests and shared caches.",
    "Write a Rust example that parses a configuration file and returns useful typed errors without panicking.",
    "用中文解释地震波为什么可以帮助科学家了解地球内部结构，包括纵波、横波和液态外核。",
    "请写一份家用阳台香草种植指南，包括薄荷、罗勒和迷迭香的日照、浇水和修剪方法。",
    "用中文介绍纸张从植物纤维到成品的制造过程，并解释再生纸的优点和限制。",
    "请讲解如何阅读一份企业现金流量表，区分经营、投资和筹资活动，并举具体例子。",
    "用中文说明数码照片的曝光、快门、光圈、感光度和景深之间的关系。",
    "请写一篇关于传统木工榫卯结构的科普短文，解释其受力方式和常见连接形式。",
    "日本語で、火山の噴火の仕組みと防災情報の読み方を説明してください。",
    "Expliquez en français comment fonctionne une pompe à chaleur et dans quelles conditions son rendement varie.",
    "Explica en español las diferencias entre interés simple y compuesto con un ejemplo de ahorro durante cinco años.",
    "Erkläre auf Deutsch, wie ein Fahrradgetriebe funktioniert und wann kleine oder große Gänge sinnvoll sind.",
    "اشرح باللغة العربية كيف تتكون السحب ولماذا تختلف أنواعها، مع أمثلة مبسطة.",
    "Explain how to plan a small community library: cataloguing, lending rules, accessibility and preserving old books.",
)



def selected_ids(counts: Counter[int], vocabulary: int, rows: int) -> list[int]:
    if not counts or not 0 < rows <= vocabulary:
        raise ValueError("nonempty calibration and valid row count required")
    if any(type(token) is not int or not 0 <= token < vocabulary or type(count) is not int or count <= 0
           for token, count in counts.items()):
        raise ValueError("invalid calibration counts")
    return sorted(sorted(range(vocabulary), key=lambda token: (-counts[token], token))[:rows])


def write_vocabulary(output, identity: str, vocabulary: int, ids: list[int],
                     recommended_drafts: int | None) -> None:
    recommendation = 'none' if recommended_drafts is None else str(recommended_drafts)
    output.write(f'hyle-draft-vocabulary-v2\n{identity}\n{vocabulary}\n'
                 f'recommended_drafts {recommendation}\nhead original\n')
    output.write(' '.join(map(str, ids)) + '\n')

def tokenizer_default_ids(tokenizer, vocabulary: int) -> tuple[list[int], str]:
    """Select bounded common IDs from local tokenizer and preparation prompts."""
    if type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError("positive source vocabulary required")
    special = set(tokenizer.all_special_ids)
    if any(type(token) is not int or not 0 <= token < vocabulary for token in special):
        raise ValueError("tokenizer special token is outside the source vocabulary")
    counts: Counter[int] = Counter()
    encoded_prompts = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or any(type(token) is not int or not 0 <= token < vocabulary for token in ids):
            raise ValueError("tokenizer prompt contains IDs outside the source vocabulary")
        encoded_prompts.append(ids)
        counts.update(ids)
    rows = min(vocabulary, max(len(special), (vocabulary + 1) // 2))
    selected = set(special)
    for token in sorted(counts, key=lambda value: (-counts[value], value)):
        if len(selected) == rows:
            break
        selected.add(token)
    for token in range(vocabulary):
        if len(selected) == rows:
            break
        selected.add(token)
    fingerprint = hashlib.sha256(json.dumps(dict(special=sorted(special), prompts=encoded_prompts),
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return sorted(selected), fingerprint
