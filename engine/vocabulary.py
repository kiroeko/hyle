"""Model-independent language corpora and selection of unchanged token IDs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re

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

_PROMPT_GROUPS = (
    'en', 'en', 'en', 'en', 'en', 'en', 'code', 'code', 'en', 'en', 'en', 'code',
    'zh', 'zh', 'zh', 'zh', 'zh', 'zh', 'ja', 'fr', 'es', 'de', 'ar', 'en',
)

# Optional language profiles use distinct local prompts. Code examples are
# shared across profiles; the target's full vocabulary is never restricted.
_ADDITIONAL_PROMPTS = {
    'ar': (
        'اشرح باللغة العربية كيف تعمل الألواح الشمسية وكيف يمكن ترشيد استهلاك الكهرباء في المنزل.',
        'اكتب دليلاً بالعربية عن إعداد وجبة صحية باستخدام الخضار الموسمية مع خطوات واضحة.',
        'اشرح بالعربية أهمية النوم وتأثيره على الذاكرة والتركيز مع أمثلة يومية.',
    ),
    'ko': (
        '한국어로 태양광 발전의 원리와 가정에서 전기를 절약하는 방법을 자세히 설명해 주세요.',
        '한국어로 제철 채소를 이용한 건강한 식사를 준비하는 방법을 단계별로 알려 주세요.',
        '한국어로 수면이 기억력과 집중력에 미치는 영향을 일상적인 예와 함께 설명해 주세요.',
    ),
    'th': (
        'อธิบายเป็นภาษาไทยว่าแผงโซลาร์เซลล์ผลิตไฟฟ้าอย่างไรและวิธีประหยัดพลังงานในบ้าน',
        'เขียนคำแนะนำภาษาไทยในการทำอาหารเพื่อสุขภาพจากผักตามฤดูกาลเป็นขั้นตอน',
        'อธิบายเป็นภาษาไทยว่าการนอนหลับส่งผลต่อความจำและสมาธิอย่างไร พร้อมตัวอย่าง',
    ),
    'ru': (
        'Объясни по-русски, как солнечные панели производят электричество и как экономить энергию дома.',
        'Напиши по-русски пошаговое руководство по приготовлению здоровой еды из сезонных овощей.',
        'Объясни на русском, как сон влияет на память и концентрацию, с бытовыми примерами.',
    ),
    'vi': (
        'Hãy giải thích bằng tiếng Việt cách tấm pin mặt trời tạo ra điện và cách tiết kiệm năng lượng trong gia đình.',
        'Viết hướng dẫn bằng tiếng Việt để chuẩn bị bữa ăn lành mạnh từ rau theo mùa, theo từng bước.',
        'Giải thích bằng tiếng Việt giấc ngủ ảnh hưởng đến trí nhớ và sự tập trung như thế nào, kèm ví dụ.',
    ),
    'hu': (
        'Magyarázd el magyarul, hogyan termelnek áramot a napelemek, és hogyan takarékoskodhatunk az energiával otthon.',
        'Írj magyar nyelvű, lépésenkénti útmutatót egy egészséges étel elkészítéséhez szezonális zöldségekből.',
        'Magyarázd el magyarul, hogyan hat az alvás a memóriára és a koncentrációra, mindennapi példákkal.',
    ),
}
LANGUAGES = frozenset(_PROMPT_GROUPS).difference({'code'}) | frozenset(_ADDITIONAL_PROMPTS)


def language_preferences(value: str) -> frozenset[str]:
    """Parse comma-separated language tags; custom corpora may define their own."""
    entries = value.lower().split(',')
    if any(not entry.strip() for entry in entries):
        raise ValueError('languages must be a comma-separated nonempty list')
    selected = frozenset(entry.strip() for entry in entries)
    if any(not re.fullmatch(r'[a-z]{2,3}(?:-[a-z0-9]{2,8})*', entry) for entry in selected):
        raise ValueError('languages must use language tags such as zh,en')
    return selected


def calibration_prompts(languages: frozenset[str] | None = None) -> tuple[str, ...]:
    if languages is None:
        return PROMPTS
    if not languages:
        raise ValueError('at least one language profile is required')
    if languages - LANGUAGES:
        raise ValueError('unsupported built-in language profiles: ' + ', '.join(sorted(languages - LANGUAGES)))
    common = tuple(prompt for group, prompt in zip(_PROMPT_GROUPS, PROMPTS, strict=True)
                   if group == 'code' or group in languages)
    return common + tuple(prompt for group in sorted(languages)
                          for prompt in _ADDITIONAL_PROMPTS.get(group, ()))

def selected_ids(counts: Counter[int], vocabulary: int, rows: int,
                 required: frozenset[int] = frozenset()) -> list[int]:
    if not counts or not 0 < rows <= vocabulary:
        raise ValueError("nonempty calibration and valid row count required")
    if any(type(token) is not int or not 0 <= token < vocabulary or type(count) is not int or count <= 0
           for token, count in counts.items()):
        raise ValueError("invalid calibration counts")
    if len(required) > rows or any(type(token) is not int or not 0 <= token < vocabulary for token in required):
        raise ValueError("required token IDs must fit the vocabulary and selected rows")
    selected = set(required)
    for token in sorted(range(vocabulary), key=lambda token: (-counts[token], token)):
        if len(selected) == rows:
            break
        selected.add(token)
    return sorted(selected)


def tokenizer_default_ids(tokenizer, vocabulary: int, rows: int | None = None,
                          languages: frozenset[str] | None = None,
                          template: str | None = None) -> tuple[list[int], str]:
    """Select bounded common IDs from local tokenizer and preparation prompts."""
    if type(vocabulary) is not int or vocabulary <= 0:
        raise ValueError("positive source vocabulary required")
    special = set(tokenizer.all_special_ids)
    if any(type(token) is not int or not 0 <= token < vocabulary for token in special):
        raise ValueError("tokenizer special token is outside the source vocabulary")
    counts: Counter[int] = Counter()
    encoded_prompts = []
    for prompt in calibration_prompts(languages):
        text = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False, chat_template=template)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids or any(type(token) is not int or not 0 <= token < vocabulary for token in ids):
            raise ValueError("tokenizer prompt contains IDs outside the source vocabulary")
        encoded_prompts.append(ids)
        counts.update(ids)
    if rows is None:
        rows = max(len(special), (vocabulary + 1) // 2)
    if type(rows) is not int or not 0 < rows <= vocabulary:
        raise ValueError("draft row count must fit the source vocabulary")
    selected = set(sorted(special, key=lambda value: (-counts[value], value))[:rows])
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
