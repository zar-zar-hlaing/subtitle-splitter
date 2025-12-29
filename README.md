# Subtitle Splitter with LLM (AWS Bedrock)

A **multi-language, character-limit-safe subtitle splitting tool** using AWS Bedrock LLMs, **specialized for Thai subtitle content**, with strict validation and zero-hallucination guarantees for professional subtitle pipelines.

It uses AWS Bedrock LLMs to propose linguistically natural split points, then applies strict post-validation to guarantee zero hallucinations, preserve original text, and enforce broadcast-grade character limits — with specialized robust handling for Thai subtitles.

---

## Features

### Multi-Language Support
- Thai (TH) — **primary language with strict validation**
- English (EN)
- Other languages supported via AWS Bedrock LLMs (configurable)
- Language-specific splitting logic

### Advanced Thai Subtitle Handling
Thai subtitles require special treatment due to:
- No spaces between words
- Clause-based sentence structure
- Strict broadcast readability rules

This tool includes:
- Robust Thai post-validation
- Token-based reconstruction using original text
- Conjunction-aware line breaking (แต่, และ, ว่า, etc.)
- Hard enforcement of max character length

### Zero-Hallucination Guarantee
- Output text is rebuilt **only from the original subtitle**
- LLM output is treated as intent, not truth
- Any added, dropped, or duplicated text is automatically corrected

### Strict Character Limits
- Absolute maximum characters per line
- Character limit always takes priority over grammar

---

## What This Tool Does

This tool splits long subtitle sentences into multiple subtitle lines while:

- Respecting maximum character limits per line
- Preserving the **original text exactly** (no rewriting, no paraphrasing)
- Remaining language-aware across multiple languages
- Enforcing **strict Thai linguistic validation**
- Preventing hallucinations and content drift
- Supporting professional broadcast and streaming workflows

This is a **structural transformation tool**, not a creative one.

---

## Where Should This Tool Be Used?

- Broadcast and OTT subtitle pipelines
- Enterprise subtitle processing workflows
- NLP-driven subtitle preprocessing
- Thai subtitle pipelines requiring strict editorial control

---

## Who Should Use This Tool?

This tool is designed for:

- Subtitle engineers and localization teams
- Broadcast and streaming platforms
- NLP engineers working with subtitle text
- Thai subtitle workflows requiring strict validation

---


## Architecture Overview

```
Input Subtitle
   ↓
Pre-Cleaning (tags, formatting)
   ↓
LLM Split Proposal (AWS Bedrock)
   ↓
Validation & Repair
   ↓
Language-Specific Validation
   ├── Thai → Robust Token Reconstruction
   └── Others → Segment Optimization
   ↓
Final Subtitle Lines (| separated)
```

---

## Requirements

### Python
- Python 3.8+

### Libraries
```bash
pip install boto3
```

---

## AWS Bedrock Setup

- AWS account with Bedrock access
- Enabled model:
  us.meta.llama3-3-70b-instruct-v1:0

Configure credentials in the script:
```python
access_key = "YOUR_AWS_ACCESS_KEY"
secret_key = "YOUR_AWS_SECRET_KEY"
```

---

## Usage (CLI)

```bash
python3 subtitle_split.py \
  --subtitle_language th \
  --subtitle_sentence "เธอบอกว่าแค่เพราะฉันแจ้งความว่าเขาหายไปไม่ได้แปลว่าฉันอยากได้เขาคืนใครไปบาร์" \
  --tokenize_subtitle_text "เธอ บอก ว่า แค่ เพราะ ฉัน แจ้งความ ว่า เขา หาย ไป ไม่ ได้ แปล ว่า ฉัน อยาก ได้ เขา คืน ใคร ไป บาร์" \
  --max_char_len 37
```

### Output
```
เธอบอกว่าแค่เพราะฉันแจ้งความว่าเขาหายไป|
ไม่ได้แปลว่าฉันอยากได้เขาคืนใครไปบาร์
```

---

> ⚠️ **Thai tokenization is required**
>
> When using this tool for Thai subtitles, you must tokenize the subtitle sentence **before** calling `split_subtitle`.
> This requires installing **PyThaiNLP**.

### Install Thai tokenization dependency

```bash
pip install pythainlp
```

## Usage (Programmatic / Python Import)

```python
from pythainlp.tokenize import word_tokenize
from subtitle_split import split_subtitle

# Original subtitle sentence
subtitle_sentence = "เธอบอกว่าแค่เพราะฉันแจ้งความว่าเขาหายไปไม่ได้แปลว่าฉันอยากได้เขาคืนใครไปบาร์"

# Thai tokenization
tokens = word_tokenize(subtitle_sentence, engine="newmm")
tokenized_text = " ".join(tokens)

# Call the subtitle splitter
result = split_subtitle(
    subtitle_language="th",
    subtitle_sentence=subtitle_sentence,
    tokenize_subtitle_text=tokenized_text,
    max_char_len=37,
)

print(result)
```


### Output
```
เธอบอกว่าแค่เพราะฉันแจ้งความว่าเขาหายไป|
ไม่ได้แปลว่าฉันอยากได้เขาคืนใครไปบาร์
```

---

## Arguments

| Argument | Description |
|--------|------------|
| --subtitle_language | ISO-639-1 language code |
| --subtitle_sentence | Subtitle text |
| --tokenize_subtitle_text | Space-tokenized text (Thai required) |
| --max_char_len | Max characters per line |

---

## Thai-Specific Logic

- LLM output is treated as *intent*
- Text is rebuilt strictly from original tokens
- Enforces start-with conjunction rules
- Prevents broken Thai words

---

## 📁 ZIP Release Structure

```
subtitle-splitter/
├── subtitle_split.py
├── README.md
└── LICENSE
```

---

## 🛡 License

Released under the **MIT [LICENSE](LICENSE)**.

---

## 👤 Author

Zar Zar Hlaing
