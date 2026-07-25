# Data availability

The full v4 JSONL splits are intentionally not distributed in this GitHub
repository. The canonical public Hugging Face repository is a rights-minimized
index; the full raw archive remains private.

This public release includes only the frozen validation/test article-ID lists
and split metadata needed to identify the evaluated cohort:

- `splits/v4/split_meta_v4.json`
- `splits/v4/val_ids_frozen_v4.json`
- `splits/v4/test_ids_frozen_v4.json`

These identifiers point to published source records; they do not include
article text, review text, author responses, or LLM-normalized concern text.
Do not infer public redistribution rights for those upstream fields from the
presence of an identifier here.

Authorized local copies should place the complete files at
`data/splits/v4/{train,val,test}.jsonl`. The evaluation and validation-pack
commands use those paths by default.
