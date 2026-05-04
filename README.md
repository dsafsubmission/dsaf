# DSAF: Domain-Specific Alignment Failure

Code and audit materials for the DSAF corpus — 209,072 matched-pair decisions from 8 frontier LLMs across 3 deployment-realistic testbeds.

## Structure

- `classifiers/` — Domain-specific violation classifiers (healthcare, lending, trading)
- `corpus_generation/` — Evaluation framework and LLM client
- `analysis/` — Corpus compilation, deduplication, and release CSV generation
- `audit/` — Compliance fabrication audit (400 dual-coded items, κ=0.94), evaluation cards, manipulation library, prompt templates, regulatory grounding
- `mechanistic/` — SAE interpretability scripts (Gemma 3 27B)

## Dataset

Full corpus on [HuggingFace](https://huggingface.co/datasets/dsafsubmission/DSAFSUBMISSION).

## License

Code: MIT | Corpus: CC BY 4.0
