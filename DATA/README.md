# DATA

Put the documents to index here. They are not included in this repository.

```
DATA/
├── true_data/    # technical documentation the assistant should answer from
└── noisy_data/   # unrelated or distractor documents (used to test noise filtering)
```

Every chunk is tagged `dataset: true | noisy` from these folder names. Supported formats:
`.pdf .html .htm .txt .md .docx .pptx`. Then run:

```bash
python -m app.ingestion.pipeline              # DATA/ -> processed_data/
python -m app.services.retrieval.indexer      # processed_data/ -> Qdrant
```

The evaluation dataset (`evals/dataset.jsonl`) references `true_data/` files by name, so if you use
different documents, update the questions, reference answers and `expected_sources` there too.
