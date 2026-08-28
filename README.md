# rag

A minimal RAG (retrieval-augmented generation) over electronic component datasheets, in a single Python file.
No framework, no vector database: PDFs → chunks → embeddings → numpy index → top-k → LLM.

## Requirements

- [Git](https://git-scm.com/downloads)
- [Python](https://www.python.org/downloads/) ≥ 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

## Install

```bash
git clone https://github.com/jlurton/rag.git && cd rag
poetry install                       
poetry run python download_data.py    # downloads the datasheets into data/
cp .env.example .env                  # then set HF_TOKEN in .env
poetry run python check_setup.py      # checks Python, token, corpus, embedding model, LLM call
```