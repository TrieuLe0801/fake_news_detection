# Fake News Detection (Vietnamese)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A small toolkit and research codebase for detecting fake health news in Vietnamese. This repository provides dataset processing, NLP utilities, crawling DAGs (Airflow), and scripts to train / evaluate classification models for fake news detection.

## Table of Contents
- [What it does](#what-it-does)
- [Why it is useful](#why-it-is-useful)
- [Key features](#key-features)
- [Quick start](#quick-start)
- [Datasets & data layout](#datasets--data-layout)
- [Airflow pipelines](#airflow-pipelines)
- [Running tests](#running-tests)
- [Where to get help](#where-to-get-help)
- [Maintainers & contributing](#maintainers--contributing)
- [License](#license)

## What it does

This project collects, normalizes, and processes Vietnamese health news articles and provides tools to analyze and build fake-news classifiers. It contains:

- Crawlers / DAGs for source websites (Airflow) located in `airflow/dags/`.
- Data processing and normalization utilities in `src/processor/` and `vncorenlp/` support files.
- Data models and controllers for ingesting and storing news in `src/data_models/` and `src/controllers/`.
- Example scripts and a top-level `app.py` for demonstration and quick runs.

## Why it is useful

- Focused on Vietnamese health news — includes language-specific normalization and tokenization helpers.
- Provides end-to-end components: crawling (Airflow), normalization, storage, and analysis/visualization.
- Ready-to-use datasets and training-ready CSV files to bootstrap experiments.

## Key features

- Pre-built crawlers for multiple Vietnamese news sources (`vnxpress`, `daikynguyen`, `suckhoetot`, `covid-19`).
- Dataset normalization using VNCoreNLP resources (included under `vncorenlp/`).
- Data visualizations and EDA views in `src/views/`.
- Example tests under `tests/` to validate crawling, processing, and database insertion.

## Quick start

### Prerequisites

- Python 3.11+ (use a virtual environment)
- Java 1.8+ (required for VNCoreNLP models if using `vncorenlp` directly)
- Miniconda or virtualenv (required for virtual environment)

### Install dependencies

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r --no-cache-dir requirements.txt
```

### Prepare datasets

- Project includes some processed CSVs under `data/` and `vncorenlp/` such as `normalized_dataset_pyvi.csv` and `full_dataset.csv`.
- If you plan to run crawlers, configure Airflow (see `airflow/`) and provide any required secrets or DB settings.

### Run the demo app

```powershell
streamlit run app.py --server.port 8000
```

Note: `app.py` is a lightweight entry point used for demonstration. For production ingestion use the Airflow DAGs in `airflow/dags/`.

### Using the processing utilities

Basic processing can be invoked via scripts in `src/processor/`. For example (from project root):

```powershell
python -m src.processor.data_processor
```

## Datasets & data layout

- `data/` — main CSVs and archival datasets. Contains `full_dataset.csv`, `normalized_dataset_pyvi.csv`, and site-specific CSVs.
- `vncorenlp/` — VNCoreNLP models and normalized CSVs created with VNCoreNLP.
- `normalized_dataset_pyvi.csv` — normalized dataset produced by pyvi/tokenizers used in experiments.

## Airflow pipelines

Airflow configuration and Docker setup are available under `airflow/`:

- `airflow/docker-compose.yaml` — compose setup for local Airflow (if present).
- `airflow/dags/` — contains DAGs: `crawl_vnxpress.py`, `crawl_daikynguyen.py`, `crawl_suckhoetot.py`.

To run Airflow locally, follow the instructions in `airflow/airflow.Dockerfile` and `airflow/docker-compose.yaml`.

## Running tests

Run the test suite with `pytest`:

```powershell
pytest -q
```

## Where to get help

- Open an issue on this repository for bugs or feature requests.
- For usage questions, include reproduction steps and which dataset/file you used.

## Maintainers & contributing

Maintainer: Trieu Le (repository owner)

Contributions are welcome — please open an issue or a pull request. For larger changes, create an issue first to discuss the approach. See `LICENSE` for licensing information.

If you want a dedicated `CONTRIBUTING.md`, I can add a starter file — tell me your preferred contribution rules and code style.

## License

This project is licensed under the terms in the `LICENSE` file.

---

If you'd like, I can also:

- Add a short `CONTRIBUTING.md` and a `CODE_OF_CONDUCT.md`.
- Add CI badges (GitHub Actions) and run a test workflow.
- Run the test suite now and report any failures.

Tell me which next step you'd like me to take.