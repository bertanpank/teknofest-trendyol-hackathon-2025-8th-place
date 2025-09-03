# Trendyol ML (Minimal)

Bu depo **sadece notebooklar (src/ içinde)** ve **pip gereksinimleri** içerir.

## Kurulum (pip + venv)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Yapı
```
trendyol-ml-min/
  src/
        01_preprocessing.ipynb
    02_model_validation_and_weight_optimization.ipynb
    03_final_model.ipynb
  requirements.txt
  .gitignore
  README.md
```
