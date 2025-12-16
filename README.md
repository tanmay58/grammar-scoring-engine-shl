# grammar-scoring-engine-shl

# Grammar Scoring Engine for Spoken English


## Approach
- Audio transcription using Whisper (ASR)
- Transcript cleaning and normalization
- Fine-tuning DistilBERT for grammar score regression
- Evaluation using RMSE and Pearson Correlation

## Files
- `final_notebook.ipynb` – Complete pipeline and results
- `submission.csv` – Final predictions

## Requirements
- Python 3.10+
- transformers
- datasets
- torch
- whisper

## How to Run
Open the notebook and run cells sequentially.
