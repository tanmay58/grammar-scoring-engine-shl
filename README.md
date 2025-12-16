🎙️ Grammar Scoring Engine for Spoken English

SHL Intern Hiring Assessment – 2025

This repository contains a complete solution for building a Grammar Scoring Engine that evaluates spoken English responses and predicts a grammar proficiency score (1–5) using audio data.

The system converts speech to text using ASR, cleans the transcripts, and fine-tunes a DistilBERT regression model to predict grammar scores in a human-like manner.

📌 Problem Overview

Input: 45–60 second English speech audio files (.wav)

Output: Grammar score from 1 to 5 (MOS Likert scale)

Training samples: 409

Test samples: 197

Evaluation Metrics:

RMSE

Pearson Correlation

🧠 Approach Summary

The solution follows a speech → text → NLP regression pipeline:

Automatic Speech Recognition (ASR) using Whisper

Transcript Cleaning to remove fillers and noise

Text-based Grammar Modeling using DistilBERT

Regression Training to predict continuous grammar scores

Evaluation & Visualization

Final Submission Generation

This approach leverages strong language modeling capabilities of transformers while keeping the pipeline simple and interpretable.

🛠️ Pipeline Architecture
🎧 1. Audio Preprocessing & Transcription

All .wav files are transcribed using OpenAI Whisper (base)

Safe settings (fp16=False) used for Kaggle compatibility

✍️ 2. Transcript Cleaning

Removed filler words like “um”, “uh”, “you know”

Normalized casing and spacing

Clean text improves grammar signal for the model

🧩 3. Dataset Preparation

Cleaned transcripts paired with grammar labels

Split into train / validation sets

Converted to HuggingFace Dataset format

🤖 4. Model – DistilBERT (Regression)

Pretrained model: distilbert-base-uncased

Fine-tuned for regression (num_labels=1)

Predicts continuous grammar score

📊 5. Training & Evaluation

Loss: MSE

Metrics:

MAE

RMSE

Pearson Correlation

📈 6. Visualization

Predicted vs True score scatter plot

Grammar score distribution histogram

📁 7. Submission

Predictions clipped to range [1, 5]

Rounded to nearest integer

Saved as submission.csv

📊 Final Validation Performance
Metric	Value
MAE	~0.55
RMSE	~0.76
Pearson Correlation	~0.54

These results show good ranking ability and stable performance given the small dataset size.
