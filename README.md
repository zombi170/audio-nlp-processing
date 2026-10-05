# Sequential Data: Audio & NLP Processing

This repository showcases data engineering and machine learning techniques applied to sequential human-generated data, specifically focusing on acoustic feature extraction and natural language tokenization.

## Core Projects

### 1. Speech Emotion Feature Extraction (RAVDESS)
*   **Data Ingestion:** Automated the downloading, extraction, and batch-processing of the RAVDESS `Audio_Speech_Actors` dataset.
*   **Acoustic Processing:** Utilized the `opensmile` toolkit to extract the `eGeMAPSv02` feature set from raw audio files.
*   **Feature Engineering:** Parsed over 1,440 audio sequences into a unified Pandas DataFrame containing 88 distinct low-level descriptors and functionals, including `F0semitoneFrom27.5Hz_sma3nz_amean` and `loudness_sma3_amean`.

### 2. NLP Sentiment Analysis Pipeline (GloVe)
*   **Tokenization:** Built a custom `Vocabulary` class leveraging `spacy` to process and tokenize raw text data.
*   **Sequence Mapping:** Implemented standard sequence padding and unknown word handling using `<pad>`, `<sos>`, `<eos>`, and `<unk>` tokens to ensure uniform tensor dimensions.
*   **Modeling:** Configured pre-trained GloVe word vectors (100-dimensional) to feed into a custom Long Short-Term Memory (LSTM) network for sentiment classification.
