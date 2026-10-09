# Audio and Text: Speech Emotion Recognition and Sentiment Analysis

Two notebooks on sequential, human-generated data: emotion recognition from acted speech, evaluated on speakers the model has never heard, and sentiment classification of movie reviews.

| Notebook | Task | Data | Model |
|---|---|---|---|
| [`opensmile.ipynb`](opensmile.ipynb) | 8-class speech emotion recognition | RAVDESS speech, 1,440 clips, 24 actors | openSMILE eGeMAPS + Random Forest |
| [`glove_sentiment_analysis.ipynb`](glove_sentiment_analysis.ipynb) | Binary sentiment classification | IMDB, 50,000 reviews | Frozen GloVe embeddings + LSTM |

## 1. Speech emotion recognition (RAVDESS)

### Data

The speech part of [RAVDESS](https://zenodo.org/records/1188976): 24 professional actors (12 female, 12 male) each say two fixed sentences in 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) at two intensities, twice. Neutral has no strong intensity, so it has half as many clips. The notebook downloads the data itself.

### Pipeline

1. Extract the **eGeMAPSv02** feature set with [openSMILE](https://github.com/audeering/opensmile-python), both as 88 utterance-level functionals and as frame-level low-level descriptors.
2. Classify the functionals with a standardised **Random Forest**.
3. Compare three evaluation protocols with the same pipeline:

| Protocol | What it measures |
|---|---|
| Shuffled, stratified 5-fold CV | Emotion recognition for speakers seen in training |
| Leave-one-actor-out (24 folds) | Emotion recognition for unseen speakers |
| Leave-one-actor-out + per-speaker normalisation | Unseen speakers, after removing each speaker's baseline |

Shuffled folds split clips, not people. Each actor recorded every sentence–emotion–intensity combination twice, so almost every test clip has the same speaker in training, often the other take of the identical utterance. eGeMAPS features (pitch, loudness, formants) carry a lot of speaker identity, so this protocol partly rewards recognising the speaker.

Per-speaker normalisation z-scores each feature within each actor. It uses the held-out speaker's unlabelled clips, as a deployed system could do from a short enrolment recording.

### Results

| Protocol | Accuracy | Macro F1 |
|---|---|---|
| Shuffled 5-fold (known speakers) | 0.624 ± 0.034 | 0.614 ± 0.032 |
| Leave-one-actor-out (unseen speakers) | 0.465 ± 0.095 | 0.414 ± 0.091 |
| **Leave-one-actor-out + speaker normalisation** | **0.594 ± 0.132** | **0.567 ± 0.139** |

Chance level is 0.125.

- **Speaker identity inflates the shuffled score.** Accuracy drops by 16 points when the test speaker is unseen.
- **Speaker normalisation recovers most of that gap** (0.465 → 0.594). Much of the difficulty comes from speakers' different baseline pitch and loudness, not from the emotions themselves.
- **Performance depends strongly on the speaker.** Accuracy ranges from 0.30 (actor 10) to 0.88 (actor 8).
- **Neutral and Sad are the hardest emotions.** Recall under leave-one-actor-out with normalisation:

  | Emotion | Recall |
  |---|---|
  | Calm | 0.77 |
  | Surprise | 0.71 |
  | Angry | 0.64 |
  | Disgust | 0.64 |
  | Happy | 0.61 |
  | Fearful | 0.53 |
  | Neutral | 0.39 |
  | Sad | 0.36 |

  Sad and Neutral are most often predicted as Calm (28% and 27% of their clips), so the three low-arousal emotions blur together. Calm's high recall partly comes from absorbing the other two.

## 2. Sentiment analysis (IMDB)

### Pipeline

1. **Cleaning,** in an order that keeps each step effective: HTML tags first (e.g. `<br />`), then URLs, then contractions while the apostrophes still exist (`isn't` → `is not`), then bracketed text and tokens containing digits, and punctuation last.
2. **Split:** stratified 70/10/20 into train, validation and test (35,000 / 5,000 / 10,000 reviews).
3. **Tokenisation:** a 10,000-word vocabulary fitted on the training reviews only. Sequences are padded to the 95th-percentile training length (563 tokens).
4. **Embeddings:** frozen, pre-trained 100-d [GloVe](https://nlp.stanford.edu/projects/glove/) vectors, which cover the whole vocabulary.
5. **Model:** LSTM(128) → Dropout(0.5) → Dense(32, ReLU) → sigmoid, trained with RMSprop. Early stopping on the validation loss picked epoch 8.
6. **Evaluation:** the test set is used once, after model selection.

The notebook also starts with a short exploration of GloVe vectors (looking up and summing word embeddings).

### Results

| Metric | Negative | Positive |
|---|---|---|
| Precision | 0.897 | 0.894 |
| Recall | 0.894 | 0.897 |
| F1 | 0.895 | 0.896 |

**Test accuracy: 0.895** on 10,000 held-out reviews. Errors are balanced between the two classes.
