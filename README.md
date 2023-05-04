# Sign Language Transformers

![](https://raw.githubusercontent.com/AugustinCombes/sign_language_transformers/main/gifs_illustration/signer.gif)

This project explores various strategies for sign language classification using time series data from mediapipe landmarks of signers' hands, body, and facial expressions in the ISLR Kaggle Challenge. The goal is to classify 250 possible sign labels.

## Baseline Model

- Built using a GRU classifier, focusing on hand landmarks.
- Achieved a local accuracy of 54% and a LB score of 51%.

## Improvements

### Data Handling & Model Ensemble

- Implemented a 7-fold split based on participant ID.
- Used the mean prediction of 7 sub-models for inference.
- Improved LB score to 56% with better CV-LB accuracy correlation.

### Hyperparameter Optimization

- Employed Bayesian optimization using KerasTuner for efficient hyperparameter search.

### Multi-modal Transformers

- Explored treating different parts of the body landmarks as separate modalities.
- Implemented a ViViT-like model with unimodal blocks and multi-layer self-attentive encoders.
- Achieved a local CV score of 76% and a LB score of 70%.

## Cross-Attentive Flow Control

- Investigated techniques to control cross-attentive flow between modalities.
- Inconclusive results but provided valuable insights for future work.

## Key Takeaways

- Importance of techniques such as model ensemble, K-Fold, and data augmentation in improving predictive performance.
- Time management and efficient training approaches are crucial for success in Kaggle challenges.

![](https://raw.githubusercontent.com/AugustinCombes/sign_language_transformers/main/gifs_illustration/multimodal.gif)
