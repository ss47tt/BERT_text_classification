# BERT_text_classification

How to use Bert model to train, validate, and test dataset?

For training and validating:

1. I classified training and validating text files in dataset/post path, with 2 classes, that is, guilty and innocent sub paths.

2. I used bert-base-uncased model with limit of 512 tokens for each text file.

3. The dataset is randomly split into training and validating sets of 8:2 respectively before training and validating.

4. I used learning rate of 1e-5, with scheduler and early stopping for 10 patience each.

5. Training loss, validating loss and validating accuracy are computed for each epoch until early stopping is triggered. Scheduler and early stopping are based on validating loss.

6. Final model is saved.

For testing:

1. I classified testing text files in dataset/pre path, with 2 classes, that is, guilty and innocent sub paths.

2. I used the saved training model for testing.

3. Confusion matrix and classification report are printed.
