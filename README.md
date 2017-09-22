# kaggle-DigitRecognizer

This repo workes on the Kaggle competition [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer/).

## Methods
The following are several methods used in this competition.

### Deep learning
- SimpleCNN: a self-designed simple CNN model, with 8 layers
- miniResNet: a smaller version of ResNet, with 22 layers
- ResNet50_freeze: pretrained ResNet50, with all weights freezed
- ResNet50_finetune: pretrained ResNet50
(all data are centered for deep learning)

#### To-do list
- data-augmented inference
- autoencoder pretraining

### Traditional methods

It can be divided into three modules: preprocessing, feature extraction, classification

#### preprocessing
- moment-based affine transformation

#### feature extraction
- SIFT, SURF, ORB
- HOG

#### classification
- regression tree
- k-nearest neighbor, kernel regression
- random forest
- support vector machine