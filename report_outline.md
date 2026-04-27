# COMP6252 Coursework 1 Report Outline

## Page 1-3: Music Genre Classification

### 1. Introduction

- Briefly describe the GTZAN dataset.
- State that the task is 10-class music genre classification.
- Summarize the six models evaluated.

### 2. Implementation Details

- Dataset split: `70% train / 20% validation / 10% test`
- Image preprocessing: `Resize(180,180)`, tensor conversion, normalization if used
- Audio preprocessing: waveform loading, STFT, log-Mel representation
- Optimizers, learning rates, batch size, stopping rule
- Hardware used

### 3. Neural Architectures

#### Net1
- Fully connected network with two hidden layers

#### Net2
- CNN following Figure 1: `conv1 -> conv2 -> maxpool -> conv3 -> conv4 -> maxpool -> fc -> output`

#### Net3
- Net2 + batch normalization

#### Net4
- Net3 + RMSProp

#### Net5
- LSTM-based sequence classifier using audio features

#### Net6
- Net5 + GAN-based data augmentation

### 4. Results

Create a compact table with:

| Model | Input Type | Epochs | Optimizer | Validation Accuracy | Test Accuracy |
|---|---|---:|---|---:|---:|
| Net1 | Image | 50 | Adam | | |
| Net1 | Image | 100 | Adam | | |
| Net2 | Image | 50 | Adam | | |
| Net2 | Image | 100 | Adam | | |
| Net3 | Image | 50 | Adam | | |
| Net3 | Image | 100 | Adam | | |
| Net4 | Image | 50 | RMSProp | | |
| Net4 | Image | 100 | RMSProp | | |
| Net5 | Audio | | Adam | | |
| Net6 | Audio + GAN | | Adam | | |

Also include:

- At least one confusion matrix figure
- A short comment about overfitting/underfitting
- A short comment about the effect of batch normalization and RMSProp
- A short comment about whether GAN augmentation helped

### 5. Discussion

- Compare image-based versus audio-based models
- Explain which models trained more stably
- Explain whether deeper CNNs improved performance
- Explain the main limitations of the current implementation
- Suggest future improvements

## Page 4: Reflection

Suggested topic: the role of optimization and representation learning in deep learning for multimedia classification.

Structure:

### 1. Why this topic is important

### 2. Current deep learning technologies relevant to the topic

### 3. Which methods you can implement now and which remain difficult

### 4. Positive and negative impact of current deep learning technologies

### 5. Your view of the future
