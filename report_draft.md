# COMP6252 Coursework 1 Report Draft

**Name:** `[Your Name]`  
**ECS User ID:** `[Your ECS ID]`

## Part I: Music Genre Classification

### 1. Introduction

This coursework investigates supervised deep learning methods for music genre classification on the GTZAN dataset. The dataset contains ten genres, including blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. The coursework requires the implementation and comparison of six neural architectures using both spectrogram images and raw audio-derived features. The overall goal is not only to maximise classification accuracy, but also to understand how network architecture, normalisation, optimisation, sequential modelling, and data augmentation influence performance.

Two different input modalities are considered. For `Net1` to `Net4`, the models are trained on the visual MEL spectrogram images provided with the dataset. For `Net5` and `Net6`, the models are trained on audio signals transformed into log-Mel time-frequency representations. This allows a comparison between image-based convolutional learning and audio-sequence modelling with recurrent networks.

### 2. Dataset and Preprocessing

The GTZAN dataset was downloaded from Kaggle and organised into two main modalities: `images_original` for the spectrogram images and `genres_original` for the corresponding audio files. The dataset was randomly split into training, validation, and test subsets in the ratio required by the brief: `70% / 20% / 10%`.

For the image-based experiments (`Net1` to `Net4`), all images were resized to `180 x 180` at load time, following the coursework specification. Each image was then converted into a tensor for input into the neural network. This ensured a fixed input shape across all image models and made the training process stable and reproducible.

For the audio-based experiments (`Net5` and `Net6`), the audio-derived `features_3_sec.csv` file provided with the GTZAN dataset was used. Each 30-second song was represented as a sequence of ten 3-second feature vectors. This keeps temporal structure while providing more stable and compact audio features than feeding the full raw waveform directly into the LSTM.

To improve reproducibility, a fixed random seed was used for data splitting and model initialisation. Model selection was performed using the validation set, while final performance was reported on the held-out test set.

### 3. Model Architectures

#### 3.1 Net1: Fully Connected Network

`Net1` is a fully connected neural network with two hidden layers. The resized `180 x 180` RGB image is flattened into a one-dimensional vector and passed through two dense hidden layers with ReLU activations. Dropout is applied between hidden layers to reduce overfitting. This model serves as a baseline and tests how far classification can be achieved without explicit convolutional feature extraction.

#### 3.2 Net2: Convolutional Neural Network

`Net2` implements the convolutional architecture specified in Figure 1 of the coursework brief. It consists of four convolutional layers grouped into two convolutional blocks, with a max-pooling layer after every two convolutions. The extracted feature maps are then flattened and passed to a fully connected layer before the final classifier output. Compared with `Net1`, this architecture is expected to perform better because convolutional filters can capture local spectral patterns and hierarchical structure in the input images.

#### 3.3 Net3: CNN with Batch Normalisation

`Net3` extends `Net2` by adding batch normalisation after each convolutional layer. Batch normalisation helps stabilise activation distributions during training, improves gradient flow, and often enables faster and more reliable convergence. This model was included to study whether improved training stability translates into better generalisation on the genre classification task.

#### 3.4 Net4: CNN with Batch Normalisation and RMSProp

`Net4` uses the same architecture as `Net3` but replaces the optimiser with RMSProp. The purpose of this experiment is to isolate the effect of optimisation strategy while keeping the network structure unchanged. RMSProp adaptively scales learning rates for each parameter and is often effective for noisy, non-stationary training dynamics.

#### 3.5 Net5: LSTM-Based Audio Classifier

`Net5` is an LSTM network trained on audio-derived 3-second feature sequences rather than spectrogram images. Each song is represented as a sequence of ten feature vectors, allowing the model to learn how genre-related audio characteristics change over time. A two-layer bidirectional LSTM is used so that the model can learn both forward and backward temporal dependencies. Layer normalisation and dropout are applied to improve optimisation and reduce overfitting. The final forward and backward hidden states from the top recurrent layer are concatenated and passed to fully connected layers to predict the music genre.

#### 3.6 Net6: LSTM with GAN-Based Data Augmentation

`Net6` uses the same recurrent classification backbone as `Net5`, but augments the training data using a conditional GAN. The GAN is trained on flattened audio feature sequences conditioned on genre labels. The generated feature sequences are then reshaped back into the same sequence format used by `Net5` and combined with the real training data to increase diversity and reduce overfitting.

### 4. Training Setup

All models were implemented in PyTorch. For `Net1` to `Net4`, the coursework requirement to train for both `50` and `100` epochs was followed. `Net1`, `Net2`, and `Net3` used Adam optimisation, while `Net4` used RMSProp as required by the specification. Cross-entropy loss was used for all classification models because the task is a ten-class classification problem.

For `Net5` and `Net6`, training was run until a chosen stopping point based on validation behaviour. Adam was used as the optimiser. The validation set was used to select the best-performing model checkpoint. Final reported results were obtained on the test set only after model selection had been completed.

The training configuration is summarised below.

| Model | Input | Main Architecture | Optimiser | Epochs |
|---|---|---|---|---:|
| Net1 | Spectrogram image | Fully connected, 2 hidden layers | Adam | 50, 100 |
| Net2 | Spectrogram image | CNN | Adam | 50, 100 |
| Net3 | Spectrogram image | CNN + BatchNorm | Adam | 50, 100 |
| Net4 | Spectrogram image | CNN + BatchNorm | RMSProp | 50, 100 |
| Net5 | Audio 3-second feature sequence | BiLSTM | Adam | `[fill in]` |
| Net6 | GAN-augmented audio feature sequence | BiLSTM + conditional GAN | Adam | `[fill in]` |

### 5. Experimental Results

The final results should be inserted into Table 2 after running all experiments.

| Model | Validation Accuracy | Test Accuracy | Validation Loss | Test Loss |
|---|---:|---:|---:|---:|
| Net1 (50 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net1 (100 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net2 (50 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net2 (100 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net3 (50 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net3 (100 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net4 (50 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net4 (100 epochs) | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net5 | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |
| Net6 | `[fill in]` | `[fill in]` | `[fill in]` | `[fill in]` |

In addition to the aggregate results table, confusion matrices and per-class precision/recall statistics should be included for the best-performing image model and the best-performing audio model. These provide a more detailed view of which genres are easy to distinguish and which are commonly confused.

### 6. Discussion

The fully connected baseline (`Net1`) is expected to perform worse than the convolutional models because flattening the input discards spatial locality. In spectrogram-like images, neighbouring regions often carry meaningful local patterns that correspond to harmonics, rhythmic textures, or transient energy. CNNs are better suited to exploit this structure because they learn local filters and hierarchical feature representations.

`Net2` should therefore improve upon `Net1` by learning convolutional features directly from the spectrogram images. If this trend is observed in the results, it would confirm that local structure in time-frequency representations is important for music genre classification.

`Net3` evaluates the contribution of batch normalisation. If it achieves higher validation accuracy, lower loss, or more stable learning curves than `Net2`, this would suggest that batch normalisation improved optimisation and reduced internal covariate shift. Even when the final test accuracy gain is modest, improved stability can still be valuable because it makes training more reliable across epochs.

`Net4` isolates the effect of changing only the optimiser. If RMSProp outperforms Adam in this setting, it would indicate that the convolutional architecture benefits from adaptive per-parameter learning-rate scaling. If it underperforms or overfits more strongly, that would suggest that optimisation alone cannot compensate for architectural or dataset limitations.

The comparison between `Net5` and `Net6` is particularly important because it tests whether synthetic augmentation improves performance in a small-data regime. If `Net6` outperforms `Net5`, especially on minority or confusable genres, then the GAN-generated training samples likely increased useful data diversity. If the gain is limited or negative, the generated samples may not be sufficiently realistic or label-consistent. This is a meaningful finding in itself, because data augmentation quality is often as important as quantity.

Another important point is the comparison between image-based and audio-based models. A stronger CNN result would suggest that the provided spectrogram images already encode enough discriminative information for genre classification. A stronger LSTM result would suggest that modelling temporal evolution directly from audio-derived sequences is more effective. If their performance is similar, it may indicate that both visual time-frequency structure and temporal sequence modelling capture complementary aspects of genre.

The main limitations of this work are the relatively small dataset size, possible label ambiguity between similar genres, and the simplified GAN training process. Additional improvements could include deeper CNNs, residual connections, attention-based sequence models, stronger audio augmentation, or transfer learning from pretrained audio encoders.

### 7. Conclusion

This coursework demonstrated how neural architecture design, optimisation, and data augmentation affect music genre classification performance. The experiments span a progression from a simple fully connected baseline to CNNs, recurrent sequence models, and GAN-augmented training. The results should show that architectures that better respect the structure of the input modality generally perform better. Convolutional models are expected to outperform the dense baseline on spectrogram images, while recurrent models provide a principled way to learn temporal dependencies from audio features. Batch normalisation and optimiser choice influence training stability, and GAN augmentation offers a promising but non-trivial route for improving performance when labelled data is limited.

## Part II: Reflection on Deep Learning Technologies

### Topic: Representation Learning and Optimisation in Deep Learning

Representation learning is one of the most important ideas in deep learning because it allows a model to automatically learn useful features from raw or weakly processed data, rather than relying entirely on hand-crafted features. This is especially important in problems such as image recognition, audio understanding, and natural language processing, where manually designing strong features is difficult and often domain-specific. In this coursework, the difference between a flattened-input fully connected network and a convolutional or recurrent architecture clearly reflects the importance of learning representations that match the structure of the data.

Several current deep learning technologies are relevant to this topic. Convolutional neural networks are effective when the data contains local spatial structure, such as images or spectrograms. Recurrent networks such as LSTMs are useful when temporal dependencies matter, such as in speech, music, or sensor data. More recently, transformer architectures and attention mechanisms have become dominant in many sequence-modelling tasks because they can capture long-range dependencies more effectively than standard recurrent models. Generative models such as GANs and diffusion models are also closely related because they learn latent representations that can be used for synthesis, augmentation, and unsupervised learning.

At the current stage, I am confident implementing foundational methods such as feedforward networks, CNNs, basic optimisation loops in PyTorch, and recurrent models such as LSTMs. I also understand the core ideas of batch normalisation, dropout, and common optimisers such as SGD, Adam, and RMSProp. However, more advanced technologies remain harder to implement robustly from scratch. In particular, GAN training is difficult because of instability, mode collapse, and the need to carefully balance generator and discriminator learning. Transformer-based models are also more complex because they require a deeper understanding of attention mechanisms, positional encoding, and efficient large-scale training. The main difficulty is not only writing the code, but also ensuring that the implementation is numerically stable, computationally efficient, and empirically well tuned.

The impact of these deep learning technologies is both positive and negative. On the positive side, they have enabled major progress in healthcare, speech recognition, computer vision, recommendation systems, and scientific discovery. They reduce the need for handcrafted features and often produce state-of-the-art performance when sufficient data and computation are available. On the negative side, they can be computationally expensive, environmentally costly, difficult to interpret, and highly dependent on data quality. Poorly designed or biased datasets may lead to unfair or misleading results. In generative settings, these technologies can also be misused to produce synthetic media that is deceptive or harmful.

My view of the future is that deep learning technologies will continue to become more capable, but success will increasingly depend on efficiency, trustworthiness, and domain adaptation rather than raw scale alone. Models that combine strong representation learning with better interpretability, lower data requirements, and more reliable training will be particularly valuable. For students and practitioners, this means that understanding the principles behind architectures, optimisation, and data representation will remain essential even as higher-level libraries become easier to use.
