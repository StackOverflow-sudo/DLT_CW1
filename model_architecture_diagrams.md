# Model Architecture Diagrams

These diagrams match the current implementation in `music_genre_source.py`.

## Net1: Fully Connected Network

```mermaid
flowchart LR
    A["Input spectrogram image<br/>3 x 180 x 180"] --> B["Flatten<br/>97200 features"]
    B --> C["Linear<br/>97200 -> 512"]
    C --> D["ReLU"]
    D --> E["Dropout<br/>p = 0.3"]
    E --> F["Linear<br/>512 -> 128"]
    F --> G["ReLU"]
    G --> H["Dropout<br/>p = 0.3"]
    H --> I["Linear<br/>128 -> 10"]
    I --> J["Genre logits"]
```

## Net2: CNN

```mermaid
flowchart LR
    A["Input spectrogram image<br/>3 x 180 x 180"] --> B["Conv2d<br/>3 -> 32<br/>3x3, padding 1"]
    B --> C["ReLU"]
    C --> D["Conv2d<br/>32 -> 64<br/>3x3, padding 1"]
    D --> E["ReLU"]
    E --> F["MaxPool2d<br/>2x2"]
    F --> G["Conv2d<br/>64 -> 128<br/>3x3, padding 1"]
    G --> H["ReLU"]
    H --> I["Conv2d<br/>128 -> 128<br/>3x3, padding 1"]
    I --> J["ReLU"]
    J --> K["MaxPool2d<br/>2x2"]
    K --> L["AdaptiveAvgPool2d<br/>4 x 4"]
    L --> M["Flatten<br/>128 x 4 x 4"]
    M --> N["Linear<br/>2048 -> 256"]
    N --> O["ReLU"]
    O --> P["Dropout<br/>p = 0.3"]
    P --> Q["Linear<br/>256 -> 10"]
    Q --> R["Genre logits"]
```

## Net3: CNN + Batch Normalisation

```mermaid
flowchart LR
    A["Input spectrogram image<br/>3 x 180 x 180"] --> B["Conv2d<br/>3 -> 32"]
    B --> C["BatchNorm2d<br/>32"]
    C --> D["ReLU"]
    D --> E["Conv2d<br/>32 -> 64"]
    E --> F["BatchNorm2d<br/>64"]
    F --> G["ReLU"]
    G --> H["MaxPool2d<br/>2x2"]
    H --> I["Conv2d<br/>64 -> 128"]
    I --> J["BatchNorm2d<br/>128"]
    J --> K["ReLU"]
    K --> L["Conv2d<br/>128 -> 128"]
    L --> M["BatchNorm2d<br/>128"]
    M --> N["ReLU"]
    N --> O["MaxPool2d<br/>2x2"]
    O --> P["AdaptiveAvgPool2d<br/>4 x 4"]
    P --> Q["Flatten"]
    Q --> R["Linear<br/>2048 -> 256"]
    R --> S["ReLU"]
    S --> T["Dropout<br/>p = 0.3"]
    T --> U["Linear<br/>256 -> 10"]
    U --> V["Genre logits"]
```

## Net4: CNN + Batch Normalisation + RMSProp

```mermaid
flowchart LR
    A["Input spectrogram image<br/>3 x 180 x 180"] --> B["Same architecture as Net3"]
    B --> C["Conv blocks with BatchNorm"]
    C --> D["AdaptiveAvgPool2d<br/>4 x 4"]
    D --> E["Flatten"]
    E --> F["Linear<br/>2048 -> 256"]
    F --> G["ReLU + Dropout"]
    G --> H["Linear<br/>256 -> 10"]
    H --> I["Genre logits"]
    J["Training difference"] --> K["RMSProp optimiser<br/>lr = 1e-4"]
    K -.-> B
```

## Net5: LSTM Audio Classifier

```mermaid
flowchart LR
    A["Input audio feature sequence<br/>10 x 57<br/>(or log-Mel sequence in wav mode)"] --> B["LayerNorm<br/>input_size"]
    B --> C["Bidirectional LSTM<br/>2 layers<br/>hidden_size = 64 or 96<br/>dropout = 0.3"]
    C --> D["Final forward hidden state"]
    C --> E["Final backward hidden state"]
    D --> F["Concatenate<br/>hidden_size x 2"]
    E --> F
    F --> G["Linear<br/>2H -> 128"]
    G --> H["ReLU"]
    H --> I["Dropout<br/>p = 0.4"]
    I --> J["Linear<br/>128 -> 10"]
    J --> K["Genre logits"]
```

## Net6: LSTM with Conditional GAN Augmentation

```mermaid
flowchart TB
    subgraph GAN["Conditional GAN training"]
        A["Real audio feature sequence<br/>10 x 57"] --> B["Flatten sequence"]
        Y["Genre label"] --> C["Label embedding<br/>32"]
        Z["Random noise<br/>64"] --> D["Generator input<br/>noise + label embedding"]
        C --> D
        D --> E["Generator<br/>Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh"]
        E --> F["Generated flattened sequence"]
        B --> G["Discriminator input<br/>real sequence + label embedding"]
        F --> H["Discriminator input<br/>fake sequence + label embedding"]
        C --> G
        C --> H
        G --> I["Discriminator<br/>Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear"]
        H --> I
        I --> J["Real / fake decision"]
    end

    subgraph Classifier["Net6 classifier"]
        K["Real + generated sequences"] --> L["Same classifier as Net5"]
        L --> M["LayerNorm"]
        M --> N["2-layer bidirectional LSTM"]
        N --> O["Concatenate final forward/backward states"]
        O --> P["Linear -> ReLU -> Dropout -> Linear"]
        P --> Q["Genre logits"]
    end

    F --> K
    A --> K
```

## Short Summary

| Model | Input | Main idea | Training difference |
|---|---|---|---|
| Net1 | Spectrogram image | Fully connected baseline | Adam |
| Net2 | Spectrogram image | CNN feature extraction | Adam |
| Net3 | Spectrogram image | CNN + BatchNorm | Adam |
| Net4 | Spectrogram image | Same as Net3 | RMSProp |
| Net5 | Audio feature sequence | BiLSTM classifier | Adam |
| Net6 | Audio feature sequence | Net5 + GAN augmentation | Adam + conditional GAN |
