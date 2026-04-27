# %% [markdown]
"""
# COMP6252 Coursework 1

This notebook wraps the reusable training code in `music_genre_coursework.py` and follows the coursework structure from the PDF.
"""

# %%
from pathlib import Path
import torch

from music_genre_coursework import (
    AudioFeatureSequenceDataset,
    IMAGE_EPOCH_OPTIONS,
    IMAGE_SIZE,
    Net1,
    Net2,
    Net3,
    Net4,
    Net5,
    Net6,
    ResizeToTensor,
    augment_with_gan,
    resolve_dataset_roots,
    save_result,
    seed_everything,
    split_dataset,
    train_classifier,
    ImageFolderDataset,
)

# %%
seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_root = Path('dataset/gtzan-dataset-music-genre-classification')
output_dir = Path('outputs')
image_root, audio_root = resolve_dataset_roots(dataset_root)
print(device)
print(image_root)
print(audio_root)

# %% [markdown]
"""
## Image Models (Net1-Net4)
"""

# %%
from torch.utils.data import DataLoader

transform = ResizeToTensor(IMAGE_SIZE)
image_dataset = ImageFolderDataset(image_root, transform=transform)
train_set, val_set, test_set = split_dataset(image_dataset, seed=42)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# %%
image_models = [
    ('Net1', Net1, 'adam'),
    ('Net2', Net2, 'adam'),
    ('Net3', Net3, 'adam'),
    ('Net4', Net4, 'rmsprop'),
]

for model_name, model_cls, optimizer_name in image_models:
    for epochs in IMAGE_EPOCH_OPTIONS:
        model = model_cls(num_classes=len(image_dataset.classes))
        result, extras = train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            optimizer_name=optimizer_name,
            learning_rate=1e-4,
            class_names=image_dataset.classes,
        )
        result.model_name = model_name
        save_result(output_dir, result, extras)

# %% [markdown]
"""
## Audio Models (Net5-Net6)
"""

# %%
audio_dataset = AudioFeatureSequenceDataset.from_dataset_root(dataset_root)
train_audio, val_audio, test_audio = split_dataset(audio_dataset, seed=42)
audio_dataset.fit_standardizer(train_audio.indices)

train_audio_loader = DataLoader(train_audio, batch_size=16, shuffle=True)
val_audio_loader = DataLoader(val_audio, batch_size=16, shuffle=False)
test_audio_loader = DataLoader(test_audio, batch_size=16, shuffle=False)

# %%
net5 = Net5(input_size=audio_dataset.n_features, hidden_size=64, num_classes=len(audio_dataset.classes))
result5, extras5 = train_classifier(
    model=net5,
    train_loader=train_audio_loader,
    val_loader=val_audio_loader,
    test_loader=test_audio_loader,
    device=device,
    epochs=60,
    optimizer_name='adam',
    learning_rate=1e-3,
    class_names=audio_dataset.classes,
)
result5.model_name = 'Net5'
save_result(output_dir, result5, extras5)

# %%
augmented_train = augment_with_gan(
    train_audio,
    num_classes=len(audio_dataset.classes),
    device=device,
    gan_waveform_length=16384,
    gan_epochs=30,
)
augmented_loader = DataLoader(augmented_train, batch_size=16, shuffle=True)
net6 = Net6(input_size=audio_dataset.n_features, hidden_size=64, num_classes=len(audio_dataset.classes))
result6, extras6 = train_classifier(
    model=net6,
    train_loader=augmented_loader,
    val_loader=val_audio_loader,
    test_loader=test_audio_loader,
    device=device,
    epochs=60,
    optimizer_name='adam',
    learning_rate=1e-3,
    class_names=audio_dataset.classes,
)
result6.model_name = 'Net6'
save_result(output_dir, result6, extras6)
