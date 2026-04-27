from __future__ import annotations

import argparse
import math
import random
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import wavfile
from scipy.signal import resample, stft
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, random_split
from tqdm.auto import tqdm

try:
    from torchvision import datasets, transforms
except ModuleNotFoundError:
    datasets = None
    transforms = None

try:
    import librosa
except ModuleNotFoundError:
    librosa = None


SEED = 42
IMAGE_SIZE = (180, 180)
IMAGE_EPOCH_OPTIONS = (50, 100)
DEFAULT_AUDIO_EPOCHS = 80
DEFAULT_DATASET_ROOT = Path("dataset") / "gtzan-dataset-music-genre-classification"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResizeToTensor:
    """Resize to 180x180 and normalise images for Net1-Net4."""

    def __init__(self, size: tuple[int, int] = IMAGE_SIZE) -> None:
        if transforms is not None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            warnings.warn(
                "torchvision is not installed, so a small PIL fallback is being used. "
                "Install torchvision for the exact coursework Resize transform.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.transform = None
            self.size = size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.transform is not None:
            return self.transform(image)

        image = image.resize(self.size, Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - 0.5) / 0.5
        return torch.from_numpy(array.transpose(2, 0, 1))


class _SimpleImageFolder(Dataset):
    """Small replacement for torchvision.datasets.ImageFolder."""

    def __init__(self, root: Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = self.root / class_name
            for pattern in ("*.png", "*.jpg", "*.jpeg"):
                self.samples.extend((path, self.class_to_idx[class_name]) for path in sorted(class_dir.glob(pattern)))

        if not self.samples:
            raise RuntimeError(f"No image files found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class ImageFolderDataset(Dataset):
    """Uses torchvision ImageFolder when available, otherwise the fallback above."""

    def __init__(self, root: str | Path, transform: Callable | None = None) -> None:
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Image folder not found: {root}")

        if datasets is not None:
            self.dataset = datasets.ImageFolder(root, transform=transform)
        else:
            self.dataset = _SimpleImageFolder(root, transform=transform)

        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.dataset[index]


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    audio = audio.astype(np.float32, copy=False)
    if audio.size >= target_length:
        return audio[:target_length]

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: audio.size] = audio
    return padded


def normalise_waveform(audio: np.ndarray) -> np.ndarray:
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(np.iinfo(audio.dtype).max, 1)
    else:
        audio = audio.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.nan_to_num(audio)


class AudioGenreDataset(Dataset):
    """Loads wav files and turns each clip into a log-Mel sequence for the LSTM."""

    def __init__(
        self,
        root: str | Path,
        clip_seconds: int = 30,
        target_sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 2048,
        n_mels: int = 128,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Audio folder not found: {self.root}")

        self.clip_seconds = clip_seconds
        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_length = clip_seconds * target_sample_rate

        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            for path in sorted((self.root / class_name).glob("*.wav")):
                if self._is_readable_wav(path):
                    self.samples.append((path, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No readable wav files found in {self.root}")

    @staticmethod
    def _is_readable_wav(path: Path) -> bool:
        try:
            wavfile.read(path, mmap=True)
            return True
        except Exception:
            return False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self._cached_item(index)

    @lru_cache(maxsize=None)
    def _cached_item(self, index: int) -> tuple[torch.Tensor, int]:
        waveform, label = self.load_processed_audio(index)
        return self.audio_to_sequence(waveform), label

    def load_processed_audio(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]

        if librosa is not None:
            audio, _ = librosa.load(
                path,
                sr=self.target_sample_rate,
                mono=True,
                duration=self.clip_seconds,
            )
        else:
            sample_rate, audio = wavfile.read(path)
            audio = normalise_waveform(audio)
            if sample_rate != self.target_sample_rate:
                new_length = round(audio.size * self.target_sample_rate / sample_rate)
                audio = resample(audio, new_length).astype(np.float32)

        audio = pad_or_trim(audio, self.target_length)
        return torch.from_numpy(audio), label

    def audio_to_sequence(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        if librosa is not None:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=self.target_sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            features = librosa.power_to_db(mel, ref=np.max)
        else:
            _, _, zxx = stft(
                audio,
                fs=self.target_sample_rate,
                nperseg=self.n_fft,
                noverlap=self.n_fft - self.hop_length,
                boundary=None,
                padded=False,
            )
            power = np.abs(zxx) ** 2
            features = np.log1p(resample(power, self.n_mels, axis=0))

        features = features.T.astype(np.float32)
        features = (features - features.mean()) / (features.std() + 1e-6)
        return torch.from_numpy(features)


class AudioFeatureSequenceDataset(Dataset):
    """Uses GTZAN 3-second audio features as a short sequence for the LSTM."""

    def __init__(self, csv_path: str | Path, sequence_length: int = 10) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Audio feature CSV not found: {self.csv_path}")

        frame = pd.read_csv(self.csv_path)
        frame["track_id"] = frame["filename"].str.replace(r"\.\d+\.wav$", ".wav", regex=True)
        frame["segment_id"] = frame["filename"].str.extract(r"\.(\d+)\.wav$").astype(int)

        self.sequence_length = sequence_length
        self.classes = sorted(frame["label"].unique())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.feature_columns = [
            column
            for column in frame.select_dtypes(include=[np.number]).columns
            if column not in {"length", "segment_id"}
        ]
        self.n_features = len(self.feature_columns)
        self.samples: list[tuple[str, int]] = []
        self.sequences: list[np.ndarray] = []
        self.targets: list[int] = []

        for track_id, group in frame.groupby("track_id", sort=True):
            group = group.sort_values("segment_id")
            sequence = group[self.feature_columns].to_numpy(dtype=np.float32)[:sequence_length]
            if len(sequence) < sequence_length:
                sequence = np.pad(sequence, ((0, sequence_length - len(sequence)), (0, 0)), mode="edge")

            label = self.class_to_idx[group["label"].iloc[0]]
            self.samples.append((track_id, label))
            self.sequences.append(sequence)
            self.targets.append(label)

        self.fit_standardizer()

    @classmethod
    def from_dataset_root(cls, dataset_root: str | Path, sequence_length: int = 10) -> "AudioFeatureSequenceDataset":
        csv_path = find_audio_feature_csv(Path(dataset_root))
        if csv_path is None:
            raise FileNotFoundError("Could not find features_3_sec.csv near the dataset root.")
        return cls(csv_path, sequence_length=sequence_length)

    def fit_standardizer(self, indices: list[int] | tuple[int, ...] | np.ndarray | None = None) -> None:
        if indices is None:
            selected = self.sequences
        else:
            selected = [self.sequences[int(index)] for index in indices]

        stacked = np.concatenate(selected, axis=0)
        self.mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
        self.std = (stacked.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sequence = (self.sequences[index] - self.mean) / self.std
        return torch.from_numpy(sequence.astype(np.float32)), self.targets[index]


def split_dataset(dataset: Dataset, seed: int = SEED) -> tuple[Subset, Subset, Subset]:
    labels = None
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "samples"):
        labels = np.asarray([sample[1] for sample in dataset.samples])

    if labels is not None and len(labels) == len(dataset):
        generator = torch.Generator().manual_seed(seed)
        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for label in sorted(set(labels.tolist())):
            class_indices = torch.tensor(np.where(labels == label)[0])
            class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)].tolist()
            train_end = int(0.7 * len(class_indices))
            val_end = train_end + int(0.2 * len(class_indices))
            train_indices.extend(class_indices[:train_end])
            val_indices.extend(class_indices[train_end:val_end])
            test_indices.extend(class_indices[val_end:])

        train_indices = torch.tensor(train_indices)[torch.randperm(len(train_indices), generator=generator)].tolist()
        val_indices = torch.tensor(val_indices)[torch.randperm(len(val_indices), generator=generator)].tolist()
        test_indices = torch.tensor(test_indices)[torch.randperm(len(test_indices), generator=generator)].tolist()
        return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)
    test_size = total - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class Net1(nn.Module):
    """Fully connected network with two hidden layers."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        flat_dim = 3 * IMAGE_SIZE[0] * IMAGE_SIZE[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def conv_block(in_channels: int, out_channels: int, batch_norm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class FigureCNN(nn.Module):
    """CNN used for Net2-Net4. Batch norm is switched on for Net3/Net4."""

    def __init__(self, num_classes: int = 10, batch_norm: bool = False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, batch_norm),
            conv_block(32, 64, batch_norm),
            nn.MaxPool2d(2),
            conv_block(64, 128, batch_norm),
            conv_block(128, 128, batch_norm),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class Net2(FigureCNN):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes, batch_norm=False)


class Net3(FigureCNN):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes, batch_norm=True)


class Net4(FigureCNN):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes, batch_norm=True)


class Net5(nn.Module):
    """LSTM classifier for audio log-Mel sequences."""

    def __init__(self, input_size: int = 128, hidden_size: int = 96, num_layers: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden.view(self.lstm.num_layers, 2, x.size(0), self.lstm.hidden_size)
        last_layer = torch.cat((hidden[-1, 0], hidden[-1, 1]), dim=1)
        return self.classifier(last_layer)


class Net6(Net5):
    """Same LSTM as Net5; training data is augmented by the GAN helper."""


class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int, num_classes: int, output_dim: int) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat((noise, self.label_embed(labels)), dim=1)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 32)
        self.net = nn.Sequential(
            nn.Linear(input_dim + 32, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, samples: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = torch.cat((samples, self.label_embed(labels)), dim=1)
        return self.net(x)


@dataclass
class ExperimentResult:
    model_name: str
    epochs: int
    optimizer: str
    val_loss: float
    val_accuracy: float
    test_loss: float
    test_accuracy: float


def build_optimizer(name: str, model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    if name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    raise ValueError(f"Unknown optimizer: {name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += batch_size

    return total_loss / total_items, total_correct / total_items


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs.to(device))
            y_true.append(labels.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer_name: str,
    learning_rate: float,
    class_names: list[str] | None = None,
) -> tuple[ExperimentResult, dict[str, object]]:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model, learning_rate)
    model.to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_acc = -1.0
    best_val_loss = math.inf
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, epochs + 1), desc=model.__class__.__name__):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc or (math.isclose(val_acc, best_val_acc) and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"{model.__class__.__name__} epoch {epoch:03d}/{epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                f"best_val_acc={best_val_acc:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = run_epoch(model, test_loader, loss_fn, device)
    y_true, y_pred = predict_all(model, test_loader, device)
    report_kwargs = {"zero_division": 0}
    matrix_kwargs = {}
    if class_names is not None:
        report_kwargs["target_names"] = class_names
        report_kwargs["labels"] = list(range(len(class_names)))
        matrix_kwargs["labels"] = list(range(len(class_names)))

    result = ExperimentResult(
        model_name=model.__class__.__name__,
        epochs=epochs,
        optimizer=optimizer_name,
        val_loss=best_val_loss,
        val_accuracy=best_val_acc,
        test_loss=test_loss,
        test_accuracy=test_acc,
    )
    extras = {
        "classification_report": classification_report(y_true, y_pred, **report_kwargs),
        "confusion_matrix": confusion_matrix(y_true, y_pred, **matrix_kwargs),
        "history": history,
    }
    return result, extras


def resample_waveform_tensor(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    if waveform.numel() == target_length:
        return waveform.float()
    return torch.from_numpy(resample(waveform.numpy(), target_length).astype(np.float32))


def train_conditional_gan(
    waveforms: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    epochs: int = 30,
    noise_dim: int = 64,
    batch_size: int = 32,
) -> ConditionalGenerator:
    loader = DataLoader(TensorDataset(waveforms, labels), batch_size=batch_size, shuffle=True)
    generator = ConditionalGenerator(noise_dim, num_classes, waveforms.shape[1]).to(device)
    discriminator = ConditionalDiscriminator(waveforms.shape[1], num_classes).to(device)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(1, epochs + 1), desc="GAN"):
        for real_samples, real_labels in loader:
            real_samples = real_samples.to(device)
            real_labels = real_labels.to(device)
            batch_size_now = real_samples.size(0)
            real_targets = torch.ones(batch_size_now, 1, device=device)
            fake_targets = torch.zeros(batch_size_now, 1, device=device)

            noise = torch.randn(batch_size_now, noise_dim, device=device)
            fake_samples = generator(noise, real_labels)

            disc_opt.zero_grad()
            real_loss = loss_fn(discriminator(real_samples, real_labels), real_targets)
            fake_loss = loss_fn(discriminator(fake_samples.detach(), real_labels), fake_targets)
            (real_loss + fake_loss).backward()
            disc_opt.step()

            gen_opt.zero_grad()
            gen_loss = loss_fn(discriminator(fake_samples, real_labels), real_targets)
            gen_loss.backward()
            gen_opt.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"GAN epoch {epoch:03d}/{epochs} gen_loss={gen_loss.item():.3f}")

    return generator


def gather_audio_waveforms(train_subset: Dataset) -> tuple[torch.Tensor, torch.Tensor, AudioGenreDataset]:
    if not isinstance(train_subset, Subset) or not isinstance(train_subset.dataset, AudioGenreDataset):
        raise TypeError("GAN augmentation expects a Subset created from AudioGenreDataset.")

    dataset = train_subset.dataset
    waveforms: list[torch.Tensor] = []
    labels: list[int] = []

    for index in train_subset.indices:
        waveform, label = dataset.load_processed_audio(index)
        waveforms.append(waveform)
        labels.append(label)

    return torch.stack(waveforms), torch.tensor(labels, dtype=torch.long), dataset


def gather_sequence_features(train_subset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(train_subset, Subset):
        raise TypeError("Feature GAN augmentation expects a Subset.")

    features: list[torch.Tensor] = []
    labels: list[int] = []
    for index in train_subset.indices:
        feature, label = train_subset.dataset[int(index)]
        features.append(feature)
        labels.append(label)

    return torch.stack(features), torch.tensor(labels, dtype=torch.long)


def augment_with_gan(
    train_subset: Dataset,
    num_classes: int,
    device: torch.device,
    gan_waveform_length: int = 16384,
    gan_epochs: int = 30,
) -> TensorDataset:
    if isinstance(train_subset, Subset) and isinstance(train_subset.dataset, AudioFeatureSequenceDataset):
        real_sequences, labels = gather_sequence_features(train_subset)
        flat_sequences = real_sequences.flatten(start_dim=1)
        generator = train_conditional_gan(
            flat_sequences,
            labels,
            num_classes=num_classes,
            device=device,
            epochs=gan_epochs,
        )

        noise = torch.randn(labels.size(0), 64, device=device)
        with torch.no_grad():
            fake_sequences = generator(noise, labels.to(device)).cpu().view_as(real_sequences)

        features = torch.cat((real_sequences, fake_sequences), dim=0)
        all_labels = torch.cat((labels, labels), dim=0)
        return TensorDataset(features, all_labels)

    waveforms, labels, dataset = gather_audio_waveforms(train_subset)
    compact_length = min(gan_waveform_length, dataset.target_length)
    compact_waveforms = torch.stack([resample_waveform_tensor(waveform, compact_length) for waveform in waveforms])

    generator = train_conditional_gan(
        compact_waveforms,
        labels,
        num_classes=num_classes,
        device=device,
        epochs=gan_epochs,
    )

    noise = torch.randn(labels.size(0), 64, device=device)
    with torch.no_grad():
        fake_compact = generator(noise, labels.to(device)).cpu()

    fake_waveforms = torch.stack(
        [resample_waveform_tensor(waveform, dataset.target_length) for waveform in fake_compact]
    ).clamp(-1.0, 1.0)

    real_sequences = torch.stack([dataset.audio_to_sequence(waveform) for waveform in waveforms])
    fake_sequences = torch.stack([dataset.audio_to_sequence(waveform) for waveform in fake_waveforms])
    features = torch.cat((real_sequences, fake_sequences), dim=0)
    all_labels = torch.cat((labels, labels), dim=0)
    return TensorDataset(features, all_labels)


def plot_training_curves(history: pd.DataFrame, output_dir: Path, stem: str) -> None:
    plots = [
        (
            "accuracy",
            "Accuracy",
            [("train_acc", "Train accuracy"), ("val_acc", "Validation accuracy")],
            output_dir / f"{stem}_accuracy_curve.png",
        ),
        (
            "error",
            "Error rate",
            [("train_error", "Train error"), ("val_error", "Validation error")],
            output_dir / f"{stem}_error_curve.png",
        ),
        (
            "loss",
            "Loss",
            [("train_loss", "Train loss"), ("val_loss", "Validation loss")],
            output_dir / f"{stem}_loss_curve.png",
        ),
    ]

    history = history.copy()
    history["train_error"] = 1.0 - history["train_acc"]
    history["val_error"] = 1.0 - history["val_acc"]

    for title, ylabel, columns, path in plots:
        plt.figure(figsize=(7, 4))
        for column, label in columns:
            plt.plot(history["epoch"], history[column], label=label)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"{stem} {title} curve")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()


def save_result(output_dir: Path, result: ExperimentResult, extras: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"
    pd.DataFrame([asdict(result)]).to_csv(csv_path, mode="a", index=False, header=not csv_path.exists())

    stem = f"{result.model_name.lower()}_{result.epochs}ep_{result.optimizer.lower()}"
    (output_dir / f"{stem}_report.txt").write_text(str(extras["classification_report"]), encoding="utf-8")
    np.savetxt(
        output_dir / f"{stem}_confusion_matrix.csv",
        np.asarray(extras["confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )
    history = pd.DataFrame(extras["history"])
    history.to_csv(output_dir / f"{stem}_history.csv", index=False)
    plot_training_curves(history, output_dir, stem)


def find_audio_feature_csv(dataset_root: Path, audio_root: Path | None = None) -> Path | None:
    candidates = [
        dataset_root / "features_3_sec.csv",
        dataset_root / "Data" / "features_3_sec.csv",
    ]
    if audio_root is not None:
        candidates.extend(
            [
                audio_root / "features_3_sec.csv",
                audio_root.parent / "features_3_sec.csv",
                audio_root.parent / "Data" / "features_3_sec.csv",
            ]
        )
    return next((path for path in candidates if path.exists()), None)


def resolve_dataset_roots(dataset_root: Path) -> tuple[Path, Path]:
    image_candidates = [dataset_root / "images_original", dataset_root / "Data" / "images_original", dataset_root]
    audio_candidates = [dataset_root / "genres_original", dataset_root / "Data" / "genres_original", dataset_root]
    image_root = next((path for path in image_candidates if path.exists()), image_candidates[0])
    audio_root = next((path for path in audio_candidates if path.exists()), audio_candidates[0])
    return image_root, audio_root


def build_audio_dataset(args: argparse.Namespace) -> Dataset:
    dataset_root = Path(args.dataset_root)
    _, audio_root = resolve_dataset_roots(dataset_root)

    if args.audio_source == "features":
        csv_path = find_audio_feature_csv(dataset_root, audio_root)
        if csv_path is not None:
            return AudioFeatureSequenceDataset(csv_path, sequence_length=args.feature_sequence_length)
        warnings.warn(
            "features_3_sec.csv was not found, so the code is falling back to raw wav log-Mel features.",
            RuntimeWarning,
            stacklevel=2,
        )

    return AudioGenreDataset(
        audio_root,
        clip_seconds=args.clip_seconds,
        target_sample_rate=args.audio_sample_rate,
        n_fft=args.audio_n_fft,
        hop_length=args.audio_hop_length,
        n_mels=args.audio_n_mels,
    )


def run_image_experiments(args: argparse.Namespace, device: torch.device) -> None:
    image_root, _ = resolve_dataset_roots(Path(args.dataset_root))
    dataset = ImageFolderDataset(image_root, transform=ResizeToTensor(IMAGE_SIZE))
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    train_loader = make_loader(train_set, args.batch_size, shuffle=True)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)

    image_models = {
        "net1": (Net1, "adam"),
        "net2": (Net2, "adam"),
        "net3": (Net3, "adam"),
        "net4": (Net4, "rmsprop"),
    }

    for name in args.models:
        if name not in image_models:
            continue
        model_class, optimizer_name = image_models[name]
        for epochs in args.image_epochs:
            model = model_class(num_classes=len(dataset.classes))
            result, extras = train_classifier(
                model,
                train_loader,
                val_loader,
                test_loader,
                device=device,
                epochs=epochs,
                optimizer_name=optimizer_name,
                learning_rate=args.learning_rate,
                class_names=dataset.classes,
            )
            save_result(Path(args.output_dir), result, extras)


def run_audio_experiments(args: argparse.Namespace, device: torch.device) -> None:
    dataset = build_audio_dataset(args)
    train_set, val_set, test_set = split_dataset(dataset, seed=args.seed)
    if hasattr(dataset, "fit_standardizer"):
        dataset.fit_standardizer(train_set.indices)

    val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    test_loader = make_loader(test_set, args.batch_size, shuffle=False)
    input_size = getattr(dataset, "n_features", args.audio_n_mels)

    if "net5" in args.models:
        train_loader = make_loader(train_set, args.batch_size, shuffle=True)
        model = Net5(input_size=input_size, hidden_size=args.audio_hidden_size, num_classes=len(dataset.classes))
        result, extras = train_classifier(
            model,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            epochs=args.audio_epochs,
            optimizer_name="adam",
            learning_rate=args.audio_learning_rate,
            class_names=dataset.classes,
        )
        save_result(Path(args.output_dir), result, extras)

    if "net6" in args.models:
        augmented_train = augment_with_gan(
            train_set,
            num_classes=len(dataset.classes),
            device=device,
            gan_waveform_length=args.gan_waveform_length,
            gan_epochs=args.gan_epochs,
        )
        train_loader = make_loader(augmented_train, args.batch_size, shuffle=True)
        model = Net6(input_size=input_size, hidden_size=args.audio_hidden_size, num_classes=len(dataset.classes))
        result, extras = train_classifier(
            model,
            train_loader,
            val_loader,
            test_loader,
            device=device,
            epochs=args.audio_epochs,
            optimizer_name="adam",
            learning_rate=args.audio_learning_rate,
            class_names=dataset.classes,
        )
        save_result(Path(args.output_dir), result, extras)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="COMP6252 CW1 music genre classifiers")
    parser.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["net1", "net2", "net3", "net4", "net5", "net6"],
        choices=["net1", "net2", "net3", "net4", "net5", "net6"],
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--audio-learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-epochs", nargs="+", type=int, default=list(IMAGE_EPOCH_OPTIONS))
    parser.add_argument("--audio-epochs", type=int, default=DEFAULT_AUDIO_EPOCHS)
    parser.add_argument("--audio-source", choices=["features", "wav"], default="features")
    parser.add_argument("--feature-sequence-length", type=int, default=10)
    parser.add_argument("--audio-hidden-size", type=int, default=64)
    parser.add_argument("--clip-seconds", type=int, default=30)
    parser.add_argument("--audio-sample-rate", type=int, default=22050)
    parser.add_argument("--audio-n-fft", type=int, default=2048)
    parser.add_argument("--audio-hop-length", type=int, default=2048)
    parser.add_argument("--audio-n-mels", type=int, default=128)
    parser.add_argument("--gan-waveform-length", type=int, default=16384)
    parser.add_argument("--gan-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    selected_models = set(args.models)

    if selected_models & {"net1", "net2", "net3", "net4"}:
        run_image_experiments(args, device)
    if selected_models & {"net5", "net6"}:
        run_audio_experiments(args, device)


if __name__ == "__main__":
    main()
