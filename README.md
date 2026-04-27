# COMP6252 Coursework 1

This workspace contains a complete, runnable code solution for the GTZAN music genre classification assignment.

## Files

- `music_genre_coursework.py`: end-to-end training pipeline for `Net1` to `Net6`
- `coursework1_runner.ipynb`: notebook wrapper that mirrors the coursework workflow
- `coursework1_runner.py`: VS Code/Jupyter-style script version of the notebook
- `requirements.txt`: recommended packages for the cleaner torchvision/librosa version
- `report_outline.md`: a 4-page report structure you can fill with your experiment outputs

## Dataset layout expected

Download the GTZAN Kaggle dataset and point the script to the dataset root. The code supports the common layout:

```text
gtzan-dataset-music-genre-classification/
  images_original/
    blues/
    classical/
    ...
  genres_original/
    blues/
    classical/
    ...
```

## Run from terminal

Install the recommended packages:

```bash
pip install -r requirements.txt
```

Train all six models using the dataset already stored in this workspace:

```bash
python music_genre_coursework.py
```

Or give a different dataset path:

```bash
python music_genre_coursework.py --dataset-root D:\path\to\gtzan-dataset-music-genre-classification
```

Run only the image models:

```bash
python music_genre_coursework.py --dataset-root D:\path\to\gtzan-dataset-music-genre-classification --models net1 net2 net3 net4
```

Run only the audio models:

```bash
python music_genre_coursework.py --dataset-root D:\path\to\gtzan-dataset-music-genre-classification --models net5 net6 --audio-epochs 80
```

By default, `Net5` and `Net6` use the GTZAN `features_3_sec.csv` file as a 10-step audio feature sequence for the LSTM. This is much stronger and more stable than learning directly from the raw waveform on only 999 clips. To force the raw wav log-Mel version instead, run:

```bash
python music_genre_coursework.py --models net5 net6 --audio-source wav
```

## Coursework mapping

- `Net1`: fully connected network with two hidden layers
- `Net2`: CNN matching the coursework Figure 1 layout
- `Net3`: `Net2` plus batch normalization
- `Net4`: `Net3` trained with RMSProp
- `Net5`: LSTM classifier on audio feature sequences
- `Net6`: same LSTM as `Net5`, trained with GAN-generated audio feature sequences

## Notes

- For image experiments, the code resizes inputs to `180x180` and splits data into `70/20/10`.
- For image models, the default epochs are `50` and `100` as required in the PDF.
- `torchvision.transforms.Resize` and `torchvision.datasets.ImageFolder` are used when `torchvision` is installed; otherwise the code falls back to a small local loader so it can still run in minimal environments.
- `librosa` is used for log-Mel audio features when installed; otherwise a SciPy log-spectrogram fallback is used.
- Training prints both loss and accuracy during training.
- The audio training log also prints `best_val_acc`, because later epochs may overfit even when the best validation checkpoint is good.
- Outputs are saved under `outputs/` as a CSV results table, classification reports, confusion matrices, history CSV files, and accuracy/error/loss curve PNGs.
