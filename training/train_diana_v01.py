from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io.wavfile import read
from openwakeword.utils import AudioFeatures


# ============================================================
# DIANA v0.1 - Custom Wake Word Trainer
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
POSITIVE_DIR = BASE_DIR / "positive"
NEGATIVE_DIR = BASE_DIR / "negative"
OUTPUT_DIR = BASE_DIR / "output"

SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
RANDOM_SEED = 42


def load_audio(directory):
    files = sorted(directory.glob("*.wav"))

    if not files:
        raise RuntimeError(f"No WAV files found in {directory}")

    audio = []

    for path in files:
        rate, samples = read(path)

        if rate != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: expected {SAMPLE_RATE} Hz, got {rate} Hz"
            )

        if samples.dtype != np.int16:
            raise ValueError(
                f"{path.name}: expected int16 audio, got {samples.dtype}"
            )

        if samples.ndim != 1:
            raise ValueError(
                f"{path.name}: expected mono audio, got shape {samples.shape}"
            )

        audio.append(samples)

    return np.stack(audio), files


def main():
    print("=" * 60)
    print("          DIANA v0.1 WAKE-WORD TRAINER")
    print("=" * 60)
    print()

    np.random.seed(RANDOM_SEED)

    print("Loading audio...")

    positive_audio, positive_files = load_audio(POSITIVE_DIR)
    negative_audio, negative_files = load_audio(NEGATIVE_DIR)

    print(f"Positive clips: {len(positive_files)}")
    print(f"Negative clips: {len(negative_files)}")
    print()

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("Initializing OpenWakeWord feature extractor...")

    feature_extractor = AudioFeatures(
        ncpu=4,
        inference_framework="onnx",
        device="cpu",
    )

    print("Extracting positive features...")

    positive_features = feature_extractor.embed_clips(
        positive_audio,
        batch_size=BATCH_SIZE,
        ncpu=4,
    )

    print("Extracting negative features...")

    negative_features = feature_extractor.embed_clips(
        negative_audio,
        batch_size=BATCH_SIZE,
        ncpu=4,
    )

    print()
    print("Positive feature shape:", positive_features.shape)
    print("Negative feature shape:", negative_features.shape)
    print()

    # --------------------------------------------------------
    # Prepare training examples
    #
    # Each 2-second clip produces 16 feature frames.
    # We turn each clip into one flattened feature vector.
    # --------------------------------------------------------

    X_positive = positive_features.reshape(len(positive_features), -1)
    X_negative = negative_features.reshape(len(negative_features), -1)

    X = np.concatenate([X_positive, X_negative], axis=0)

    y = np.concatenate(
        [
            np.ones(len(X_positive), dtype=np.float32),
            np.zeros(len(X_negative), dtype=np.float32),
        ]
    )

    print("Training matrix:", X.shape)
    print("Labels:", y.shape)
    print()

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    indices = np.random.permutation(len(X))

    X = X[indices]
    y = y[indices]

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split = int(len(X) * 0.8)

    X_train = X[:split]
    y_train = y[:split]

    X_val = X[split:]
    y_val = y[split:]

    print(f"Training samples:   {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print()

    # --------------------------------------------------------
    # Standardize features
    # --------------------------------------------------------

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    preprocessing_path = OUTPUT_DIR / "diana_v01_preprocessing.npz"
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        preprocessing_path,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )

    print(
        f"Preprocessing data: {preprocessing_path}"
    )

    # --------------------------------------------------------
    # Build small DNN
    # --------------------------------------------------------

    import torch

    torch.manual_seed(RANDOM_SEED)

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    input_size = X_train.shape[1]

    network = torch.nn.Sequential(
        torch.nn.Linear(input_size, 64),
        torch.nn.ReLU(),

        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),

        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid(),
    )

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = torch.nn.BCELoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_accuracy = -1.0
    best_state = None

    print("Training...")
    print()

    for epoch in range(1, EPOCHS + 1):

        network.train()

        permutation = torch.randperm(
            len(X_train_tensor)
        )

        total_loss = 0.0

        for start in range(
            0,
            len(X_train_tensor),
            BATCH_SIZE,
        ):

            batch_indices = permutation[
                start:start + BATCH_SIZE
            ]

            xb = X_train_tensor[batch_indices]
            yb = y_train_tensor[batch_indices]

            optimizer.zero_grad()

            predictions = network(xb).squeeze(-1)

            loss = loss_function(
                predictions,
                yb,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        network.eval()

        with torch.no_grad():

            validation_predictions = (
                network(X_val_tensor)
                .squeeze(-1)
            )

            validation_labels = (
                validation_predictions >= 0.5
            ).float()

            accuracy = (
                validation_labels == y_val_tensor
            ).float().mean().item()

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in network.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0:

            average_loss = (
                total_loss
                / max(1, len(X_train_tensor) // BATCH_SIZE)
            )

            print(
                f"Epoch {epoch:03d} | "
                f"loss={average_loss:.4f} | "
                f"val_accuracy={accuracy:.3f}"
            )

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model."
        )

    network.load_state_dict(best_state)
    network.eval()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Export TorchScript
    # --------------------------------------------------------

    torchscript_path = (
        OUTPUT_DIR / "diana_v01.pt"
    )

    example_input = torch.randn(
        1,
        input_size,
        dtype=torch.float32,
    )

    traced_model = torch.jit.trace(
        network,
        example_input,
    )

    traced_model.save(
        str(torchscript_path)
    )

    # --------------------------------------------------------
    # Export ONNX
    # --------------------------------------------------------

    onnx_path = (
        OUTPUT_DIR / "diana_v01.onnx"
    )

    torch.onnx.export(
        network,
        example_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {
                0: "batch"
            },
            "output": {
                0: "batch"
            },
        },
        opset_version=17,
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()
    print(
        f"Best validation accuracy: "
        f"{best_accuracy:.3f}"
    )
    print()
    print(f"TorchScript model: {torchscript_path}")
    print(f"ONNX model:        {onnx_path}")
    print()

    # --------------------------------------------------------
    # Verify ONNX model can actually load
    # --------------------------------------------------------

    print("Testing ONNX model...")

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    result = session.run(
        None,
        {
            "input": X_val[:1].astype(np.float32)
        },
    )

    print(
        "ONNX output:",
        result[0]
    )

    print()
    print("DIANA v0.1 is ready for testing.")


if __name__ == "__main__":
    main()