from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io.wavfile import read
from openwakeword.utils import AudioFeatures


# ============================================================
# DIANA v0.1 - Unseen Test Evaluation
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POSITIVE_DIR = BASE_DIR / "test" / "positive"
NEGATIVE_DIR = BASE_DIR / "test" / "negative"

MODEL_PATH = BASE_DIR / "output" / "diana_v01.onnx"
PREPROCESSING_PATH = (
    BASE_DIR / "output" / "diana_v01_preprocessing.npz"
)

SAMPLE_RATE = 16000
THRESHOLD = 0.5


def load_audio(directory):
    files = sorted(directory.glob("*.wav"))

    if not files:
        raise RuntimeError(
            f"No WAV files found in {directory}"
        )

    audio = []

    for path in files:
        rate, samples = read(path)

        if rate != SAMPLE_RATE:
            raise ValueError(
                f"{path.name}: expected "
                f"{SAMPLE_RATE} Hz, got {rate} Hz"
            )

        if samples.dtype != np.int16:
            raise ValueError(
                f"{path.name}: expected int16 audio, "
                f"got {samples.dtype}"
            )

        if samples.ndim != 1:
            raise ValueError(
                f"{path.name}: expected mono audio, "
                f"got shape {samples.shape}"
            )

        audio.append(samples)

    return np.stack(audio), files


def extract_features(audio, feature_extractor):
    features = feature_extractor.embed_clips(
        audio,
        batch_size=16,
        ncpu=4,
    )

    return features.reshape(len(features), -1)


def predict(
    features,
    session,
    mean,
    std,
):
    features = (features - mean) / std

    result = session.run(
        None,
        {
            "input": features.astype(np.float32)
        },
    )

    return result[0].reshape(-1)


def main():

    print("=" * 60)
    print("          DIANA v0.1 UNSEEN TEST")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load preprocessing
    # --------------------------------------------------------

    print("Loading preprocessing parameters...")

    preprocessing = np.load(PREPROCESSING_PATH)

    mean = preprocessing["mean"]
    std = preprocessing["std"]

    print("Mean shape:", mean.shape)
    print("Std shape: ", std.shape)
    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading frozen ONNX model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # Feature extractor
    # --------------------------------------------------------

    print(
        "Initializing OpenWakeWord feature extractor..."
    )

    feature_extractor = AudioFeatures(
        ncpu=4,
        inference_framework="onnx",
        device="cpu",
    )

    print()

    # --------------------------------------------------------
    # Load test recordings
    # --------------------------------------------------------

    print("Loading unseen positive recordings...")

    positive_audio, positive_files = load_audio(
        POSITIVE_DIR
    )

    print(
        f"Positive test recordings: "
        f"{len(positive_files)}"
    )

    print("Loading unseen negative recordings...")

    negative_audio, negative_files = load_audio(
        NEGATIVE_DIR
    )

    print(
        f"Negative test recordings: "
        f"{len(negative_files)}"
    )

    print()

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    print("Extracting positive features...")

    positive_features = extract_features(
        positive_audio,
        feature_extractor,
    )

    print("Extracting negative features...")

    negative_features = extract_features(
        negative_audio,
        feature_extractor,
    )

    print()

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("Running frozen DIANA v0.1...")

    positive_scores = predict(
        positive_features,
        session,
        mean,
        std,
    )

    negative_scores = predict(
        negative_features,
        session,
        mean,
        std,
    )

    positive_predictions = (
        positive_scores >= THRESHOLD
    )

    negative_predictions = (
        negative_scores >= THRESHOLD
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    true_positives = int(
        np.sum(positive_predictions)
    )

    false_negatives = int(
        np.sum(~positive_predictions)
    )

    false_positives = int(
        np.sum(negative_predictions)
    )

    true_negatives = int(
        np.sum(~negative_predictions)
    )

    total = (
        true_positives
        + false_negatives
        + true_negatives
        + false_positives
    )

    accuracy = (
        (true_positives + true_negatives)
        / total
    )

    precision = (
        true_positives
        / max(
            1,
            true_positives + false_positives
        )
    )

    recall = (
        true_positives
        / max(
            1,
            true_positives + false_negatives
        )
    )

    specificity = (
        true_negatives
        / max(
            1,
            true_negatives + false_positives
        )
    )

    false_positive_rate = (
        false_positives
        / max(
            1,
            true_negatives + false_positives
        )
    )

    false_negative_rate = (
        false_negatives
        / max(
            1,
            true_positives + false_negatives
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("=" * 60)
    print("                 RESULTS")
    print("=" * 60)
    print()

    print("Confusion matrix:")
    print()
    print(
        f"True positives:   {true_positives}"
    )
    print(
        f"False negatives:  {false_negatives}"
    )
    print(
        f"True negatives:   {true_negatives}"
    )
    print(
        f"False positives:  {false_positives}"
    )

    print()
    print("Metrics:")
    print()

    print(
        f"Accuracy:          {accuracy:.3f}"
    )

    print(
        f"Precision:         {precision:.3f}"
    )

    print(
        f"Recall:            {recall:.3f}"
    )

    print(
        f"Specificity:       {specificity:.3f}"
    )

    print(
        f"False positive:    "
        f"{false_positive_rate:.3f}"
    )

    print(
        f"False negative:    "
        f"{false_negative_rate:.3f}"
    )

    # --------------------------------------------------------
    # Individual scores
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("             POSITIVE TEST SCORES")
    print("=" * 60)

    for path, score in zip(
        positive_files,
        positive_scores,
    ):

        status = (
            "PASS"
            if score >= THRESHOLD
            else "FAIL"
        )

        print(
            f"{path.name:<22}"
            f"score={score:.4f} {status}"
        )

    print()
    print("=" * 60)
    print("             NEGATIVE TEST SCORES")
    print("=" * 60)

    for path, score in zip(
        negative_files,
        negative_scores,
    ):

        status = (
            "FAIL"
            if score >= THRESHOLD
            else "PASS"
        )

        print(
            f"{path.name:<22}"
            f"score={score:.4f} {status}"
        )

    print()
    print("=" * 60)
    print("       DIANA v0.1 UNSEEN TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()