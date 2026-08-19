from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io.wavfile import read
from openwakeword.utils import AudioFeatures


# ============================================================
# DIANA v0.1 - Evaluation
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

POSITIVE_DIR = BASE_DIR / "positive"
NEGATIVE_DIR = BASE_DIR / "negative"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = OUTPUT_DIR / "diana_v01.onnx"
PREPROCESSING_PATH = OUTPUT_DIR / "diana_v01_preprocessing.npz"

SAMPLE_RATE = 16000
BATCH_SIZE = 16
THRESHOLD = 0.5


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


def evaluate_directory(
    directory,
    expected_label,
    feature_extractor,
    session,
    mean,
    std,
):
    audio, files = load_audio(directory)

    features = feature_extractor.embed_clips(
        audio,
        batch_size=BATCH_SIZE,
        ncpu=4,
    )

    X = features.reshape(len(features), -1)

    X = (X - mean) / std

    predictions = session.run(
        None,
        {
            "input": X.astype(np.float32),
        },
    )[0].reshape(-1)

    results = []

    for path, score in zip(files, predictions):
        predicted_label = int(score >= THRESHOLD)

        correct = predicted_label == expected_label

        results.append(
            {
                "file": path.name,
                "score": float(score),
                "predicted": predicted_label,
                "expected": expected_label,
                "correct": correct,
            }
        )

    return results


def main():
    print("=" * 60)
    print("             DIANA v0.1 EVALUATION")
    print("=" * 60)
    print()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing data not found: {PREPROCESSING_PATH}"
        )

    print("Loading preprocessing parameters...")

    preprocessing = np.load(PREPROCESSING_PATH)

    mean = preprocessing["mean"]
    std = preprocessing["std"]

    print("Mean shape:", mean.shape)
    print("Std shape: ", std.shape)
    print()

    print("Loading ONNX model...")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )

    print("Initializing OpenWakeWord feature extractor...")

    feature_extractor = AudioFeatures(
        ncpu=4,
        inference_framework="onnx",
        device="cpu",
    )

    print()
    print("Evaluating positive recordings...")

    positive_results = evaluate_directory(
        POSITIVE_DIR,
        expected_label=1,
        feature_extractor=feature_extractor,
        session=session,
        mean=mean,
        std=std,
    )

    print("Evaluating negative recordings...")

    negative_results = evaluate_directory(
        NEGATIVE_DIR,
        expected_label=0,
        feature_extractor=feature_extractor,
        session=session,
        mean=mean,
        std=std,
    )

    all_results = positive_results + negative_results

    true_positive = sum(
        r["expected"] == 1 and r["predicted"] == 1
        for r in all_results
    )

    true_negative = sum(
        r["expected"] == 0 and r["predicted"] == 0
        for r in all_results
    )

    false_positive = sum(
        r["expected"] == 0 and r["predicted"] == 1
        for r in all_results
    )

    false_negative = sum(
        r["expected"] == 1 and r["predicted"] == 0
        for r in all_results
    )

    total = len(all_results)

    accuracy = (
        (true_positive + true_negative) / total
        if total
        else 0.0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 0.0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if (true_negative + false_positive)
        else 0.0
    )

    false_positive_rate = 1.0 - specificity

    false_negative_rate = 1.0 - recall

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    print()
    print("Confusion matrix:")
    print()
    print(f"True positives:   {true_positive}")
    print(f"False negatives:  {false_negative}")
    print(f"True negatives:   {true_negative}")
    print(f"False positives:  {false_positive}")

    print()
    print("Metrics:")
    print()
    print(f"Accuracy:          {accuracy:.3f}")
    print(f"Precision:         {precision:.3f}")
    print(f"Recall:            {recall:.3f}")
    print(f"Specificity:       {specificity:.3f}")
    print(f"False positive:    {false_positive_rate:.3f}")
    print(f"False negative:    {false_negative_rate:.3f}")

    print()
    print("=" * 60)
    print("INDIVIDUAL SCORES")
    print("=" * 60)

    print()
    print("POSITIVE RECORDINGS")
    print("-" * 60)

    for result in positive_results:
        status = "PASS" if result["correct"] else "FAIL"

        print(
            f"{result['file']:20} "
            f"score={result['score']:.4f} "
            f"{status}"
        )

    print()
    print("NEGATIVE RECORDINGS")
    print("-" * 60)

    for result in negative_results:
        status = "PASS" if result["correct"] else "FAIL"

        print(
            f"{result['file']:20} "
            f"score={result['score']:.4f} "
            f"{status}"
        )

    print()
    print("=" * 60)
    print("DIANA v0.1 EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()