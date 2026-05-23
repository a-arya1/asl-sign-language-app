import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import train_test_split


LABEL_NAME = "Letter Label"
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 99,
    "n_jobs": -1,
}


def load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        return None, None
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    if data.size == 0:
        return None, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :-1].astype(float), data[:, -1]


def load_contribution_json_rows(contributions_dir):
    rows = []
    labels = []
    for metadata_path in Path(contributions_dir).glob("*/*.json"):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        features = metadata.get("features")
        letter = metadata.get("letter")
        if letter and features and len(features) == 72:
            rows.append(features)
            labels.append(letter)
    if not rows:
        return None, None
    return np.asarray(rows, dtype=float), np.asarray(labels)


def combine_datasets(datasets):
    xs = [x for x, _ in datasets if x is not None]
    ys = [y for _, y in datasets if y is not None]
    if not xs:
        raise FileNotFoundError("No training rows were found.")
    return np.vstack(xs), np.concatenate(ys)


def build_model():
    return RandomForestClassifier(**MODEL_PARAMS)


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }


def write_metrics(path, metrics, sample_count, contribution_count):
    payload = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "sample_count": sample_count,
        "contribution_count": contribution_count,
        "model_params": MODEL_PARAMS,
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train the ASL fingerspelling Random Forest model.")
    parser.add_argument("--base-csv", default="handsData.csv")
    parser.add_argument("--contribution-csv", default="contributions/training_rows.csv")
    parser.add_argument("--contributions-dir", default="contributions/pending")
    parser.add_argument("--output", default="hand_gesture_model.joblib")
    parser.add_argument("--candidate-output", default="hand_gesture_model.candidate.joblib")
    parser.add_argument("--metrics-output", default="training_metrics.json")
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.985)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()

    base = load_csv_rows(args.base_csv)
    contribution_csv = load_csv_rows(args.contribution_csv)
    contribution_json = load_contribution_json_rows(args.contributions_dir)
    x, y = combine_datasets([base, contribution_csv, contribution_json])
    contribution_count = sum(
        0 if rows is None else len(rows)
        for rows, _ in [contribution_csv, contribution_json]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=99,
        stratify=y,
    )
    model = build_model()
    model.fit(x_train, y_train)
    metrics = evaluate(model, x_test, y_test)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(metrics["report"])

    output = Path(args.output if args.promote else args.candidate_output)
    if args.promote and metrics["balanced_accuracy"] < args.min_balanced_accuracy:
        raise RuntimeError(
            f"Refusing to promote: balanced accuracy {metrics['balanced_accuracy']:.4f} "
            f"is below {args.min_balanced_accuracy:.4f}."
        )

    joblib.dump(model, output)
    write_metrics(args.metrics_output, metrics, len(x), contribution_count)
    print(f"Saved model to {output}")
    print(f"Saved metrics to {args.metrics_output}")


if __name__ == "__main__":
    main()
