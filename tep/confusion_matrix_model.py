"""MLP on split seed 0 and its test confusion matrix.

Input: ROOT/data/tep_clean.npz. Output: ROOT/out/tep_model_artifacts.npz (and the fitted model).
Paper: Supplementary Fig. S1."""
import os
import json
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data_split import episode_train_test_split
from fault_metadata import FAULT_INFO
ROOT = os.environ.get("TEP_ROOT", "../runs/revised")  # folder with data/tep_clean.npz and out/

SEED = 0
N_HIDDEN = (128, 64)


def main():
    d = np.load(ROOT + "/data/tep_clean.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    cols = list(d["cols"])

    Xtr, ytr, Xte, yte, split_record = episode_train_test_split(X, y, test_frac=0.3, seed=SEED)
    print(f"Train: {Xtr.shape}, Test: {Xte.shape}")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = MLPClassifier(hidden_layer_sizes=N_HIDDEN, activation="relu", solver="adam",
                         alpha=1e-4, max_iter=300, random_state=SEED, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=15)
    t0 = time.perf_counter()
    clf.fit(Xtr_s, ytr)
    train_time = time.perf_counter() - t0
    print(f"Training time: {train_time:.1f}s, iterations: {clf.n_iter_}")

    pred = clf.predict(Xte_s)
    acc = accuracy_score(yte, pred)
    macro_f1 = f1_score(yte, pred, average="macro")
    weighted_f1 = f1_score(yte, pred, average="weighted")

    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    report = classification_report(yte, pred, output_dict=True, zero_division=0)
    print("\nPer-class F1:")
    for label in sorted(FAULT_INFO.keys()):
        key = str(label)
        if key in report:
            print(f"  fault {label:2d} ({FAULT_INFO[label]['desc'][:40]:40s}): "
                  f"F1={report[key]['f1-score']:.3f}, n={int(report[key]['support'])}")

    cm = confusion_matrix(yte, pred, labels=sorted(FAULT_INFO.keys()))

    np.savez_compressed(ROOT + "/out/tep_model_artifacts.npz",
                         scaler_mean=scaler.mean_, scaler_scale=scaler.scale_,
                         confusion_matrix=cm, Xte_s=Xte_s, yte=yte, pred=pred)

    results = {
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_time_s": train_time, "n_iterations": int(clf.n_iter_),
        "accuracy": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1),
        "per_class_report": report,
        "split_record": {str(k): v for k, v in split_record.items()},
        "hidden_layers": list(N_HIDDEN), "seed": SEED,
    }
    with open(ROOT + "/out/tep_classifier_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import pickle
    with open(ROOT + "/out/tep_classifier.pkl", "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "cols": cols}, f)

    print("\nSaved tep_classifier_results.json, tep_model_artifacts.npz, tep_classifier.pkl")


if __name__ == "__main__":
    main()
