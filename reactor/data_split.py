"""Episode-level train/test split of a continuous labelled time series.

Adjacent samples of one episode are near-duplicates, so whole episodes (maximal runs of one label)
are assigned to training or test. Imported by the analysis scripts."""
import numpy as np


def find_episodes(y):
    """Return list of (start_idx, end_idx_exclusive, label) for each contiguous run."""
    episodes = []
    start = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            episodes.append((start, i, int(y[start])))
            start = i
    return episodes


def episode_train_test_split(X, y, test_frac=0.3, seed=0, min_episode_len=5):
    """
    Split by episode, stratified per fault label so both train and test
    contain examples of every class, using entire episodes on one side
    only. Episodes shorter than min_episode_len are dropped (too few
    points to be a meaningful within-episode training signal and prone to
    edge-transition contamination).
    """
    rng = np.random.default_rng(seed)
    episodes = find_episodes(y)
    episodes = [e for e in episodes if (e[1] - e[0]) >= min_episode_len]

    by_label = {}
    for ep in episodes:
        by_label.setdefault(ep[2], []).append(ep)

    train_idx, test_idx = [], []
    episode_split_record = {}
    for label, eps in by_label.items():
        eps = eps.copy()
        rng.shuffle(eps)
        n_test = max(1, int(round(len(eps) * test_frac))) if len(eps) > 1 else 0
        test_eps = eps[:n_test]
        train_eps = eps[n_test:]
        episode_split_record[label] = {"n_train_episodes": len(train_eps), "n_test_episodes": len(test_eps)}
        for s, e, _ in train_eps:
            train_idx.extend(range(s, e))
        for s, e, _ in test_eps:
            test_idx.extend(range(s, e))

    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], episode_split_record


if __name__ == "__main__":
    d = np.load("../data/reactor.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    Xtr, ytr, Xte, yte, record = episode_train_test_split(X, y, test_frac=0.3, seed=0)
    print(f"Total rows: {len(y)}, train: {len(ytr)}, test: {len(yte)}")
    print("\nPer-class episode split:")
    for label in sorted(record.keys()):
        r = record[label]
        n_train_rows = (ytr == label).sum()
        n_test_rows = (yte == label).sum()
        print(f"  fault {label:2d}: {r['n_train_episodes']:3d} train episodes ({n_train_rows:5d} rows), "
              f"{r['n_test_episodes']:3d} test episodes ({n_test_rows:5d} rows)")
