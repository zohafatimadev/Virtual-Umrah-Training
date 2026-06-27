"""
lightweight_models.py
=====================
Dr. Shoaib's point: a Transformer is heavy/costly for this relatively small
task. This script compares LIGHTER models for the same rule-free error-detection
task (reading action sequences from the database), so we can show empirically
whether a cheaper model matches the Transformer.

Models compared (all read sequences from the SQLite database):
  1. Bag-of-Actions + Logistic Regression  (ultra-light, no deep learning)
  2. Bag-of-Actions + Random Forest         (light, no deep learning)
  3. 1D-CNN  (lightweight deep model)        [needs torch]
  4. GRU     (lighter recurrent than LSTM)   [needs torch]

The Transformer result (from train_transformer_db.py) is the heavy reference.

Run:  python lightweight_models.py
"""
import sqlite3
import json
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time

from action_logger import DB
MAX_LEN = 64


def load_runs():
    con = sqlite3.connect(DB)
    vocab = sorted(r[0] for r in con.execute('SELECT DISTINCT action FROM actions'))
    vmap = {a: i for i, a in enumerate(vocab)}
    runs = con.execute("SELECT run_id, label FROM runs WHERE label IS NOT NULL").fetchall()
    seqs, y = [], []
    for rid, label in runs:
        s = [r[0] for r in con.execute(
            'SELECT action FROM actions WHERE run_id=? ORDER BY step', (rid,)).fetchall()]
        seqs.append(s); y.append(1 if label == 'had_errors' else 0)
    con.close()
    json.dump(vmap, open('action_vocab.json', 'w'))
    return seqs, np.array(y), vmap


def bag_of_actions(seqs, vmap):
    """Count how many times each action appears -> fixed-length vector.
    This is the 'cheap' representation: no order, just counts."""
    X = np.zeros((len(seqs), len(vmap)))
    for i, s in enumerate(seqs):
        for a in s:
            if a in vmap:
                X[i, vmap[a]] += 1
    return X


def report(name, yt, yp, train_time):
    acc = accuracy_score(yt, yp)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average='binary', zero_division=0)
    print(f'{name:30s} acc={acc:.3f} prec={p:.3f} rec={r:.3f} f1={f1:.3f}  train={train_time:.2f}s')
    return [name, round(acc, 4), round(p, 4), round(r, 4), round(f1, 4), round(train_time, 3)]


def main():
    seqs, y, vmap = load_runs()
    print(f'Loaded {len(seqs)} runs, {len(vmap)} action types\n')
    Xbag = bag_of_actions(seqs, vmap)
    Xtr, Xte, ytr, yte = train_test_split(Xbag, y, test_size=0.25, random_state=42, stratify=y)

    rows = []

    # 1. Logistic Regression (ultra-light)
    t = time.time()
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    rows.append(report('LogReg (bag-of-actions)', yte, lr.predict(Xte), time.time() - t))

    # 2. Random Forest (light)
    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    rows.append(report('RandomForest (bag-of-actions)', yte, rf.predict(Xte), time.time() - t))

    # 3 + 4: deep light models (torch) — 1D-CNN and GRU on the sequence
    try:
        import torch
        import torch.nn as nn

        # build padded id sequences
        def to_ids(seqs):
            out = []
            for s in seqs:
                ids = [vmap[a] + 1 for a in s if a in vmap][:MAX_LEN]
                ids += [0] * (MAX_LEN - len(ids))
                out.append(ids)
            return np.array(out)
        Xids = to_ids(seqs)
        Xtr_i, Xte_i, ytr_i, yte_i = train_test_split(Xids, y, test_size=0.25,
                                                      random_state=42, stratify=y)
        Xtr_t = torch.tensor(Xtr_i); ytr_t = torch.tensor(ytr_i, dtype=torch.float32)
        Xte_t = torch.tensor(Xte_i)
        V = len(vmap) + 1

        # 3. 1D-CNN
        class CNN(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(V, 32, padding_idx=0)
                s.conv = nn.Conv1d(32, 64, 3, padding=1)
                s.fc = nn.Linear(64, 1)
            def forward(s, x):
                h = s.emb(x).transpose(1, 2)
                h = torch.relu(s.conv(h)).max(dim=2).values
                return torch.sigmoid(s.fc(h)).squeeze(-1)

        t = time.time()
        cnn = CNN(); opt = torch.optim.Adam(cnn.parameters(), 1e-3); lf = nn.BCELoss()
        for _ in range(20):
            opt.zero_grad(); loss = lf(cnn(Xtr_t), ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (cnn(Xte_t).numpy() > 0.5).astype(int)
        rows.append(report('1D-CNN (light deep)', yte_i, pred, time.time() - t))

        # 4. GRU (lighter than LSTM)
        class GRUNet(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(V, 32, padding_idx=0)
                s.gru = nn.GRU(32, 32, batch_first=True)
                s.fc = nn.Linear(32, 1)
            def forward(s, x):
                h = s.emb(x); o, _ = s.gru(h)
                return torch.sigmoid(s.fc(o[:, -1, :])).squeeze(-1)

        t = time.time()
        gru = GRUNet(); opt = torch.optim.Adam(gru.parameters(), 1e-3)
        for _ in range(20):
            opt.zero_grad(); loss = lf(gru(Xtr_t), ytr_t); loss.backward(); opt.step()
        with torch.no_grad():
            pred = (gru(Xte_t).numpy() > 0.5).astype(int)
        rows.append(report('GRU (light deep)', yte_i, pred, time.time() - t))
    except Exception as e:
        print('\n(torch not available here — 1D-CNN and GRU will run on your machine)')

    # save
    import csv
    with open('lightweight_results.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'TrainTime_s'])
        w.writerows(rows)
    print('\nSaved lightweight_results.csv')


if __name__ == '__main__':
    main()
