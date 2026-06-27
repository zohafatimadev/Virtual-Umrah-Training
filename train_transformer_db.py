"""
train_transformer_db.py
=======================
Trains a TRANSFORMER to detect Umrah ritual errors by reading ORDERED ACTION
SEQUENCES from the SQLite database. NO RULES are used here — the Transformer
is the sole error detector.

Pipeline (rule-free detection):
    SQLite database  ->  action sequences  ->  Transformer  ->  correct/had_errors

The labels used for training came from the one-time weak-supervision step in
populate_database.py; the rules themselves are not present in this file or in
the deployed model.

Run:
    python populate_database.py      # build the database first
    python train_transformer_db.py
Requires: torch
"""
import json
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from action_logger import DB

MAX_LEN = 64


def build_vocab(con):
    cur = con.execute('SELECT DISTINCT action FROM actions')
    actions = sorted(r[0] for r in cur.fetchall())
    return {a: i + 1 for i, a in enumerate(actions)}  # 0 = padding


def load_from_db():
    con = sqlite3.connect(DB)
    vocab = build_vocab(con)
    runs = con.execute("SELECT run_id, label FROM runs WHERE label IS NOT NULL").fetchall()
    X, y = [], []
    for run_id, label in runs:
        seq = [r[0] for r in con.execute(
            'SELECT action FROM actions WHERE run_id=? ORDER BY step', (run_id,)).fetchall()]
        ids = [vocab[a] for a in seq][:MAX_LEN]
        ids = ids + [0] * (MAX_LEN - len(ids))
        X.append(ids)
        y.append(1 if label == 'had_errors' else 0)
    con.close()
    json.dump(vocab, open('action_vocab.json', 'w'), indent=2)
    return np.array(X), np.array(y), len(vocab) + 1


def main():
    X, y, vocab_size = load_from_db()
    print(f'Loaded {len(X)} runs from database, vocab size {vocab_size}')
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    try:
        import torch
        import torch.nn as nn
    except Exception:
        print('PyTorch not installed. Run: pip install torch')
        return

    Xtr_t = torch.tensor(Xtr, dtype=torch.long); ytr_t = torch.tensor(ytr, dtype=torch.float32)
    Xte_t = torch.tensor(Xte, dtype=torch.long)

    class PosEnc(nn.Module):
        def __init__(self, d, mx=MAX_LEN):
            super().__init__()
            pe = torch.zeros(mx, d); pos = torch.arange(0, mx).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer('pe', pe.unsqueeze(0))
        def forward(self, x): return x + self.pe[:, :x.size(1)]

    class Transformer(nn.Module):
        def __init__(self, vocab, d=64, heads=4, layers=2):
            super().__init__()
            self.emb = nn.Embedding(vocab, d, padding_idx=0)
            self.pos = PosEnc(d)
            enc = nn.TransformerEncoderLayer(d, heads, 128, 0.2, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.cls = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        def forward(self, x):
            mask = (x == 0)
            h = self.pos(self.emb(x))
            h = self.tr(h, src_key_padding_mask=mask)
            valid = (~mask).unsqueeze(-1).float()
            pooled = (h * valid).sum(1) / valid.sum(1).clamp(min=1)
            return torch.sigmoid(self.cls(pooled)).squeeze(-1)

    model = Transformer(vocab_size)
    opt = torch.optim.Adam(model.parameters(), 1e-3); lossf = nn.BCELoss()
    print('Training Transformer (rule-free)...')
    n = len(Xtr_t); bs = 64
    for ep in range(30):
        model.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]; opt.zero_grad()
            loss = lossf(model(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
        if (ep + 1) % 10 == 0:
            print(f'  epoch {ep+1}/30 loss={loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        pred = (model(Xte_t).numpy() > 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
    print('\n===== TRANSFORMER (rule-free error detection) =====')
    print(f'Accuracy={acc:.3f}  Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}')
    torch.save(model.state_dict(), 'transformer_db.pt')
    print('Saved transformer_db.pt')


if __name__ == '__main__':
    main()
