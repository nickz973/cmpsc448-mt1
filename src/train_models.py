"""Train CNN/LSTM LLM-origin classifiers for CMPSC 448 RQ1 and RQ2.

Run from repository root: python src/train_models.py --model both --epochs 5
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = ['ChatGPT', 'Claude', 'Gemini']
PAD, UNK = '<pad>', '<unk>'
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text):
    return TOKEN_RE.findall(str(text).lower())


def get_text(frame, mode):
    if mode == 'input':
        return frame['LLM_Input'].astype(str).tolist()
    if mode == 'output':
        return frame['LLM_output'].astype(str).tolist()
    return ('[PROMPT] ' + frame['LLM_Input'].astype(str) + ' [RESPONSE] ' + frame['LLM_output'].astype(str)).tolist()


def build_vocab(texts, max_vocab, min_freq):
    counts = Counter(token for text in texts for token in tokenize(text))
    frequent = sorted(((t, c) for t, c in counts.items() if c >= min_freq), key=lambda tc: (-tc[1], tc[0]))
    vocab = {PAD: 0, UNK: 1}
    for token, _ in frequent[:max_vocab - 2]:
        vocab[token] = len(vocab)
    return vocab


class TextDataset(Dataset):
    def __init__(self, frame, mode, vocab, max_len):
        self.labels = [LABELS.index(x) for x in frame['LLM_name']]
        self.tokens = []
        self.lengths = []
        for text in get_text(frame, mode):
            ids = [vocab.get(t, 1) for t in tokenize(text)[:max_len]]
            ids = ids or [1]
            self.lengths.append(len(ids))
            self.tokens.append(ids + [0] * (max_len - len(ids)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.tokens[idx], dtype=torch.long),
                torch.tensor(self.lengths[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=96, hidden_dim=96, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, hidden_dim, k) for k in (3, 4, 5)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 3, len(LABELS))

    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(1, 2)
        features = [torch.relu(conv(emb)).amax(dim=2) for conv in self.convs]
        return self.head(self.dropout(torch.cat(features, dim=1)))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=96, hidden_dim=96, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim * 2, len(LABELS))

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # Packing excludes padding and allows correct last hidden states.
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.head(self.dropout(features))


def choose_device(requested):
    if requested != 'auto':
        if requested == 'mps' and not torch.backends.mps.is_available():
            raise ValueError('MPS is not available; try --device cpu')
        if requested == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is not available; try --device cpu')
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def evaluate(model, loader, device, criterion):
    model.eval()
    truth, pred = [], []
    losses = []
    with torch.no_grad():
        for x, lengths, labels in loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x, lengths)
            losses.append(criterion(logits, labels).item() * len(labels))
            truth.extend(labels.cpu().tolist())
            pred.extend(logits.argmax(1).cpu().tolist())
    n = len(truth)
    return {'loss': sum(losses) / n, 'accuracy': accuracy_score(truth, pred),
            'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
            'classification_report': classification_report(truth, pred, labels=[0, 1, 2],
                                                           target_names=LABELS, output_dict=True, zero_division=0)}


def run_experiment(model_name, mode, splits, args, device):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    vocab = build_vocab(get_text(splits['train'], mode), args.max_vocab, args.min_freq)
    datasets = {k: TextDataset(df, mode, vocab, args.max_len) for k, df in splits.items()}
    loaders = {k: DataLoader(dataset, batch_size=args.batch_size, shuffle=(k == 'train'), num_workers=0)
               for k, dataset in datasets.items()}
    cls = CNNClassifier if model_name == 'cnn' else LSTMClassifier
    model = cls(len(vocab), args.embed_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state, history = -1, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, lengths, labels in loaders['train']:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, lengths), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        validation = evaluate(model, loaders['val'], device, criterion)
        record = {'epoch': epoch, 'train_loss': total_loss / len(datasets['train']),
                  'val_loss': validation['loss'], 'val_accuracy': validation['accuracy'],
                  'val_macro_f1': validation['macro_f1']}
        history.append(record)
        print(f'  Epoch {epoch}/{args.epochs}: train loss {record["train_loss"]:.4f}, val acc {validation["accuracy"]:.3f}, val macro F1 {validation["macro_f1"]:.3f}', flush=True)
        if validation['accuracy'] > best_acc:
            best_acc = validation['accuracy']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    results = {'model': model_name, 'input_mode': mode, 'best_val_accuracy': best_acc,
               'best_epoch': int(max(history, key=lambda h: h['val_accuracy'])['epoch']),
               'vocabulary_size': len(vocab), 'history': history,
               'test': evaluate(model, loaders['test'], device, criterion)}
    args.out.mkdir(parents=True, exist_ok=True)
    prefix = f'{model_name}_{mode}'
    with open(args.out / f'{prefix}_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    torch.save({'model_state_dict': best_state, 'vocab': vocab, 'labels': LABELS,
                'model': model_name, 'input_mode': mode, 'max_len': args.max_len,
                'embed_dim': args.embed_dim, 'hidden_dim': args.hidden_dim,
                'dropout': args.dropout}, args.out / f'{prefix}_checkpoint.pt')
    print(f'  TEST accuracy: {results["test"]["accuracy"]:.3f}, macro F1: {results["test"]["macro_f1"]:.3f}', flush=True)
    return {'model': model_name, 'input_mode': mode, 'test_accuracy': results['test']['accuracy'],
            'test_macro_f1': results['test']['macro_f1'], 'best_epoch': results['best_epoch']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--model', choices=['cnn', 'lstm', 'both'], default='both')
    parser.add_argument('--mode', choices=['input', 'output', 'combined', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=160)
    parser.add_argument('--max-vocab', type=int, default=12000)
    parser.add_argument('--min-freq', type=int, default=2)
    parser.add_argument('--embed-dim', type=int, default=96)
    parser.add_argument('--hidden-dim', type=int, default=96)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=448)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--limit', type=int, default=0, help='Debug only: first N rows per split, not for final results')
    args = parser.parse_args()
    if args.max_len < 5:
        parser.error('--max-len must be at least 5 for CNN kernel size 5')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    splits = {s: pd.read_csv(args.data / f'{s}.csv').dropna(subset=['LLM_Input', 'LLM_output', 'LLM_name'])
              for s in ('train', 'val', 'test')}
    if args.limit:
        splits = {s: d.groupby('LLM_name', group_keys=False).head(args.limit) for s, d in splits.items()}
        print('DEBUG MODE: results from --limit are not final research results')
    if set(splits['train']['LLM_name']) != set(LABELS):
        raise ValueError('Training data must have ChatGPT, Claude, and Gemini labels')
    print('Device:', device, '| rows:', {k: len(v) for k, v in splits.items()}, flush=True)
    models = ['cnn', 'lstm'] if args.model == 'both' else [args.model]
    modes = ['input', 'output', 'combined'] if args.mode == 'all' else [args.mode]
    summary = []
    for model in models:
        for mode in modes:
            print(f'\n=== {model.upper()}: {mode} ===', flush=True)
            summary.append(run_experiment(model, mode, splits, args, device))
            pd.DataFrame(summary).to_csv(args.out / 'summary.csv', index=False)
    print('\nSaved metrics, checkpoints, and summary.csv to', args.out)


if __name__ == '__main__':
    main()
