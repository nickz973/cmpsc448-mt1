"""Prepare TXD-22 for leakage-safe CMPSC 448 LLM attribution experiments.

Usage:
    python src/prepare_data.py --input '/path/to/TXD-22 A Large-Scale Benchmark Dataset.csv'
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

LABELS = ('ChatGPT', 'Claude', 'Gemini')
SEED = 448


def prepare(input_csv: Path, output_dir: Path) -> None:
    # The supplied TXD-22 CSV contains a mixture of non-UTF-8 bytes.
    raw = pd.read_csv(input_csv, encoding='latin1', low_memory=False)
    required = {'Question', 'Answer', 'Source'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Expected columns {required}, found {list(raw.columns)}')

    df = raw.loc[raw['Source'].isin(LABELS), ['Source', 'Question', 'Answer']].copy()
    df.columns = ['LLM_name', 'LLM_Input', 'LLM_output']
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df.loc[(df != '').all(axis=1)].drop_duplicates().reset_index(drop=True)
    if set(df.LLM_name.unique()) != set(LABELS):
        raise ValueError('Missing at least one required LLM family')

    # Shuffle *unique question groups*, never individual rows: prevents the
    # same question across multiple LLMs from leaking into another split.
    split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    dev_idx, held_idx = next(split.split(df, groups=df['LLM_Input']))
    dev = df.iloc[dev_idx].reset_index(drop=True)
    held = df.iloc[held_idx].reset_index(drop=True)
    second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_idx, test_idx = next(second.split(held, groups=held['LLM_Input']))
    sets = {
        'train': dev,
        'val': held.iloc[val_idx].reset_index(drop=True),
        'test': held.iloc[test_idx].reset_index(drop=True),
    }
    questions = [set(sub.LLM_Input) for sub in sets.values()]
    assert not (questions[0] & questions[1] or questions[0] & questions[2] or questions[1] & questions[2])

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in sets.items():
        subset.to_csv(output_dir / f'{name}.csv', index=False, encoding='utf-8')
    df.to_csv(output_dir / 'filtered_all.csv', index=False, encoding='utf-8')

    report = {
        'source_filename': input_csv.name,
        'source_rows': len(raw),
        'filtered_rows': len(df),
        'seed': SEED,
        'grouping_key': 'LLM_Input (entire stripped prompt)',
        'dataset_counts': {label: int((df.LLM_name == label).sum()) for label in LABELS},
        'unique_prompts': int(df.LLM_Input.nunique()),
        'split_counts': {
            name: {'total': len(sub), 'unique_prompts': int(sub.LLM_Input.nunique()),
                   'by_model': {label: int((sub.LLM_name == label).sum()) for label in LABELS}}
            for name, sub in sets.items()
        },
    }
    (output_dir / 'dataset_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    prepare(args.input, args.output)
