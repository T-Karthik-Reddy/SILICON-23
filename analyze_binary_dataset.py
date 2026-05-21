


   

import os
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(BASE_DIR, 'processed_dataset')


def _collect_npy(cls):
    cls_dir = os.path.join(PROCESSED_PATH, cls)
    if not os.path.exists(cls_dir):
        return []
    return [f for f in os.listdir(cls_dir) if f.endswith('.npy')]


def main():
    pos = _collect_npy('dog_bark')
    neg = _collect_npy('not_dog_bark')

    print('\n' + '=' * 64)
    print('SILICON-23 BINARY DATASET ANALYSIS')
    print('=' * 64)
    print(f"dog_bark samples    : {len(pos)}")
    print(f"not_dog_bark samples: {len(neg)}")

    src_counts = Counter()
    for fname in neg:
        if '_baby_cry_' in fname:
            src_counts['baby_cry'] += 1
        elif '_unknown_' in fname:
            src_counts['unknown'] += 1
        else:
            src_counts['other'] += 1

    total_neg = max(1, len(neg))
    print('\nNegative composition:')
    for src in ('baby_cry', 'unknown', 'other'):
        c = src_counts[src]
        print(f"- {src:<10}: {c:5d} ({100.0*c/total_neg:5.1f}%)")

    if len(pos) and len(neg):
        ratio = len(neg) / len(pos)
        print(f"\nNeg/Pos ratio: {ratio:.2f}")

    print('=' * 64 + '\n')


if __name__ == '__main__':
    main()
