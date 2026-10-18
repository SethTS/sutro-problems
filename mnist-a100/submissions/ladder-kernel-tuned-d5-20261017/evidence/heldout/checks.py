"""Data/protocol checks before looking at any model's held-out predictions."""
import gzip
import struct
import numpy as np
from evaluate import read_idx, pixels, query_ids, release


img = np.arange(2 * 784, dtype=np.uint16).astype(np.uint8).reshape(2, 28, 28)
decoded = read_idx(gzip.compress(struct.pack('>IIII', 2051, 2, 28, 28) + img.tobytes()), True)
assert np.array_equal(img, decoded)
lab = read_idx(gzip.compress(struct.pack('>II', 2049, 2) + bytes([2, 8])), False)
assert np.array_equal(lab, [2, 8])
try:
    read_idx(gzip.compress(struct.pack('>II', 2049, 3) + bytes([2, 8])), False)
except AssertionError:
    pass
else:
    raise AssertionError('Truncated IDX accepted')
assert pixels(img).shape == (2, 9, 9) and pixels(img).dtype == np.float32
np.testing.assert_allclose(pixels(np.full((1, 28, 28), 255, np.uint8)), np.ones((1, 9, 9), np.float32), atol=2e-7, rtol=0)
ids = np.concatenate(query_ids())
assert np.array_equal(np.sort(ids), np.arange(10000, 60000))
# Verify release fits only the selected training rows and applies one shared transform.
rng = np.random.default_rng(45)
train = rng.normal(size=(10001, 9, 9)).astype(np.float32)
target = np.arange(len(train), dtype=np.uint8) % 10
query = rng.normal(size=(10000, 9, 9)).astype(np.float32)
query_labels = np.arange(len(query), dtype=np.uint8) % 10
d, ti = release(train, target, query, query_labels, 912, 'test')
assert len(np.unique(ti)) == 10000
assert d['train_x'].shape == d['test_x'].shape == (10000, 60)
assert np.max(np.abs(d['train_x'].mean(0))) < 1e-5
assert np.max(np.abs(np.cov(d['train_x'], rowvar=False) - np.eye(60))) < 1e-5
q2 = query.copy()
q2[0] += 3
d2, ti2 = release(train, target, q2, query_labels[::-1], 912, 'test')
assert np.array_equal(ti, ti2) and np.array_equal(d['train_x'], d2['train_x'])
assert np.array_equal(d['test_x'][1:], d2['test_x'][1:])
print('PASS: IDX parsing, preprocessing, disjoint QMNIST cohorts, and train-only whitening/release.')
