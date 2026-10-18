"""Frozen external-test evaluation through the canonical sandbox worker."""
import argparse
import dataclasses
import gzip
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import mnist


def read_idx(data, images):
    raw = gzip.decompress(data)
    magic, n = struct.unpack('>II', raw[:8])
    assert magic == (2051 if images else 2049)
    if images:
        rows, cols = struct.unpack('>II', raw[8:16])
        assert (rows, cols) == (28, 28) and len(raw) == 16 + n * 784
        return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 28, 28)
    assert len(raw) == 8 + n
    out = np.frombuffer(raw, dtype=np.uint8, offset=8)
    assert out.max() <= 9
    return out


def pixels(images):
    weights = mnist.area_weights(28, 9)
    out = np.matmul(np.matmul(weights, images.astype(np.float32) / np.float32(255)), weights.T)
    np.clip(out, 0, 1, out=out)
    return np.ascontiguousarray(out, dtype=np.float32)


def query_ids():
    # The first 10k QMNIST examples reconstruct MNIST test; exclude them entirely.
    ids = np.random.default_rng(627000).permutation(np.arange(10000, 60000))
    assert len(np.unique(ids)) == 50000 and ids.min() >= 10000
    return [ids[i:i + 10000] for i in range(0, 50000, 10000)]


def release(train_pixels, train_labels, query_pixels, query_labels, seed, dataset):
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(train_pixels), 10000, replace=False)
    relabel = rng.permutation(10).astype(np.int64)
    mu, whitener = mnist.fit_whitener(train_pixels[ids])
    transform = mnist.haar_rotation(60, rng) @ whitener
    draw = dict(dataset=dataset, train_x=mnist.apply_release(train_pixels[ids], mu, transform),
                train_y=relabel[train_labels[ids]], test_x=mnist.apply_release(query_pixels, mu, transform),
                test_y=relabel[query_labels])
    return draw, ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('function')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    data_dir = Path(__file__).resolve().parent / 'data'
    provenance = json.loads((data_dir.parent / 'data-manifest.json').read_text())
    for row in provenance['files']:
        assert hashlib.sha256((data_dir / row['file']).read_bytes()).hexdigest() == row['sha256']
    train_x, train_y = mnist.load_pool('mnist')
    standard_x = pixels(read_idx((data_dir / 'mnist-test-images.gz').read_bytes(), True))
    standard_y = read_idx((data_dir / 'mnist-test-labels.gz').read_bytes(), False)
    extra_x = pixels(read_idx((data_dir / 'qmnist-test-images-idx3-ubyte.gz').read_bytes(), True))
    extra_y = read_idx((data_dir / 'qmnist-test-labels-idx1-ubyte.gz').read_bytes(), False)
    assert len(standard_x) == len(standard_y) == 10000 and len(extra_x) == len(extra_y) == 60000
    source = a.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    mnist.check_source(source, a.function)
    reason = mnist.sandbox_available()
    assert reason is None, reason
    mnist.not_dumpable()
    records = []
    cases = [('mnist-test', 627101 + i, np.arange(10000)) for i in range(3)]
    cases += [('qmnist-extra', 627201 + i, ids) for i, ids in enumerate(query_ids())]
    for dataset, seed, qids in cases:
        xp, yp = (standard_x, standard_y) if dataset == 'mnist-test' else (extra_x, extra_y)
        draw, train_ids = release(train_x, train_y, xp[qids], yp[qids], seed, dataset)
        wx, wy, wq, wt = mnist.draw('fashion', seed=seed + 1000000)
        warm = dict(dataset='fashion', train_x=wx, train_y=wy, test_x=wq, test_y=wt)
        hello = {}
        call = mnist.timed_call(source, a.function, True, warm, draw, announce=hello.update)
        row = dict(**dataclasses.asdict(call), seed=seed, error_pct=100 * (call.total - call.correct) / call.total,
                   source_sha256=digest, sandboxed=True, device=hello.get('device'), torch=hello.get('torch'),
                   training_indices_sha256=hashlib.sha256(train_ids.astype('<i8').tobytes()).hexdigest(),
                   query_indices_sha256=hashlib.sha256(qids.astype('<i8').tobytes()).hexdigest(),
                   release_sha256=hashlib.sha256(draw['train_x'].tobytes() + draw['train_y'].tobytes() + draw['test_x'].tobytes()).hexdigest())
        records.append(row)
        a.output.write_text(json.dumps(records, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
