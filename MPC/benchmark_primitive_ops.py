import argparse
import json
import time
from statistics import mean, stdev

import jax
from flax import linen as nn
from jax import numpy as jnp
from spu.utils import distributed as ppd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--op")
parser.add_argument("--results")
parser.add_argument("--config", default="configs/3pc.json")
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument(
    "--vec-size",
    type=int,
    choices=(128, 256, 512),
    help="Vector size for distance ops (l1/l2/cos). Required for those ops.",
)
args = parser.parse_args()

if args.op in ("l1", "l2", "cos") and args.vec_size is None:
    parser.error("--vec-size is required when --op is one of l1, l2, cos")

with open(args.config, "r") as f:
    conf = json.load(f)
ppd.init(conf["nodes"], conf["devices"])


def benchmark_act(act_fn, name, rng):
    for in_shape in ((1, 8), (1, 16), (1, 32)):
        time_p = []
        time_s = []
        for _ in range(10):
            x = jax.random.normal(rng, in_shape)
            xs = ppd.device("P1")(lambda x: x)(x)
            ys = ppd.device("SPU")(act_fn)(xs)
            y = ppd.get(ys)
        for _ in range(args.num_epochs):
            x = jax.random.normal(rng, in_shape)
            start = time.time()
            y = act_fn(x)
            end = time.time()
            time_p.append(end - start)
            xs = ppd.device("P1")(lambda x: x)(x)
            start = time.time()
            ys = ppd.device("SPU")(act_fn)(xs)
            end = time.time()
            time_s.append(end - start)
            y = ppd.get(ys)
        print(
            f"Act: {name}, Shape: {in_shape}, Mean time SMPC: {mean(time_s):.4f}+-{stdev(time_s):.4f}, Mean time plain: {mean(time_p):.4f}+-{stdev(time_p):.4f}"
        )


def l1_dist(x1, x2):
    return jnp.sum(jnp.abs(x1 - x2))


def l2_dist(x1, x2):
    return jnp.linalg.norm(x1 - x2)


def cos_dist(x1, x2):
    cos_sim = jnp.dot(x1.ravel(), x2.ravel()) / (
        jnp.linalg.norm(x1) * jnp.linalg.norm(x2)
    )
    return 1.0 - cos_sim


def benchmark_dist(dist_fn, name, rng, size):
    in_shape = (1, size)
    time_p = []
    time_s = []
    for _ in range(10):
        x1 = jax.random.normal(rng, in_shape)
        x2 = jax.random.normal(rng, in_shape)
        x1s = ppd.device("P1")(lambda x: x)(x1)
        x2s = ppd.device("P2")(lambda x: x)(x2)
        ys = ppd.device("SPU")(dist_fn)(x1s, x2s)
        y = ppd.get(ys)
    for _ in range(args.num_epochs):
        x1 = jax.random.normal(rng, in_shape)
        x2 = jax.random.normal(rng, in_shape)
        start = time.time()
        y = dist_fn(x1, x2)
        end = time.time()
        time_p.append(end - start)
        x1s = ppd.device("P1")(lambda x: x)(x1)
        x2s = ppd.device("P2")(lambda x: x)(x2)
        start = time.time()
        ys = ppd.device("SPU")(dist_fn)(x1s, x2s)
        end = time.time()
        time_s.append(end - start)
        y = ppd.get(ys)
    print(
        f"Dist: {name}, Size: {size}, Mean time SMPC: {mean(time_s):.4f}+-{stdev(time_s):.4f}, Mean time plain: {mean(time_p):.4f}+-{stdev(time_p):.4f}"
    )


def main():
    rng = jax.random.PRNGKey(42)
    if args.op == "matmul":
        for in_shape in ((1, 10, 10), (1, 50, 50), (1, 100, 100)):
            time_p = []
            time_s = []
            for _ in range(10):
                x1 = jax.random.normal(rng, in_shape)
                x2 = jax.random.normal(rng, in_shape)
                x1s = ppd.device("P1")(lambda x: x)(x1)
                x2s = ppd.device("P2")(lambda x: x)(x2)
                ys = ppd.device("SPU")(jnp.matmul)(x1s, x2s)
                y = ppd.get(ys)
            for _ in range(args.num_epochs):
                x1 = jax.random.normal(rng, in_shape)
                x2 = jax.random.normal(rng, in_shape)
                start = time.time()
                y = jnp.matmul(x1, x2)
                end = time.time()
                time_p.append(end - start)
                x1s = ppd.device("P1")(lambda x: x)(x1)
                x2s = ppd.device("P2")(lambda x: x)(x2)
                start = time.time()
                ys = ppd.device("SPU")(jnp.matmul)(x1s, x2s)
                end = time.time()
                time_s.append(end - start)
                y = ppd.get(ys)
            print(
                f"Shape: {in_shape}, Mean time SMPC: {mean(time_s):.4f}+-{stdev(time_s):.4f}, Mean time plain: {mean(time_p):.4f}+-{stdev(time_p):.4f}"
            )
    elif args.op == "relu":
        benchmark_act(nn.relu, "ReLU", rng)
    elif args.op == "sigmoid":
        benchmark_act(nn.sigmoid, "Sigmoid", rng)
    elif args.op == "gelu":
        benchmark_act(nn.gelu, "GeLU", rng)
    elif args.op == "l1":
        benchmark_dist(l1_dist, "L1", rng, args.vec_size)
    elif args.op == "l2":
        benchmark_dist(l2_dist, "L2", rng, args.vec_size)
    elif args.op == "cos":
        benchmark_dist(cos_dist, "Cosine", rng, args.vec_size)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
