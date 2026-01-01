import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download, hf_hub_url


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            parts = []
            for c in cols:
                v = r.get(c, "")
                s = str(v)
                s = s.replace("\n", " ").replace("\r", " ").replace(",", " ")
                parts.append(s)
            f.write(",".join(parts) + "\n")


def hf_download(
    repo_id: str, filename: str, revision: str, token: Optional[str]
) -> str:
    return hf_hub_download(
        repo_id=repo_id, filename=filename, revision=revision, token=token
    )


def load_config(repo_id: str, revision: str, token: Optional[str]) -> Dict[str, Any]:
    path = hf_download(repo_id, "config.json", revision, token)
    return read_json(path)


def load_index_json(
    repo_id: str, revision: str, token: Optional[str]
) -> Dict[str, Any]:
    path = hf_download(repo_id, "model.safetensors.index.json", revision, token)
    return read_json(path)


def http_range_get(
    url: str, start: int, end: int, token: Optional[str], retries: int = 5
) -> bytes:
    headers = {"Range": f"bytes={start}-{end}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=180, allow_redirects=True)
            if r.status_code in (200, 206):
                return r.content
            last_err = RuntimeError(
                f"HTTP {r.status_code} for {url} range {start}-{end}"
            )
        except Exception as e:
            last_err = e
        time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Range GET failed after retries: {last_err}")


def fetch_safetensors_header(
    repo_id: str,
    filename: str,
    revision: str,
    token: Optional[str],
    max_header_bytes: int,
) -> Dict[str, Any]:
    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    raw8 = http_range_get(url, 0, 7, token)
    header_len = int.from_bytes(raw8, "little")
    if header_len > max_header_bytes:
        raise RuntimeError(f"header too large: {header_len} > {max_header_bytes}")
    hb = http_range_get(url, 8, 8 + header_len - 1, token)
    header = json.loads(hb.decode("utf-8").rstrip())
    header["__header_len__"] = header_len
    header["__url__"] = url
    return header


def tensor_nbytes(info: Dict[str, Any]) -> int:
    a, b = info["data_offsets"]
    return int(b) - int(a)


def decode_tensor(raw: bytes, dtype: str, shape: List[int]) -> np.ndarray:
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32)
    elif dtype == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    elif dtype == "I32":
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    elif dtype == "I64":
        arr = np.frombuffer(raw, dtype=np.int64).astype(np.float32)
    elif dtype == "U8":
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    else:
        raise RuntimeError(f"Unsupported dtype: {dtype}")

    size = int(np.prod(shape)) if shape else 1
    if arr.size != size:
        arr = arr[:size]
    return arr.reshape(shape)


def fetch_tensor(key: str, header: Dict[str, Any], token: Optional[str]) -> np.ndarray:
    info = header[key]
    dtype = info["dtype"]
    shape = info["shape"]
    off0, off1 = info["data_offsets"]
    header_len = int(header["__header_len__"])
    url = header["__url__"]
    begin = 8 + header_len + int(off0)
    end_incl = 8 + header_len + int(off1) - 1
    raw = http_range_get(url, begin, end_incl, token)
    return decode_tensor(raw, dtype, shape)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def shuffled_cosine(a: np.ndarray, b: np.ndarray, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    a = a.reshape(-1)
    b = b.reshape(-1)
    a_shuffled = rng.permutation(a)
    b_shuffled = rng.permutation(b)
    na = np.linalg.norm(a_shuffled)
    nb = np.linalg.norm(b_shuffled)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a_shuffled, b_shuffled) / (na * nb))


LAYER_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")


@dataclass
class TensorKeyRef:
    key: str
    shard: str


def get_layer_index(key: str) -> Optional[int]:
    m = LAYER_RE.search(key)
    if not m:
        return None
    return int(m.group(1))


def classify_key(key: str) -> Optional[str]:
    lk = key.lower()
    if not lk.endswith(".weight"):
        return None

    if "norm" in lk or "layernorm" in lk or "rms" in lk:
        if any(
            x in lk
            for x in [
                "input",
                "pre",
                "attn_norm",
                "attention_norm",
                "ln_1",
                "norm1",
                "layernorm1",
            ]
        ):
            return "norm_pre"
        if any(x in lk for x in ["post", "ffn", "mlp", "ln_2", "norm2", "layernorm2"]):
            return "norm_post"
        return "norm_any"

    return None


def build_layer_keyrefs(
    weight_map: Dict[str, str], max_layers: int
) -> Dict[int, Dict[str, TensorKeyRef]]:
    per_layer = defaultdict(lambda: defaultdict(list))
    for k, shard in weight_map.items():
        li = get_layer_index(k)
        if li is None or li >= max_layers:
            continue
        cat = classify_key(k)
        if cat is None:
            continue
        per_layer[li][cat].append(TensorKeyRef(key=k, shard=shard))

    out: Dict[int, Dict[str, TensorKeyRef]] = {}
    for li, cats in per_layer.items():
        chosen: Dict[str, TensorKeyRef] = {}
        for cat, lst in cats.items():
            lst_sorted = sorted(lst, key=lambda x: x.key)
            chosen[cat] = lst_sorted[0]

        if "norm_pre" not in chosen and "norm_any" in chosen:
            chosen["norm_pre"] = chosen["norm_any"]
        if "norm_post" not in chosen and "norm_any" in chosen:
            chosen["norm_post"] = chosen["norm_any"]

        out[li] = chosen

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="upstage/Solar-Open-100B", help="model A repo_id")
    ap.add_argument("--b", default="zai-org/GLM-4.5-Air", help="model B repo_id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--outdir", default="out_sanity")
    ap.add_argument("--max-tensor-bytes", type=int, default=2_500_000)
    ap.add_argument("--max-header-mb", type=int, default=64)
    ap.add_argument("--max-layers", type=int, default=-1)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN", None)
    ensure_outdir(args.outdir)
    max_header_bytes = args.max_header_mb * 1024 * 1024

    cfgA = load_config(args.a, args.revision, token)
    cfgB = load_config(args.b, args.revision, token)

    idxA = load_index_json(args.a, args.revision, token)
    idxB = load_index_json(args.b, args.revision, token)

    wmA = idxA.get("weight_map", {})
    wmB = idxB.get("weight_map", {})

    layersA = int(cfgA.get("num_hidden_layers", 0))
    layersB = int(cfgB.get("num_hidden_layers", 0))

    probe_layers = min(layersA, layersB)
    if args.max_layers > 0:
        probe_layers = min(probe_layers, args.max_layers)

    refsA = build_layer_keyrefs(wmA, max_layers=layersA)
    refsB = build_layer_keyrefs(wmB, max_layers=layersB)

    common_layers = sorted(set(refsA.keys()) & set(refsB.keys()))
    common_layers = [L for L in common_layers if L < probe_layers]

    print(f"Probing {len(common_layers)} layers...")

    header_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def get_header(repo: str, shard: str) -> Dict[str, Any]:
        k = (repo, shard)
        if k not in header_cache:
            header_cache[k] = fetch_safetensors_header(
                repo, shard, args.revision, token, max_header_bytes
            )
        return header_cache[k]

    def get_tensor_by_key(repo: str, ref: TensorKeyRef) -> np.ndarray:
        header = get_header(repo, ref.shard)
        if ref.key not in header:
            raise KeyError(f"Key not in header: {repo} {ref.shard} {ref.key}")
        return fetch_tensor(ref.key, header, token)

    rows: List[Dict[str, Any]] = []
    categories = ["norm_pre", "norm_post"]

    for lb in common_layers:
        la = lb

        catsA = refsA.get(la, {})
        catsB = refsB.get(lb, {})

        for cat in categories:
            if cat not in catsA or cat not in catsB:
                continue

            refA = catsA[cat]
            refB = catsB[cat]

            try:
                hA = get_header(args.a, refA.shard)
                hB = get_header(args.b, refB.shard)
                if refA.key not in hA or refB.key not in hB:
                    continue

                bytesA = tensor_nbytes(hA[refA.key])
                bytesB = tensor_nbytes(hB[refB.key])
                if bytesA > args.max_tensor_bytes or bytesB > args.max_tensor_bytes:
                    continue

                A = get_tensor_by_key(args.a, refA)
                B = get_tensor_by_key(args.b, refB)

                if (
                    A.shape != B.shape
                    and A.ndim == 2
                    and B.ndim == 2
                    and A.shape == B.shape[::-1]
                ):
                    B = B.T

                if A.shape != B.shape:
                    continue

                cos_ab = cosine(A, B)
                cos_shuffled = shuffled_cosine(A, B)
                cos_centered = centered_cosine(A, B)

                print(
                    f"Layer {lb:3d} {cat:12s}: "
                    f"cos(A,B)={cos_ab:.4f}, cos(shuf(A),shuf(B))={cos_shuffled:.4f}, "
                    f"centered={cos_centered:.4f}, μ_A={A.mean():.4f}, μ_B={B.mean():.4f}"
                )

                rows.append(
                    {
                        "layer": lb,
                        "category": cat,
                        "cosine": cos_ab,
                        "cosine_shuffled": cos_shuffled,
                        "cosine_centered": cos_centered,
                        "A_mean": float(A.mean()),
                        "A_std": float(A.std()),
                        "B_mean": float(B.mean()),
                        "B_std": float(B.std()),
                        "key_A": refA.key,
                        "key_B": refB.key,
                    }
                )

            except Exception as e:
                eprint(f"[warn] layer {lb} cat={cat} failed: {e}")

    csv_path = os.path.join(args.outdir, "layernorm_sanity.csv")
    cols = [
        "layer",
        "category",
        "cosine",
        "cosine_shuffled",
        "cosine_centered",
        "A_mean",
        "A_std",
        "B_mean",
        "B_std",
        "key_A",
        "key_B",
    ]
    write_csv(csv_path, rows, cols)
    print(f"\n[ok] wrote {csv_path}")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.edgecolor": "#dee2e6",
            "grid.color": "#dee2e6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.5,
        }
    )

    COLOR_ORIGINAL = "#2563eb"
    COLOR_SHUFFLED = "#dc2626"
    COLOR_CENTERED = "#16a34a"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, cat in enumerate(categories):
        cat_rows = [r for r in rows if r["category"] == cat]
        if len(cat_rows) < 2:
            continue
        layers = [r["layer"] for r in cat_rows]
        cos_vals = [r["cosine"] for r in cat_rows]
        shuf_vals = [r["cosine_shuffled"] for r in cat_rows]
        cent_vals = [r["cosine_centered"] for r in cat_rows]

        ax = axes[idx]
        ax.plot(
            layers,
            cos_vals,
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLOR_ORIGINAL,
            label="cosine(A, B)",
        )
        ax.plot(
            layers,
            shuf_vals,
            marker="D",
            markersize=3,
            linewidth=1.5,
            color=COLOR_SHUFFLED,
            linestyle="--",
            label="cosine(shuffle(A), shuffle(B))",
            alpha=0.8,
        )
        ax.plot(
            layers,
            cent_vals,
            marker="s",
            markersize=3,
            linewidth=2,
            color=COLOR_CENTERED,
            label=r"cosine(A−$\mu_A$, B−$\mu_B$)",
        )
        ax.axhline(y=0, color="#6b7280", linestyle=":", alpha=0.5)
        ax.set_xlabel("Layer Index", fontweight="medium")
        if idx == 0:
            ax.set_ylabel("Similarity", fontweight="medium")
        ax.set_title(
            cat.upper().replace("_", " "),
            fontsize=12,
            fontweight="medium",
            fontstyle="italic",
        )
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.1, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=12,
        framealpha=0.95,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "LayerNorm Weight Similarity: Sanity Check",
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_png = os.path.join(args.outdir, "plot_summary.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[ok] wrote {out_png}")

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue

        cos_mean = np.mean([r["cosine"] for r in cat_rows])
        shuf_mean = np.mean([r["cosine_shuffled"] for r in cat_rows])
        cent_mean = np.mean([r["cosine_centered"] for r in cat_rows])

        print(f"\n{cat}:")
        print(f"  Mean cosine(A, B):                      {cos_mean:.4f}")
        print(f"  Mean cosine(shuffle(A), shuffle(B)):    {shuf_mean:.4f}")
        print(f"  Mean centered cosine (A−μ_A, B−μ_B):    {cent_mean:.4f}")

        if abs(cos_mean - shuf_mean) < 0.1:
            print(f"  [!] Original ~ Shuffled: cosine may be meaningless")
        else:
            print(f"  [v] Original >> Shuffled: real similarity exists")

        if abs(cent_mean) < 0.1:
            print(f"  [!] Centered ~ 0: no pattern similarity after mean removal")
        elif cent_mean > 0.5:
            print(f"  [v] Centered > 0.5: genuine pattern similarity")

    print("\n[done]")


if __name__ == "__main__":
    main()
