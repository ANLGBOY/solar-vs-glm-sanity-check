import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.style.use("dark_background")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10


def compute_cosine_similarity(v1, v2):
    dot_product = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_sim = dot_product / (norm1 * norm2)
        cos_sim = np.nan_to_num(cos_sim, nan=0.0)
    return cos_sim


def generate_random_vectors_and_cossim(n_samples, dim, mean1, mean2, std):
    """Generate vectors with different means for v1 and v2."""
    v1 = np.random.normal(mean1, std, size=(n_samples, dim))
    v2 = np.random.normal(mean2, std, size=(n_samples, dim))
    cos_sim = compute_cosine_similarity(v1, v2)
    pos_elem_ratio_v1 = np.mean(v1 > 0)
    pos_elem_ratio_v2 = np.mean(v2 > 0)
    return cos_sim, pos_elem_ratio_v1, pos_elem_ratio_v2


def main():
    os.makedirs("./viz_cossim", exist_ok=True)

    dimensions = [64, 512, 4096]
    mean_pairs = [
        (0.1, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
    ]
    stds = [1.0, 0.75, 0.5, 0.25, 0.1]
    n_samples = 10000

    np.random.seed(42)

    results = {}

    print("Computing cosine similarities...")
    for mean1, mean2 in tqdm(mean_pairs, desc="Mean pairs"):
        for std in stds:
            results[(mean1, mean2, std)] = {}
            for dim in dimensions:
                cos_sims, pos_ratio_v1, pos_ratio_v2 = (
                    generate_random_vectors_and_cossim(
                        n_samples, dim, mean1, mean2, std
                    )
                )
                results[(mean1, mean2, std)][dim] = {
                    "mean": np.mean(cos_sims),
                    "std": np.std(cos_sims),
                    "min": np.min(cos_sims),
                    "max": np.max(cos_sims),
                    "pos_ratio_v1": pos_ratio_v1,
                    "pos_ratio_v2": pos_ratio_v2,
                }

    marker_styles = ["o", "s", "^", "D", "v", "p"]
    line_styles = ["-", "--", ":", "-."]
    line_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA", "#34D399", "#F472B6"]

    n_pairs = len(mean_pairs)
    fig, axes = plt.subplots(3, n_pairs, figsize=(4 * n_pairs, 11))
    fig.suptitle(
        "Cosine Similarity Analysis",
        fontsize=16,
        fontweight="bold",
        color="#E0E0E0",
        y=0.98,
    )

    for j, (mean1, mean2) in enumerate(mean_pairs):
        ax = axes[0, j]
        for i, dim in enumerate(dimensions):
            mean_vals = [results[(mean1, mean2, std)][dim]["mean"] for std in stds]
            std_vals = [results[(mean1, mean2, std)][dim]["std"] for std in stds]

            ax.errorbar(
                stds,
                mean_vals,
                yerr=std_vals,
                marker=marker_styles[i],
                linestyle=line_styles[i % len(line_styles)],
                color=line_colors[i],
                label=f"d={dim}",
                capsize=3,
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Std")
        ax.set_ylabel("Cosine Similarity Mean (± Std)")
        ax.set_title(f"μ₁={mean1}, μ₂={mean2}", fontweight="bold", color="#FFD700")
        ax.invert_xaxis()
        ax.legend(loc="lower right", fontsize=7, ncol=1)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.3)

    # Row 1: Positive Element Ratio for v1
    for j, (mean1, mean2) in enumerate(mean_pairs):
        ax = axes[1, j]
        for i, dim in enumerate(dimensions):
            pos_vals_v1 = [
                results[(mean1, mean2, std)][dim]["pos_ratio_v1"] for std in stds
            ]

            ax.plot(
                stds,
                pos_vals_v1,
                marker=marker_styles[i],
                linestyle=line_styles[i % len(line_styles)],
                color=line_colors[i],
                label=f"d={dim}",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Std")
        ax.set_ylabel("Positive Element Ratio (v1)")
        ax.set_title(
            f"v1 Positive Element Ratio (μ₁={mean1})",
            fontweight="bold",
            color="#4ECDC4",
        )
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")

    # Row 2: Positive Element Ratio for v2
    for j, (mean1, mean2) in enumerate(mean_pairs):
        ax = axes[2, j]
        for i, dim in enumerate(dimensions):
            pos_vals_v2 = [
                results[(mean1, mean2, std)][dim]["pos_ratio_v2"] for std in stds
            ]

            ax.plot(
                stds,
                pos_vals_v2,
                marker=marker_styles[i],
                linestyle=line_styles[i % len(line_styles)],
                color=line_colors[i],
                label=f"d={dim}",
                linewidth=2,
                markersize=6,
            )

        ax.set_xlabel("Std")
        ax.set_ylabel("Positive Element Ratio (v2)")
        ax.set_title(
            f"v2 Positive Element Ratio (μ₂={mean2})",
            fontweight="bold",
            color="#FF6B6B",
        )
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=4)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()
    fig.savefig(
        "./viz_cossim/results.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0f0f1a",
    )
    print("Saved: viz_cossim/results.png")

    for mean1, mean2 in mean_pairs:
        for std in stds:
            print(f"\n{'─'*70}")
            print(f"Vector Distribution: Mean_v1={mean1}, Mean_v2={mean2}, Std={std}")
            print(f"{'─'*70}")
            print(
                f"{'Dim':>6} | {'CosSim Mean':>12} | {'CosSim Std':>12} | {'Min':>8} | {'Max':>8} | {'PosR_v1':>7} | {'PosR_v2':>7}"
            )
            print(f"{'─'*70}")
            for dim in dimensions:
                r = results[(mean1, mean2, std)][dim]
                print(
                    f"{dim:>6} | {r['mean']:>12.6f} | {r['std']:>12.6f} | {r['min']:>8.4f} | {r['max']:>8.4f} | {r['pos_ratio_v1']:>7.4f} | {r['pos_ratio_v2']:>7.4f}"
                )


if __name__ == "__main__":
    main()
