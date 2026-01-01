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


def generate_random_vectors_and_cossim(n_samples, dim, mean, std):
    v1 = np.random.normal(mean, std, size=(n_samples, dim))
    v2 = np.random.normal(mean, std, size=(n_samples, dim))
    cos_sim = compute_cosine_similarity(v1, v2)
    pos_elem_ratio = np.mean(np.concatenate([v1, v2]) > 0)
    return cos_sim, pos_elem_ratio


def main():
    os.makedirs("./viz_cossim", exist_ok=True)

    dimensions = [64, 256, 1024, 4096]
    means = [0.1, 0.5, 1.0]
    stds = [1.0, 0.75, 0.5, 0.25, 0.1]
    n_samples = 10000

    np.random.seed(42)

    results = {}

    print("Computing cosine similarities...")
    for mean in tqdm(means, desc="Mean values"):
        for std in stds:
            results[(mean, std)] = {}
            for dim in dimensions:
                cos_sims, pos_elem_ratio = generate_random_vectors_and_cossim(
                    n_samples, dim, mean, std
                )
                results[(mean, std)][dim] = {
                    "mean": np.mean(cos_sims),
                    "std": np.std(cos_sims),
                    "min": np.min(cos_sims),
                    "max": np.max(cos_sims),
                    "pos_ratio": pos_elem_ratio,  # 벡터 원소 중 양수 비율
                }

    marker_styles = ["o", "s", "^", "D", "v", "p"]
    line_styles = ["-", "--", ":", "-."]
    line_colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#A78BFA", "#34D399", "#F472B6"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Cosine Similarity Statistics",
        fontsize=16,
        fontweight="bold",
        color="#E0E0E0",
        y=0.98,
    )

    for j, mean in enumerate(means):
        ax = axes[0, j]
        for i, dim in enumerate(dimensions):
            mean_vals = [results[(mean, std)][dim]["mean"] for std in stds]
            std_vals = [results[(mean, std)][dim]["std"] for std in stds]

            ax.errorbar(
                stds,
                mean_vals,
                yerr=std_vals,
                marker=marker_styles[i],
                linestyle=line_styles[i],
                color=line_colors[i],
                label=f"d={dim}",
                capsize=3,
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Std")
        ax.set_ylabel("Cosine Similarity Mean (± Std)")
        ax.set_title(
            f"Cosine Similarity (Mean={mean})", fontweight="bold", color="#FFD700"
        )
        ax.invert_xaxis()
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")
        ax.axhline(y=0, color="white", linestyle="--", alpha=0.3)

    for j, mean in enumerate(means):
        ax = axes[1, j]
        for i, dim in enumerate(dimensions):
            pos_vals = [results[(mean, std)][dim]["pos_ratio"] for std in stds]

            ax.plot(
                stds,
                pos_vals,
                marker=marker_styles[i],
                linestyle=line_styles[i],
                color=line_colors[i],
                label=f"d={dim}",
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Std")
        ax.set_ylabel("Positive Element Ratio")
        ax.set_title(
            f"Positive Element Ratio (Mean={mean})",
            fontweight="bold",
            color="#FFD700",
        )
        ax.invert_xaxis()
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="white", linestyle="--", alpha=0.3)
        ax.legend(loc="lower right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#1a1a2e")

    plt.tight_layout()
    fig.savefig(
        "/data/private/codes/solar-vs-glm/viz_cossim/results.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="#0f0f1a",
    )
    print("Saved: viz_cossim/results.png")

    for mean in means:
        for std in stds:
            print(f"\n{'─'*60}")
            print(f"Vector Distribution: Mean={mean}, Std={std}")
            print(f"{'─'*60}")
            print(
                f"{'Dim':>6} | {'CosSim Mean':>12} | {'CosSim Std':>12} | {'Min':>8} | {'Max':>8}"
            )
            print(f"{'─'*60}")
            for dim in dimensions:
                r = results[(mean, std)][dim]
                print(
                    f"{dim:>6} | {r['mean']:>12.6f} | {r['std']:>12.6f} | {r['min']:>8.4f} | {r['max']:>8.4f}"
                )


if __name__ == "__main__":
    main()
