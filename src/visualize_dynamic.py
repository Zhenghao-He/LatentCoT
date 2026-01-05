

import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt


def load_stats_from_pkl(pkl_path):
    """
    Load mean / stderr (optional) / count from pkl.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)


    if isinstance(obj, dict):
        mean = obj["mean"].float().cpu()
        count = obj.get("count", None)
        count = count.cpu()
        stderr = obj["stderr"].float().cpu()
    else:
        raise TypeError(f"Unsupported PKL content type: {type(obj)}")

    return mean, stderr, count


def plot_multiple_curves(
    files,
    labels,
    title=None,
    save_path=None,
    min_count=5
):
    assert len(files) == len(labels)

    stats = []
    max_valid_steps = []

    # ---------- First pass: load & find valid lengths ----------
    for pkl_path in files:
        mean, stderr, count = load_stats_from_pkl(pkl_path)
        # import pdb; pdb.set_trace()
        if count is not None:
            valid = (count >= min_count).nonzero(as_tuple=False).squeeze(-1)
            if len(valid) == 0:
                max_valid_steps.append(0)
            else:
                max_valid_steps.append(valid[-1].item() + 1)
        else:
            max_valid_steps.append(len(mean))

        stats.append((mean, stderr, count))

    # Global truncation length
    T_min = min(max_valid_steps)
    # import pdb; pdb.set_trace()
    
    if T_min == 0:
        raise ValueError("No valid steps across curves with current min_count.")
    
    max_steps = 50
    T_min = min(T_min, max_steps)
    plt.figure(figsize=(16, 12))
    # colors = ["tab:orange", "tab:blue", "tab:green"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (mean, stderr, count), label, color in zip(stats, labels, colors):
        mean = mean[:T_min]
        t = np.arange(T_min)

        if stderr is not None:
            stderr = stderr[:T_min]

        plt.plot(t, mean.numpy(), color=color, label=label, linewidth=2)

        if stderr is not None:
            plt.fill_between(
                t,
                (mean - stderr).numpy(),
                (mean + stderr).numpy(),
                color=color,
                alpha=0.25
            )

    plt.xlabel("Step", fontsize=70)
    plt.ylabel("SAE Activation", fontsize=70)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.grid(True, alpha=0.3)


    # if title:
    #     plt.title(title)

    # plt.legend(frameon=False)
    plt.legend(fontsize=70)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    # ========= CONFIG =========
    dataset_name = "gsm8k"
    FILES = [
        f"/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/Llama-3.3-70B-Instruct/latent_acts/{dataset_name}/layers.50/cot_latent_acts_tokenpos1_200.pkl",
        f"/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/Llama-3.3-70B-Instruct/latent_acts/{dataset_name}/layers.50/direct_latent_acts_tokenpos1_200.pkl",
        # f"/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/Meta-Llama-3.1-8B-Instruct/latent_acts/{dataset_name}/layers.19/cot_latent_acts_tokenpos1_10.pkl",
        # f"/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/Meta-Llama-3.1-8B-Instruct/latent_acts/{dataset_name}/layers.19/direct_latent_acts_tokenpos1_10.pkl",
    ]

    LABELS = [
        "CoT",
        "Direct",
    ]

    TITLE = "Reasoning Feature Activation During Generation"
    # SAVE_PATH = f"visual/comparison_{dataset_name}_Llama8B_8629.pdf"
    SAVE_PATH = f"visual/comparison_{dataset_name}_Llama70B_13709.pdf"
    MIN_COUNT = 0  # 只画样本数 >=10 的 step
    # ==========================

    plot_multiple_curves(
        FILES,
        LABELS,
        title=TITLE,
        save_path=SAVE_PATH,
        min_count=MIN_COUNT
    )
