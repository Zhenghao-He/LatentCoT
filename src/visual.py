# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# data = np.array([
#     [73.3, 28.2, 61.6],
#     [72.5, 28.8, 64.0],
#     [69.5, 20.2, 54.0]
# ])

# rows = ["GSM8K", "GPQA", "BBH"]
# cols = ["GSM8K", "GPQA", "BBH"]

# plt.figure(figsize=(5,4))

# ax = sns.heatmap(
#     data,
#     annot=True,
#     fmt=".1f",
#     cmap="Greens",
#     vmin=20, vmax=75,
#     xticklabels=cols,
#     yticklabels=rows,
#     linewidths=0.6,
#     linecolor="white",
#     annot_kws={"size":15},
#     cbar_kws={'label': 'Accuracy (%)'}
# )

# # 坐标轴字体
# ax.set_xlabel("Evaluation Dataset", fontsize=14)
# ax.set_ylabel("Feature Identified From", fontsize=14)

# # tick字体
# ax.tick_params(axis='both', labelsize=13)

# # colorbar字体
# cbar = ax.collections[0].colorbar
# cbar.ax.tick_params(labelsize=12)
# cbar.set_label("Accuracy (%)", fontsize=13)

# plt.tight_layout()
# plt.savefig("cross_dataset_heatmap.pdf")
# plt.show()

##############shift of first token##############
# import matplotlib.pyplot as plt

# tokens_no = ["$", "16", "12", "32", "8"]
# probs_no = [0.18, 0.14, 0.12, 0.11, 0.09]

# tokens_yes = ["Let", "$", "16", "12", "18"]
# probs_yes = [0.22, 0.15, 0.12, 0.10, 0.08]

# fig, axes = plt.subplots(1, 2, figsize=(9,4))  # 图稍微大一点

# # Without steering
# axes[0].bar(tokens_no, probs_no, color="#2e7d32")
# axes[0].set_title("Without Steering", fontsize=16)
# axes[0].set_ylabel("Probability", fontsize=15)
# axes[0].tick_params(axis='x', labelsize=14)
# axes[0].tick_params(axis='y', labelsize=13)

# # With steering
# axes[1].bar(tokens_yes, probs_yes, color="#2e7d32")
# axes[1].set_title("With Steering", fontsize=16)
# axes[1].tick_params(axis='x', labelsize=14)
# axes[1].tick_params(axis='y', labelsize=13)

# plt.tight_layout()
# plt.savefig("first_token_shift.pdf", dpi=300)
# plt.show()

############ early activation bar plot ############
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

# ── Publication-quality style ──────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "errorbar.capsize": 3,
    "pdf.fonttype": 42,   # embed fonts for publishers
    "ps.fonttype": 42,
})

# Color palette (colorblind-friendly, from Paul Tol)
C_BASE   = "#6baed6"   # mid-blue  — CoT-like prompts
C_DIRECT = "#2171b5"   # dark blue — "Direct" (baseline, stands out)
C_PSEUDO = ["#c6dbef", "#6baed6", "#08519c"]  # light→dark for Direct/Pseudo/CoT

# =========================
# 1. Load cross-prompt data
# =========================
home_path = "/p/realai/zhenghao/Hint_And_Reason/results/meta-llama/Meta-Llama-3.1-8B-Instruct/latent_acts_full/gsm8k/layers.19"

with open(f"{home_path}/direct_acts.pkl", 'rb') as f:
    direct_acts = pickle.load(f)
with open(f"{home_path}/cot_acts.pkl", 'rb') as f:
    cot_acts = pickle.load(f)
with open(f"{home_path}/explain_acts.pkl", 'rb') as f:
    explain_acts = pickle.load(f)
with open(f"{home_path}/solve_acts.pkl", 'rb') as f:
    solve_acts = pickle.load(f)
with open(f"{home_path}/think_acts.pkl", 'rb') as f:
    think_acts = pickle.load(f)
with open(f"{home_path}/pseudo_acts.pkl", 'rb') as f:
    pseudo_acts = pickle.load(f)

def to_scalar_list(acts):
    return [a.max().item() for a in acts]

direct_vals = to_scalar_list(direct_acts)
cot_vals    = to_scalar_list(cot_acts)
explain_vals = to_scalar_list(explain_acts)
solve_vals  = to_scalar_list(solve_acts)
think_vals  = to_scalar_list(think_acts)
pseudo_vals = to_scalar_list(pseudo_acts)

cross_groups = {
    "Direct":          direct_vals,
    "Pseudo-CoT":      pseudo_vals,
    "Step-by-step":    cot_vals,
    "Explain how":     explain_vals,
    "Solve carefully": solve_vals,
    "Think":           think_vals,
}

cross_labels  = list(cross_groups.keys())
cross_means   = [np.mean(v) for v in cross_groups.values()]
cross_stderrs = [np.std(v, ddof=1) / np.sqrt(len(v)) for v in cross_groups.values()]

# Colors:
#   Direct       → darkest blue  (#08519c)
#   Pseudo-CoT   → mid blue      (#4292c6)
#   Step-by-step → light blue    (#9ecae1)
#   others       → warm grey     (#b0b0b0)
_color_map = {
    "Direct":          "#08519c",
    "Pseudo-CoT":      "#4292c6",
    "Step-by-step":    "#9ecae1",
    "Explain how":     "#9ecae1",
    "Solve carefully": "#9ecae1",
    "Think":           "#9ecae1",
}
cross_colors = [_color_map[lbl] for lbl in cross_labels]

# =========================
# 3. y-axis range
# =========================
y_max = max(m + e for m, e in zip(cross_means, cross_stderrs)) * 1.22

# =========================
# 4. Single plot
# =========================
fig, ax = plt.subplots(figsize=(6.5, 3.6))

BAR_WIDTH = 0.58
x = np.arange(len(cross_labels))

ax.bar(
    x, cross_means,
    width=BAR_WIDTH,
    yerr=cross_stderrs,
    error_kw=dict(elinewidth=1.2, ecolor="#444444", capthick=1.2),
    color=cross_colors,
    edgecolor="white",
    linewidth=0.5,
    zorder=3,
)

ax.yaxis.grid(True, linewidth=0.5, color="#cccccc", zorder=0)
ax.set_axisbelow(True)
ax.set_xticks(x)
ax.set_xticklabels(cross_labels, rotation=28, ha="right")
ax.set_ylabel("Mean max activation", labelpad=6)
ax.set_xlabel("Prompt type", labelpad=4)
ax.set_ylim(0, y_max)
ax.set_xlim(-0.55, len(cross_labels) - 0.45)
ax.set_title("Reasoning-feature activation across prompt types", pad=7)

# Legend to explain the color grouping
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#08519c", label="Direct (no CoT)"),
    Patch(facecolor="#4292c6", label="Pseudo-CoT"),
    Patch(facecolor="#9ecae1", label="CoT variants"),
]
ax.legend(handles=legend_handles, fontsize=9.5,
          frameon=True, framealpha=0.9, edgecolor="#cccccc",
          loc="upper left", borderpad=0.6, handlelength=1.2)

plt.tight_layout()
plt.savefig("cross_prompt_and_pseudo_cot.pdf", dpi=300, bbox_inches="tight")
plt.show()