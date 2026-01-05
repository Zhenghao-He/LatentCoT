import numpy as np
import matplotlib.pyplot as plt
import pickle
# 每个 list 是一组样本的 early activation（比如 step=1，或前3步 max）
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

means = []
stderrs = []
all_acts = [direct_acts, cot_acts, explain_acts, solve_acts, think_acts]
# all_acts = [direct_acts, cot_acts]
for acts in all_acts:
    for i in range(len(acts)):
        acts[i] = acts[i].max().item()
    means.append(np.mean(acts))
    stderr = np.std(acts, ddof=1) / np.sqrt(len(acts))
    stderrs.append(stderr)
# ====== data ======
groups = {
    "Direct": direct_acts,
    "Step by step": cot_acts,
    "Explain how": explain_acts,
    "Solve carefully": solve_acts,
    "Think": think_acts,
}

labels = list(groups.keys())
colors = ["#add8e6"] * len(labels)   # 统一蓝色
# ====== plot ======
plt.figure(figsize=(5, 3.2))

x = np.arange(len(labels))
plt.bar(
    x,
    means,
    yerr=stderrs,
    capsize=4,
    color=colors,
    edgecolor="black",
    linewidth=0.8,
)



plt.xticks(x, labels, fontsize=11, rotation=25)
plt.ylabel("Activation", fontsize=16)
plt.xlabel("Prompt type", fontsize=16)

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("early_activation_barplot.pdf", dpi=200)
plt.show()
