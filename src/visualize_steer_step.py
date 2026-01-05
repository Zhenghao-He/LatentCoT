import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)

steps = [1, 2, 3, 4, 5, 6]
steps_np = np.array(steps)
acc_top10_mean = [81.0, 75.5, 56.0, 51.0, 47.5, 43.5]

tok_top10_mean = [106, 110, 107, 62, 73, 56]
tok_top10_std  = [61,  68,  117, 66, 92, 80]

acc_top1_mean = [74.0, 33.0, 35.5, 35.5, 35.0, 35.0]

tok_top1_mean = [81, 29, 29, 30, 30, 29]
tok_top1_std  = [56,  48,  44, 52, 51, 49]

acc_direct = 24.8
tok_direct_mean = 17


LEGEND_FONT_SIZE = 14
LABEL_FONT_SIZE = 16
TICK_FONT_SIZE = 14 

# 上：accuracy
axes[0].plot(steps, acc_top1_mean, marker='o', linewidth=2, label='Top-1')
axes[0].plot(steps, acc_top10_mean, marker='s', linewidth=2, label='Top-10')
axes[0].set_ylabel('Accuracy (%)', fontsize=LABEL_FONT_SIZE)
axes[0].grid(True, linestyle='--', alpha=0.3)

axes[0].axhline(
    acc_direct,
    linestyle='--',
    color='gray',
    linewidth=2,
    label='Direct (no steering)'
)
axes[0].tick_params(axis='x', labelsize=TICK_FONT_SIZE)
axes[0].tick_params(axis='y', labelsize=TICK_FONT_SIZE)

# 下：token
axes[1].plot(steps, tok_top1_mean, marker='o', linewidth=2)
axes[1].fill_between(steps_np,
                     np.array(tok_top1_mean)-np.array(tok_top1_std),
                     np.array(tok_top1_mean)+np.array(tok_top1_std),
                     alpha=0.25)
axes[1].plot(steps, tok_top10_mean, marker='s', linewidth=2)
axes[1].fill_between(steps_np,
                     np.array(tok_top10_mean)-np.array(tok_top10_std),
                     np.array(tok_top10_mean)+np.array(tok_top10_std),
                     alpha=0.25)

axes[1].set_xlabel('Steering step', fontsize=LABEL_FONT_SIZE)
axes[1].set_ylabel('Generated tokens', fontsize=LABEL_FONT_SIZE)
axes[1].grid(True, linestyle='--', alpha=0.3)
axes[1].axhline(
    tok_direct_mean,
    linestyle='--',
    color='gray',
    linewidth=2,
    label='Direct (no steering)',
)
axes[0].legend(frameon=True, fontsize=LEGEND_FONT_SIZE)
axes[1].tick_params(axis='x', labelsize=TICK_FONT_SIZE)
axes[1].tick_params(axis='y', labelsize=TICK_FONT_SIZE)
plt.tight_layout()
# plt.show()
plt.savefig('steer_step_analysis.pdf')
