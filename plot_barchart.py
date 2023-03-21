import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



plt.style.use("ggplot")
# plt.rcParams['text.usetex'] = True
# plt.rcParams.update({'legend.fontsize': 14})
# p = sns.color_palette()
# sns.set_palette([p[i] for i in range(3)])
f, ax = plt.subplots(1, 1, figsize=(6, 4))

# format: [chase, box locking, use ramp, ramp defense]
# zacl
x1 = [1.5, 6.5, 11.5, 16.5]
samples1 = [0.40, 1.97, 2.54, 6.44]
# std
yerr1=[0.04, 0.09, 0.08, 1.01]
width = 1.5
bar1 = ax.bar(x1, samples1, width, yerr=yerr1)
ax.bar_label(bar1, labels=["0.40B", "1.97B", "2.54B", "6.44B"], fontsize='12')
# phase1 [0.42, 0.44, 0.347]
# phase2 [1.96, 2.09, 1.87]
# phase3 [2.56, 2.43, 2.62]
# phase4 [7.5, 6.74, 5.08]

# mappo
x2 = [3.5, 8.5, 13.5, 18.5]
samples2 = [0.53, 2.175, 2.65, 7.26]
yerr2=[0.03, 0.085, 0.08, 0.4]
width = 1.5
bar1 = ax.bar(x2, samples2, width, yerr=yerr2)
ax.bar_label(bar1, labels=["0.53B", "2.175B", "2.65B", "7.26B"], fontsize='12')
# phase1 [0.56,  0.50]
# phase2 [2.26,  2.09]
# phase3 [2.73,  2.57]
# phase4 [7.67,  6.86]
ax.set_ylabel(r"Samples ($\times 10^9$)", fontsize='12')

phase = ["Running and Chasing", "Fort-building", "Ramp-Use", "Ramp-Defense"]
x_phase = [2.5, 7.5, 12.5, 17.5]
ax.set_xticks(x_phase, phase)
ax.set_xticklabels(phase, rotation=0, fontsize='10')
ax.legend(['SACL', 'MAPPO'], fontsize='12', loc="upper left")

f.tight_layout()
plt.savefig("samples_bar.png")
plt.show()