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
samples1 = [0.54, 1.93, 2.25, 6.6]
# std
yerr1=[0.0, 0.0, 0.0, 0.0]
width = 1.5
bar1 = ax.bar(x1, samples1, width, yerr=yerr1)
# fake number : zacl
# ax.bar_label(bar1, labels=["683M", "683M", "683M", "2.6B", "683M", "3.2B", "683M", "10.8B"])
ax.bar_label(bar1, labels=["0.54B", "1.93B", "2.25B", "6.6B"], fontsize='12')

# mappo
x2 = [3.5, 8.5, 13.5, 18.5]
samples2 = [0.68, 2.64, 3.14, 9.68]
# samples2 = [0.66, 2.2, 2.72, 8.37]
# samples2 = [0.67, 2.93, 3.74, 9.51]
yerr2=[0.0, 0.0, 0.0, 0.0]
width = 1.5
bar1 = ax.bar(x2, samples2, width, yerr=yerr2)
# fake number : zacl
ax.bar_label(bar1, labels=["0.68B", "2.64B", "3.14B", "9.68B"], fontsize='12')
ax.set_ylabel(r"Samples ($\times 10^9$)", fontsize='12')

phase = ["Running and Chasing", "Fort-building", "Ramp-Use", "Ramp-Defense"]
x_phase = [2.5, 7.5, 12.5, 17.5]
ax.set_xticks(x_phase, phase)
ax.set_xticklabels(phase, rotation=0, fontsize='10')
ax.legend(['SACL', 'MAPPO'], fontsize='12')

f.tight_layout()
plt.savefig("samples_bar.png")
plt.show()