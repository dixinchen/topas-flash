# %%
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import matplotlib.ticker as mticker
import seaborn as sns
sns.set(style="white", font_scale=1.2)
plt.rcParams['font.family'], plt.rcParams['axes.linewidth'] = 'DejaVu Sans', 1.5
plt.rcParams['xtick.bottom'], plt.rcParams['ytick.left'] = True, True

# %%
# load /Users/dchen/Desktop/hello/Stanford/Sep/L16_FLASH15x_PDD.xlsx into a pandas dataframe
df = pd.read_excel('/Users/dchen/Desktop/Desktop - nickelfish/hello/Stanford/Sep/L16_FLASH15x_PDD.xlsx')

depth = df['Depth']
dose = df['15X']

# %%
d_max = depth[np.argmax(dose)]
x90 = []
y90 = []
for i in range(len(dose)-np.argmax(dose)):
    ii = i + np.argmax(dose)
    x90.append(depth[ii])
    y90.append(dose[ii])
    if dose[ii] < 85:
        break
# linear fit to x90 and y90
p = np.polyfit(x90, y90, 1)
# get fitted values
yfit = np.polyval(p, np.linspace(min(x90), max(x90), 1000))
plt.plot(x90, y90, 'k.', label='Measured')
plt.plot(np.linspace(min(x90), max(x90), 1000), yfit, 'r--', label='Fitted')
# find the intersection of the fitted line and the 90% dose line
r_90 = (90 - p[1]) / p[0]
plt.axvline(x=r_90, color='b', linestyle='--', label=f'$R_{{90\%}}$ = {r_90:.2f} cm')
plt.legend(loc='lower left', frameon=False)
plt.xlabel('Depth (cm)')
plt.ylabel('PDD (%)')
plt.tight_layout()
plt.show(block=True)

# %%
d_max = depth[np.argmax(dose)]
x80 = []
y80 = []
for i in range(len(dose)):
    for i in range(len(dose)):
        if dose[i] > 70 and dose[i] < 90 and depth[i] > d_max:
            x80.append(depth[i])
            y80.append(dose[i])
# linear fit to x90 and y90
p = np.polyfit(x80, y80, 1)
# get fitted values
yfit = np.polyval(p, np.linspace(min(x80), max(x80), 1000))
plt.plot(x80, y80, 'k.', label='Measured')
plt.plot(np.linspace(min(x80), max(x80), 1000), yfit, 'r--', label='Fitted')
# find the intersection of the fitted line and the 90% dose line
r_80 = (80 - p[1]) / p[0]
plt.axvline(x=r_80, color='b', linestyle='--', label=f'$R_{{80\%}}$ = {r_80:.2f} cm')
plt.legend(loc='lower left', frameon=False)
plt.xlabel('Depth (cm)')
plt.ylabel('PDD (%)')
plt.tight_layout()

# %%
x50 = []
y50 = []
for i in range(len(dose)-np.argmax(dose)):
    for i in range(len(dose)):
        if dose[i] > 40 and dose[i] < 60:
            x50.append(depth[i])
            y50.append(dose[i])
# linear fit to x90 and y90
p = np.polyfit(x50, y50, 1)
# get fitted values
yfit = np.polyval(p, np.linspace(min(x50), max(x50), 1000))
plt.plot(x50, y50, 'k.', label='Measured')
plt.plot(np.linspace(min(x50), max(x50), 1000), yfit, 'r--', label='Fitted')
# find the intersection of the fitted line and the 90% dose line
r_50 = (50 - p[1]) / p[0]
plt.axvline(x=r_50, color='b', linestyle='--', label=f'$R_{{50\%}}$ = {r_50:.2f} cm')

plt.legend(loc='lower left', frameon=False)
plt.xlabel('Depth (cm)')
plt.ylabel('PDD (%)')
plt.tight_layout()

# %%
fig = plt.figure(figsize=(7, 7/1.6))
plt.plot(depth, dose, '-', markersize=8, color='k')
plt.axvline(x=d_max, color='r', linestyle='--', label=f'D_max = {d_max:.2f} cm')
plt.axvline(x=r_90, color='g', linestyle='--', label=f'R_90 = {r_90:.2f} cm')
plt.axvline(x=r_80, color='b', linestyle='--', label=f'R_80 = {r_80:.2f} cm')
plt.axvline(x=r_50, color='m', linestyle='--', label=f'R_50 = {r_50:.2f} cm')
plt.legend(loc='upper right', frameon=False)
plt.xlabel('Depth (cm)')
plt.ylabel('PDD (%)')
plt.tight_layout()
plt.show(block=True)