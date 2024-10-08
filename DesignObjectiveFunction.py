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
def LoadData(filename):
    fp = open(filename)
    rdr = csv.reader(filter(lambda row: row[0]!='#', fp))
    data = []
    for row in rdr:
        data.append(row)
    fp.close()
    data = np.array([[float(y) for y in x] for x in data])
    return data

path = '/Users/dchen/Downloads/100724_beam_model/Emittance_model_bayes/Results'
itt = 6

[binx, biny, binz] = [100, 100, 1]
mid = binx//2
count = binx//2

hlx = 8.460559
hly = 8.460559

pdd_binz = 600
hlz = 5.071535

# %% load TOPAS data
film1_dose = np.zeros([binx, biny, binz])
data_dose = LoadData(path + '/DoseAtFilm1_itt_'+ str(itt) + '.csv')
for j in range(len(data_dose)):
    film1_dose[int(data_dose[j,0]), int(data_dose[j,1]), int(data_dose[j,2])] = data_dose[j,3]

xy_film1_dose = np.sum(film1_dose, axis=2)

topas_top_x = xy_film1_dose[mid, mid-count:mid+count]
topas_top_y = xy_film1_dose[mid-count:mid+count, mid]

film2_dose = np.zeros([binx, biny, binz])
data_dose = LoadData(path + '/DoseAtFilm2_itt_'+ str(itt) + '.csv')
for j in range(len(data_dose)):
    film2_dose[int(data_dose[j,0]), int(data_dose[j,1]), int(data_dose[j,2])] = data_dose[j,3]

xy_film2_dose = np.sum(film2_dose, axis=2)

topas_bot_x = xy_film2_dose[mid, mid-count:mid+count]
topas_bot_y = xy_film2_dose[mid-count:mid+count, mid]

topas_x = np.linspace(-hlx, hlx, binx)
topas_y = np.linspace(-hly, hly, biny)

# normalize xy profiles to 1
topas_top_x = topas_top_x / np.max(topas_top_x)
topas_top_y = topas_top_y / np.max(topas_top_y)
topas_bot_x = topas_bot_x / np.max(topas_bot_x)
topas_bot_y = topas_bot_y / np.max(topas_bot_y)

# load TOPAS PDD
temp_topas_pdd = LoadData(path + '/DoseAtWaterTank_itt_'+ str(itt) + '.csv')
topas_pdd = temp_topas_pdd[:,3]
topas_pdd_z = np.linspace(0, hlz*2, pdd_binz) # 12 cm depth

# normalize to 100
topas_pdd = topas_pdd / np.max(topas_pdd) * 100

# remove the first three according to the measured data
topas_pdd = topas_pdd[3:]
topas_pdd_z = topas_pdd_z[3:]

# %% load measured data
path = '/Users/dchen/Desktop/Desktop_nickelfish/hello/Stanford/Oct/BeamModelOptimisation/'
exp_top_x = np.load(path + 'Measured_data/top_x.npy')
exp_top_y = np.load(path + 'Measured_data/top_y.npy')
exp_bot_x = np.load(path + 'Measured_data/bot_x.npy')
exp_bot_y = np.load(path + 'Measured_data/bot_y.npy')
exp_x = np.load(path + 'Measured_data/x.npy')
exp_y = np.load(path + 'Measured_data/y.npy')

# normalize to 1
exp_top_x = exp_top_x / np.max(exp_top_x)
exp_top_y = exp_top_y / np.max(exp_top_y)
exp_bot_x = exp_bot_x / np.max(exp_bot_x)
exp_bot_y = exp_bot_y / np.max(exp_bot_y)

# downsample to 100 points by averaging
n = 100
exp_top_x_downsampled = exp_top_x.reshape(-1,len(exp_top_x)//n).mean(axis=1)
exp_top_y_downsampled = exp_top_y.reshape(-1,len(exp_top_y)//n).mean(axis=1)
exp_bot_x_downsampled = exp_bot_x.reshape(-1,len(exp_bot_x)//n).mean(axis=1)
exp_bot_y_downsampled = exp_bot_y.reshape(-1,len(exp_bot_y)//n).mean(axis=1)
exp_x_downsampled = np.linspace(min(exp_x), max(exp_x), n)
exp_y_downsampled = np.linspace(min(exp_y), max(exp_y), n)

# load pdd (already normalized to 100)
exp_pdd = np.load(path + 'Measured_data/pdd.npy')
exp_pdd_z = np.load(path + 'Measured_data/exp_pdd_depth.npy')


# %% calculate the mean difference
# x profile
mse_top_x = np.mean((topas_top_x - exp_top_x_downsampled)**2)
mse_bot_x = np.mean((topas_bot_x - exp_bot_x_downsampled)**2)

# y profile
mse_top_y = np.mean((topas_top_y - exp_top_y_downsampled)**2)
mse_bot_y = np.mean((topas_bot_y - exp_bot_y_downsampled)**2)

# pdd
mse_pdd = np.mean((topas_pdd - exp_pdd)**2)

# total mse
mse_total = (mse_top_x + mse_bot_x + mse_top_y + mse_bot_y + mse_pdd) / 5

# %%
plt.plot(topas_pdd_z, topas_pdd)
plt.plot(exp_pdd_z, exp_pdd)
plt.show(block=True)

# %%
plt.figure(figsize=(10,4))
plt.plot(topas_x, topas_top_x, 'r-', label='Film1')
plt.plot(topas_x, topas_bot_x, 'b-', label='Film2')
plt.plot(exp_x, exp_top_x, 'r.', label='Measured Film1')
plt.plot(exp_x, exp_bot_x, 'b.', label='Measured Film2')

# # check symmetry
# plt.plot(np.abs(topas_x), np.abs(topas_top_x), 'r-', label='Film1')
# plt.plot(np.abs(topas_x), np.abs(topas_bot_x), 'b-', label='Film2')
# plt.plot(np.abs(exp_x), np.abs(exp_top_x), 'r.', label='Measured Film1')
# plt.plot(np.abs(exp_x), np.abs(exp_bot_x), 'b.', label='Measured Film2')

plt.show(block=True)
