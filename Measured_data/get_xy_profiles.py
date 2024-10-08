# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", font_scale=1.2)
plt.rcParams['font.family'], plt.rcParams['axes.linewidth'] = 'DejaVu Sans', 1.5
plt.rcParams['xtick.bottom'], plt.rcParams['ytick.left'] = True, True
import cv2
import scipy.optimize as opt

# %%
pixel2cm = 0.016921118
path = '/Users/dchen/Desktop/Desktop_nickelfish/hello/Stanford/Sep/films_092324/'

bkg_file = '/Users/dchen/Desktop/Desktop_nickelfish/hello/Stanford/Sep/091224_dose_profile/film001.tif'
data = cv2.imread(bkg_file, cv2.IMREAD_GRAYSCALE)
y = 50
h = 450
x = 50
w = 450
data = cv2.bitwise_not(data[y:y+h, x:x+w])
# plt.imshow(data)
# plt.show(block=True)
bkg = np.mean(data)

# %%
# 33 cm no shield (top)
image_path = path + 'film001.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# plt.imshow(image)
w = len(image[0])
h = len(image)
M = cv2.getRotationMatrix2D((w/2, h/2), 0.4, 1)
data = cv2.warpAffine(image, M, (w, h))
y = 135 # checked
h = 1200
x = 14 # checked
w = 1200
data = cv2.bitwise_not(data[y:y+h, x:x+w])
# plt.figure()
# plt.imshow(data, extent=[-h*pixel2cm/2, h*pixel2cm/2, -w*pixel2cm/2, w*pixel2cm/2], cmap='gray')
# plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
# plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
# plt.xlabel('X (cm)')
# plt.ylabel('Y (cm)')
# plt.colorbar(label='Dose (a.u.)')
# plt.tight_layout()
# plt.show(block=True)

# 23 cm no shield (bottom)
image_path = path + 'film002.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# plt.imshow(image)
w = len(image[0])
h = len(image)
M = cv2.getRotationMatrix2D((w/2, h/2), -0.1, 1)
data2 = cv2.warpAffine(image, M, (w, h))
y = 152 # checked
h = 1200
x = 26 # checked
w = 1200
data2 = cv2.bitwise_not(data2[y:y+h, x:x+w])
# plt.figure()
# plt.imshow(data2, extent=[-h*pixel2cm/2, h*pixel2cm/2, -w*pixel2cm/2, w*pixel2cm/2], cmap='gray')
# plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
# plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
# plt.xlabel('X (cm)')
# plt.ylabel('Y (cm)')
# plt.colorbar(label='Dose (a.u.)')
# plt.tight_layout()
# plt.show(block=True)

mid = 1200//2
count = 500
top_x = data[mid, mid-count:mid+count]
top_y = data[mid-count:mid+count, mid]
bot_x = data2[mid, mid-count:mid+count]
bot_y = data2[mid-count:mid+count, mid]

x = np.linspace(-count,count,count*2)*pixel2cm
y = np.linspace(-count,count,count*2)*pixel2cm

plt.plot(np.abs(x), np.abs(top_x), '-', label='No shield', color='blue', alpha=1)
plt.plot(np.abs(y), np.abs(top_y), '-', label = 'No shield', color = 'red')
plt.plot(np.abs(x), np.abs(bot_x), '-', label='With shield', color='green', alpha=1)
plt.plot(np.abs(y), np.abs(bot_y), '-', label = 'With shield', color = 'orange')
plt.legend(frameon=False)
plt.xlabel('Y (cm)')
plt.ylabel('Dose (a.u.)')
plt.tight_layout()
plt.show(block=True)

# %%

top_x_norm = top_x-bkg
bot_x_norm = bot_x-bkg

ini = -1
fin = 1
diff = x[1]-x[0]
for i in range(len(x)):
    if x[i] > ini and x[i] < ini+diff:
        temp_ini = i
    if x[i] > fin-diff and x[i] < fin:
        temp_fin = i
max_top_x = np.mean(top_x_norm[temp_ini:temp_fin])
max_bot_x = np.mean(bot_x_norm[temp_ini:temp_fin])

plt.plot(x, top_x_norm, '-', label='No shield', color='blue', alpha=1)
plt.plot(x, bot_x_norm, '-', label='With shield', color='green', alpha=1)
plt.show(block=True)

# %%
hm_top_x = max_top_x/2
hm_bot_x = max_bot_x/2

from scipy.interpolate import splrep, sproot, splev
s = splrep(x, top_x_norm, k=5)
s2 = splrep(x, bot_x_norm, k=5)

x2 = np.linspace(min(x), max(x), 20000)
y2 = splev(x2, s)

y22 = splev(x2, s2)

# plt.plot(x, top_x_norm, 'r.', x2, y2)
# plt.show(block=True)
# %%
fwhm_i = np.where(y2 == min(y2[:len(y2)//2], key=lambda x:abs(x-hm_top_x)))
fwhm_f = np.where(y2 == min(y2[len(y2)//2:], key=lambda x:abs(x-hm_top_x)))

fwhm_i2 = np.where(y22 == min(y22[:len(y22)//2], key=lambda x:abs(x-hm_bot_x)))
fwhm_f2 = np.where(y22 == min(y22[len(y22)//2:], key=lambda x:abs(x-hm_bot_x)))

fwhm = np.abs(x2[fwhm_f]-x2[fwhm_i])
fwhm2 = np.abs(x2[fwhm_f2]-x2[fwhm_i2])

# %%
# plot the fwhm on the graph
plt.figure(figsize=(8.5, 4.5))
plt.plot(x, top_x_norm, '-', label='Film 1', color='blue', alpha=1)
plt.plot(x, bot_x_norm, '-', label='Film 2', color='green', alpha=1)
plt.axvline(x=x2[fwhm_i], color='blue', linestyle='--', linewidth=1)
plt.axvline(x=x2[fwhm_f], color='blue', linestyle='--', linewidth=1)
plt.axvline(x=x2[fwhm_i2], color='green', linestyle='--', linewidth=1)
plt.axvline(x=x2[fwhm_f2], color='green', linestyle='--', linewidth=1)
plt.annotate(f'FWHM = {fwhm[0]:.2f} cm', xy=(4, hm_top_x), xytext=(4, hm_top_x+0.1), color='blue')
plt.annotate(f'FWHM = {fwhm2[0]:.2f} cm', xy=(4, hm_bot_x), xytext=(4, hm_bot_x+0.1), color='green')
plt.xlabel('X (cm)')
plt.ylabel('Dose (a.u.)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show(block=True)

# %%
top_y_norm = top_y-bkg
bot_y_norm = bot_y-bkg

ini = -0.75
fin = 1.25
diff = y[1]-y[0]
for i in range(len(y)):
    if y[i] > ini and y[i] < ini+diff:
        temp_ini = i
    if y[i] > fin-diff and y[i] < fin:
        temp_fin = i
max_top_y = np.mean(top_y_norm[temp_ini:temp_fin])
max_bot_y = np.mean(bot_y_norm[temp_ini:temp_fin])

plt.plot(y, top_y_norm, '-', label='No shield', color='red', alpha=1)
plt.plot(y, bot_y_norm, '-', label='With shield', color='orange', alpha=1)
plt.show(block=True)

# %%
hm_top_y = max_top_y/2
hm_bot_y = max_bot_y/2

from scipy.interpolate import splrep, sproot, splev
s = splrep(y, top_y_norm, k=5)
s2 = splrep(y, bot_y_norm, k=5)

fit_y = np.linspace(min(y), max(y), 20000)
fit_dose = splev(fit_y, s)
fit_dose2 = splev(fit_y, s2)

# %%
fwhm_iy = np.where(fit_dose == min(fit_dose[:len(fit_dose)//2], key=lambda x:abs(x-hm_top_y)))
fwhm_fy = np.where(fit_dose == min(fit_dose[len(fit_dose)//2:], key=lambda x:abs(x-hm_top_y)))

fwhm_iy2 = np.where(fit_dose2 == min(fit_dose2[:len(fit_dose2)//2], key=lambda x:abs(x-hm_bot_y)))
fwhm_fy2 = np.where(fit_dose2 == min(fit_dose2[len(fit_dose2)//2:], key=lambda x:abs(x-hm_bot_y)))

fwhmy = np.abs(fit_y[fwhm_fy]-fit_y[fwhm_iy])
fwhmy2 = np.abs(fit_y[fwhm_fy2]-fit_y[fwhm_iy2])

# %%

plt.figure(figsize=(8.5, 4.5))
plt.plot(y, top_y_norm, '-', label='Film 1', color='red', alpha=1)
plt.plot(y, bot_y_norm, '-', label='Film 2', color='orange', alpha=1)
plt.axvline(x=fit_y[fwhm_i], color='red', linestyle='--', linewidth=1)
plt.axvline(x=fit_y[fwhm_f], color='red', linestyle='--', linewidth=1)
plt.axvline(x=fit_y[fwhm_i2], color='orange', linestyle='--', linewidth=1)
plt.axvline(x=fit_y[fwhm_f2], color='orange', linestyle='--', linewidth=1)
plt.annotate(f'FWHM = {fwhmy[0]:.2f} cm', xy=(4, hm_top_y), xytext=(4, hm_top_y+0.1), color='red')
plt.annotate(f'FWHM = {fwhmy2[0]:.2f} cm', xy=(4, hm_bot_y), xytext=(4, hm_bot_y+0.1), color='orange')
plt.xlabel('Y (cm)')
plt.ylabel('Dose (a.u.)')
plt.legend(frameon=False)
plt.tight_layout()
plt.show(block=True)

# %%
path = '/Users/dchen/Desktop/Desktop_nickelfish/hello/Stanford/Oct/BeamModelOptimisation/Measured_data/'
np.save(path + 'bot_x.npy', bot_x_norm)
np.save(path + 'bot_y.npy', bot_y_norm)
np.save(path + 'top_x.npy', top_x_norm)
np.save(path + 'top_y.npy', top_y_norm)
np.save(path + 'x.npy', x)
np.save(path + 'y.npy', y)