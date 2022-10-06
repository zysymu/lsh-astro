import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import urllib.error
import os

def download(df):
    obj_paths = []

    for i, row in df.iterrows():
        print(i)
        object_id = round(row['COADD_OBJECT_ID'])
        dist = row['distCol']

        candidate = f'{object_id}-{dist}.jpg'

        ra = row['RA']
        dec = row['DEC']            

        print(f'downloading ra={ra}, dec={dec}')

        legacy_survey = f'https://www.legacysurvey.org/viewer/cutout.jpg?ra={ra}&dec={dec}&layer=ls-dr9&pixscale={ps}'
        obj_paths.append((candidate, i))

        print(legacy_survey)
        try:
            urllib.request.urlretrieve(legacy_survey, os.path.join(output_dir, candidate))
        except urllib.error.HTTPError as e:
            print(e)
            continue

    return obj_paths
            

lsbgs_path = 'LSBG_catalog_v2 (1).csv'
lsbgs = pd.read_csv(lsbgs_path)
lsbgs_ids = lsbgs['COADD_OBJECT_ID']

filepath = 'lsbgs-5.csv'
df = pd.read_csv(filepath, names=['COADD_OBJECT_ID', 'RA', 'DEC' ,'distCol'])
obj = round(df.iloc[0]['COADD_OBJECT_ID'])

merged = pd.merge(df, lsbgs, on='COADD_OBJECT_ID')

# plot histogram

plt.figure(figsize=(15,10), dpi=200)
sns.histplot(df.iloc[1:], x='distCol', color='blue', bins=70, label=f'Nearest neighbors to object ID={obj}')
sns.histplot(merged.iloc[1:], x='distCol', color='orange', bins=70, label='T21 LSBGs')

plt.title(f'Matched objects: {len(merged)}')
plt.xlabel('Distance from key')
plt.ylabel('Count')

plt.legend()
plt.show()


# fetch and plot neighbors
ps = 0.10
output_dir = 'neighbors\\'

print(obj)

tops = download(df.iloc[:6])
mids = download(df.iloc[int(len(df)/2):int(len(df)/2)+5])
fartest = download(df.iloc[-5:])

key = tops[0]
nearest = tops[1:]

fig = plt.figure(figsize=(15, 10))

ax_key= fig.add_subplot(3,6,7)

ax1= fig.add_subplot(3,6,2)
ax2= fig.add_subplot(3,6,3)
ax3= fig.add_subplot(3,6,4)
ax4= fig.add_subplot(3,6,5)
ax5= fig.add_subplot(3,6,6)

ax6= fig.add_subplot(3,6,8)
ax7= fig.add_subplot(3,6,9)
ax8= fig.add_subplot(3,6,10)
ax9= fig.add_subplot(3,6,11)
ax10= fig.add_subplot(3,6,12)

ax11= fig.add_subplot(3,6,14)
ax12= fig.add_subplot(3,6,15)
ax13= fig.add_subplot(3,6,16)
ax14= fig.add_subplot(3,6,17)
ax15= fig.add_subplot(3,6,18)

import matplotlib.image as mpimg

ax_nearest = [ax1, ax2, ax3, ax4, ax5]
ax_mid = [ax6, ax7, ax8, ax9, ax10]
ax_fartest = [ax11, ax12, ax13, ax14, ax15]

for ax, n in zip(ax_nearest, nearest):
    img = mpimg.imread(os.path.join(output_dir, n[0]))
    ax.grid(False)
    ax.set_axis_off()
    title = n[0].split('-')[1][:-4][:5]
    ax.set_title(f'd = {title}; i = {n[1]}')

    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.06 * width
    ypad = 0.01 * height

    if int(n[0].split('-')[0]) in lsbgs_ids.values:
        color = 'green'    
    else:
        color = 'red'

    fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor=color, linewidth=3, fill=False))

    ax.imshow(img)

for ax, n in zip(ax_mid, mids):
    img = mpimg.imread(os.path.join(output_dir, n[0]))
    ax.grid(False)
    ax.set_axis_off()
    title = n[0].split('-')[1][:-4][:6]
    ax.set_title(f'd = {title}; i = {n[1]}')

    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.0 * width
    ypad = 0.01 * height

    if int(n[0].split('-')[0]) in lsbgs_ids.values:
        color = 'green'    
    else:
        color = 'red'

    fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor=color, linewidth=3, fill=False))

    ax.imshow(img)

for ax, n in zip(ax_fartest, fartest):
    img = mpimg.imread(os.path.join(output_dir, n[0]))
    ax.grid(False)
    ax.set_axis_off()
    title = n[0].split('-')[1][:-4][:6]
    ax.set_title(f'd = {title}; i = {n[1]+1}')

    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    x0, y0, width, height = bbox.transformed(fig.transFigure.inverted()).bounds
    # slightly increase the very tight bounds:
    xpad = 0.0 * width
    ypad = 0.01 * height

    if int(n[0].split('-')[0]) in lsbgs_ids.values:
        color = 'green'    
    else:
        color = 'red'

    fig.add_artist(plt.Rectangle((x0-xpad, y0-ypad), width+2*xpad, height+2*ypad, edgecolor=color, linewidth=3, fill=False))

    ax.imshow(img)

img = mpimg.imread(os.path.join(output_dir, key[0]))
ax_key.grid(False)
ax_key.set_axis_off()
#title = key.split('-')[1][:-4][:7]
#ax_key.set_title(f'Key')
ax_key.set_title(f'Key')
ax_key.imshow(img)

fig.suptitle(f'Neighbors of object ID={obj}', fontsize='xx-large')


plt.show()