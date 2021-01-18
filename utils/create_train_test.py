import os
import numpy as np
import shutil
import sys
import pandas as pd

sys.path.append('../')

root_dir = '../images/'
positive = '/covid'
negative = '/non-covid'

os.makedirs(root_dir + '/train/' + positive, exist_ok=True)
os.makedirs(root_dir + '/train/' + negative, exist_ok=True)
os.makedirs(root_dir + '/val/' + positive, exist_ok=True)
os.makedirs(root_dir + '/val/' + negative, exist_ok=True)
os.makedirs(root_dir + '/test/' + positive, exist_ok=True)
os.makedirs(root_dir + '/test/' + negative, exist_ok=True)

metadata = pd.read_csv('../metadata.csv')

# select only COVID inds
# according to https://github.com/mlmed/covid-baselines/blob/master/xray_covid_baselines.ipynb
target_str = "COVID-19"
labels_mask = ~metadata.finding.str.contains("todo") & \
    (metadata.offset.astype("float") > 0) & \
    (metadata.offset.astype("float") < 8)
labels = metadata.finding.str.contains("COVID-19")[labels_mask]

# slice dataset
metadata = metadata.iloc[labels.axes[0], :]
metadata['label'] = labels

for _class, _cls in zip([True, False], [positive, negative]):

    filenames = [x for x in metadata[metadata.label == _class]
                 ['filename'].values if x in os.listdir('../images')]
    np.random.shuffle(filenames)

    train_files, val_files, test_files = np.split(np.array(filenames),
                                                  [int(len(filenames)*0.7), int(len(filenames)*0.85)])

    train_files = [root_dir + name for name in train_files]
    val_files = [root_dir + name for name in val_files]
    test_files = [root_dir + name for name in test_files]

    print('Total images:', len(filenames))  # Total images:  259
    print('Training:', len(train_files))  # Training:  181
    print('Validation:', len(val_files))  # Validation:  39
    print('Testing:', len(test_files))  # Testing:  39

    for name in train_files:
        shutil.copy(name, '../images/train' + _cls)

    for name in val_files:
        shutil.copy(name, '../images/val' + _cls)

    for name in test_files:
        shutil.copy(name, '../images/test' + _cls)
