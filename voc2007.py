#%%
import os
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
#%%

data_dir = r"C:\won\data\pascal_voc"
#
train_data, dataset_info = tfds.load("voc/2007", split="train", data_dir = data_dir, with_info=True)

val_data, _ = tfds.load("voc/2007", split="validation", data_dir = data_dir, with_info=True)

test_data, _ = tfds.load("voc/2007", split="test", data_dir = data_dir, with_info=True)
#
#
#
#%%
info = dict()
info['labels'] = dataset_info.features['labels'].names
info['dataset_columns'] = ['image','bbox','label','is_difficult']

info['dataset_columns']
info['train_filename'] = []
info['val_filename'] = []
info['test_filename'] = []


#%%
inst_train_dir = r"C:\won\data\pascal_voc\downloads\extracted\TAR.pjreddi.com_media_files_VOCtrai_6-Nov-2007fYzZURAbCVfd_XpTC9yKlPBhIc_B5RG7WTfpcwIMdQg.tar\VOCdevkit\VOC2007\SegmentationObject\\"
inst_test_dir = r"C:\won\data\pascal_voc\downloads\extracted\TAR.pjreddi.com_media_files_VOCtest_6-Nov-2007aDaIji4B3KhFd6hJ0zn6T3Ph5PE10xJDDEhWtWCbSJI.tar\VOCdevkit\VOC2007\SegmentationObject\\"
#%%
for i in tfds.as_numpy(train_data):
    tmp = [] 
    if os.path.isfile(inst_train_dir + str(i['image/filename'])[2:-5] + '.png') == True:
        info['train_filename'].append(str(i['image/filename'])[2:-5])
        tmp.append(i['image'])
        tmp.append(i['objects']['bbox'])
        tmp.append(i['objects']['label'])
        tmp.append(i['objects']['is_difficult'])
        mask_tmp = np.array(Image.open(inst_train_dir + str(i['image/filename'])[2:-5] + '.png'))
        mask_tmp = np.where(mask_tmp==255, 0, mask_tmp)
        mask_tmp = np.reshape(mask_tmp, (mask_tmp.shape[0], mask_tmp.shape[1], 1))
        tmp.append(mask_tmp)
        tmp = np.array(tmp)
        np.save((r"C:\won\data\pascal_voc\voc2007_np_inst\train_val\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
for i in tfds.as_numpy(val_data):
    tmp = []
    if os.path.isfile(inst_train_dir + str(i['image/filename'])[2:-5] + '.png') == True:
        info['val_filename'].append(str(i['image/filename'])[2:-5])
        tmp.append(i['image'])
        tmp.append(i['objects']['bbox'])
        tmp.append(i['objects']['label'])
        tmp.append(i['objects']['is_difficult'])
        mask_tmp = np.array(Image.open(inst_train_dir + str(i['image/filename'])[2:-5] + '.png'))
        mask_tmp = np.where(mask_tmp==255, 0, mask_tmp)
        mask_tmp = np.reshape(mask_tmp, (mask_tmp.shape[0], mask_tmp.shape[1], 1))
        tmp.append(mask_tmp)
        tmp = np.array(tmp)
        np.save((r"C:\won\data\pascal_voc\voc2007_np_inst\train_val\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
for i in tfds.as_numpy(test_data):
    tmp = []
    if os.path.isfile(inst_test_dir + str(i['image/filename'])[2:-5] + '.png') == True:
        info['test_filename'].append(str(i['image/filename'])[2:-5])
        tmp.append(i['image'])
        tmp.append(i['objects']['bbox'])
        tmp.append(i['objects']['label'])
        tmp.append(i['objects']['is_difficult'])
        mask_tmp = np.array(Image.open(inst_test_dir + str(i['image/filename'])[2:-5] + '.png'))
        mask_tmp = np.where(mask_tmp==255, 0, mask_tmp)
        mask_tmp = np.reshape(mask_tmp, (mask_tmp.shape[0], mask_tmp.shape[1], 1))
        tmp.append(mask_tmp)
        tmp = np.array(tmp)
        np.save((r"C:\won\data\pascal_voc\voc2007_np_inst\test\\"+str(i['image/filename'])[2:-5] + ".npy"), tmp, allow_pickle=True)

#%%
info_numpy = np.array([info])
#%%
save_dir = data_dir + r'\voc2007_np_inst'
os.makedirs(save_dir, exist_ok=True)
np.save(save_dir + r'\info.npy', info_numpy, allow_pickle=True)

# %%
