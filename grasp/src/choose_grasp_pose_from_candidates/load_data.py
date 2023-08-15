import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import os

class CustomDataset(Dataset):
    def __init__(self, path, size, input_size, online=False):
        self.imgs_path = path
        self.size = size
        self.input_size = input_size
        file_list = glob.glob(self.imgs_path + "*")
        self.data = []
        if online:
            for dirpath, dirs, _ in os.walk(path):
                for dir_name in dirs:
                    if dir_name.lower() == "pointcloud":
                        if os.path.exists(os.path.join(dirpath, "pointcloud/success")) and os.path.exists(os.path.join(dirpath, "pointcloud/failure")):
                            img_path, class_name = None, None
                            success_pc = os.listdir(os.path.join(dirpath, "pointcloud/success"))
                            failure_pc = os.listdir(os.path.join(dirpath, "pointcloud/failure"))
                            success_pc = [filename for filename in success_pc if filename.endswith(".png")]#filter out only the image
                            failure_pc = [filename for filename in failure_pc if filename.endswith(".png")]
                            if len(success_pc) > 0:#If folder has a labeled pc in it
                                img_path = os.path.join(os.path.join(dirpath, "pointcloud/success"), success_pc[0][:-4]+".png")
                                class_name = "success"
                            elif len(failure_pc) > 0:#If folder has a labeled pc in it
                                img_path = os.path.join(os.path.join(dirpath, "pointcloud/failure"), failure_pc[0][:-4]+".png")
                                class_name = "failure"
                            if img_path is not None:
                                self.data.append([img_path, class_name, '000', "online"])
                                self.data.append([img_path, class_name, '001', "online"])
                                self.data.append([img_path, class_name, '010', "online"])
                                self.data.append([img_path, class_name, '011', "online"])
                                self.data.append([img_path, class_name, '100', "online"])
                                self.data.append([img_path, class_name, '101', "online"])
                                self.data.append([img_path, class_name, '110', "online"])
                                self.data.append([img_path, class_name, '111', "online"])                        
        else:
            for class_path in file_list:
                class_name = class_path.split("/")[-1]
                img_paths = glob.glob(class_path + "/*.png")
                for img_path in img_paths:
                    self.data.append([img_path, class_name, '000', "offline"])
                    self.data.append([img_path, class_name, '001', "offline"])
                    self.data.append([img_path, class_name, '010', "offline"])
                    self.data.append([img_path, class_name, '011', "offline"])
                    self.data.append([img_path, class_name, '100', "offline"])
                    self.data.append([img_path, class_name, '101', "offline"])
                    self.data.append([img_path, class_name, '110', "offline"])
                    self.data.append([img_path, class_name, '111', "offline"])
        self.class_map = {"failure" : 0, "success" : 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name, augment, online_or_offline = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if online_or_offline == "offline":
            old_size = 0.05#captured size
            new_size = [self.size, self.size]#y,x
            img_size = img.shape[0]
            img = img[int(img_size//2-new_size[0]/old_size/2*img_size):int(img_size//2+new_size[0]/old_size/2*img_size), int(img_size//2-new_size[1]/old_size/2*img_size):int(img_size//2+new_size[1]/old_size/2*img_size)]
            img = cv2.resize(img, (self.input_size, self.input_size), interpolation = cv2.INTER_AREA)

            normalize_value = np.max(img)
            nonzero_rows, nonzero_cols = np.where(img != 0)
            img[nonzero_rows, nonzero_cols] = cv2.add(img[nonzero_rows, nonzero_cols], int(255-normalize_value)).squeeze()

        distance = 0
        with open(img_path[:-4]+"distance.txt", 'r') as file:
            distance = float(file.read())
        if int(augment[2]) == 1:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if int(augment[1]) == 1:
            img = cv2.flip(img, 0)#vertical
        if int(augment[0]) == 1:
            img = cv2.flip(img, 1)#horizontal
        img = img/255#normalize
        img = img[..., np.newaxis].astype(np.float32)
        class_id = self.class_map[class_name]
        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        class_id = torch.tensor([class_id])
        return img_tensor, class_id, torch.tensor([distance])