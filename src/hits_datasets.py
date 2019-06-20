import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset  # For custom datasets
from torchvision import transforms
from torch.utils.data import DataLoader


class HitsDataset(object):
  def __init__(self, pkl_path):
    self.data_dict = pd.read_pickle(pkl_path)
    images = self.normalize_by_image(self.data_dict['images'])
    labels = np.array(self.data_dict['labels'])
    train_array, test_array, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42)
    self.train_set = Hits(train_array, train_labels)
    self.test_set = Hits(test_array, test_labels)

  def normalize_by_image(self, images):
    images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
    images = images / np.nanmax(images, axis=(1, 2))[
                      :, np.newaxis, np.newaxis, :]
    return images

  def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False,
      num_workers: int = 1) -> (
      DataLoader, DataLoader):
    train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers)
    test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size,
                             shuffle=shuffle_test,
                             num_workers=num_workers)
    return train_loader, test_loader


class Hits(Dataset):
  def __init__(self, images, labels):
    """
    Args:
        csv_path (string): path to csv file
        img_path (string): path to the folder where images are
        transform: pytorch transforms for transforms and tensor conversion
    """
    # Transforms
    self.to_tensor = transforms.ToTensor()
    # # Read the csv file
    # self.data_dict = pd.read_pkl(pkl_path)
    # images = self.normalize_by_image(self.data_dict['images'])
    # labels = np.array(self.data_dict['labels'])
    # train_array, test_array, train_labels, test_labels = train_test_split(
    #     images, labels, test_size=0.3, random_state=42)
    # images
    # print(self.label_arr)
    # Calculate len
    self.image_arr = images
    self.label_arr = labels
    print(self.image_arr.shape)
    self.data_len = self.label_arr.shape[0]

  def __getitem__(self, index):
    # Get image name from the pandas df
    single_image = self.image_arr[index]
    img_as_np = single_image

    # Transform image to tensor
    img_as_tensor = self.to_tensor(img_as_np)

    # Get label(class) of the image based on the cropped pandas column
    single_image_label = self.label_arr[index]

    return (img_as_tensor, single_image_label)

  def __len__(self):
    return self.data_len
