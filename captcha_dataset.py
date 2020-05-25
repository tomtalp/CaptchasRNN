import glob
import string

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import LabelConverter

def get_metadata_df(images_path):
    """
    Returns a Pandas DF containing all the metadata for our PyTorch dataset.
    This DF has 2 columns - An image path and it's label. 
    The label can be one of 62 characters - 26 English letters * 2 (upper/lowercase) + 10 digits = 62

    @param images_path (str) - The path from which image paths will be collected
    @return Pandas DF (<img_path:str>, <label:str>)
    """
    dataset_images = glob.glob("{base_dataset}/*.png".format(base_dataset=images_path))

    images_data_for_df = []
    for img in dataset_images:
        # The path we're getting is the full path - /path/to/img.png. Split by `/` and get the last part - that's our image name!
        filename = img.split("/")[-1] 

        # Our file names are of the following format - <char>_<random_id>.png
        # Extract the char name by splitting via the `_` char
        label = filename.split("_")[0]

        info = {
            "img_path": img,
            "raw_label": label,
        }
        images_data_for_df.append(info)

    df = pd.DataFrame(images_data_for_df)
    return df


class CaptchaDataset(Dataset):
    def __init__(self, dataset_metadata_df, vocab):
        self.dataset_metadata_df = dataset_metadata_df
        self.vocab = vocab
        self.label_converter = LabelConverter(self.vocab)
    
    def __len__(self):
        return len(self.dataset_metadata_df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_metadata = self.dataset_metadata_df.iloc[idx]
        img_path = img_metadata[0]
        raw_label = img_metadata[1]

        image = Image.open(img_path)
        preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image)

        label = self.label_converter.encode(raw_label)

        return (image, label)
