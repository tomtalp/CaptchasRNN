import string
from captcha_dataset import get_metadata_df, CaptchaDataset
from utils import custom_collate_func, LabelConverter
import torch


if __name__ == "__main__":
    vocab = "01"
    lc = LabelConverter(vocab)
    # train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/sample_full_captchas'
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/generated_images_1590229754'
    
    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    # train_dataset_metadata_df = train_dataset_metadata_df[train_dataset_metadata_df['raw_label'].str.len() == 4]

    train_dataset = CaptchaDataset(train_dataset_metadata_df, vocab)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)


import torchvision
import numpy as np
import matplotlib.pyplot as plt
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

inputs, classes = next(iter(train_dataset_loader))
classes_as_str = []
for c in classes:
    classes_as_str.append(lc.decode(c))

out = torchvision.utils.make_grid(inputs, scale_each = True, normalize = True, padding = 4, nrow=4)
imshow(out, title=classes_as_str)
# imshow(out, title=[letters_for_model[x] for x in classes])