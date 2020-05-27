import string
from captcha_dataset import get_metadata_df, CaptchaDataset
from utils import custom_collate_func, LabelConverter
from model import ConvRNN
import torch


if __name__ == "__main__":
    vocab = "01"
    lc = LabelConverter(vocab)
    # train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/sample_full_captchas'
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/generated_images_1590229754'
    
    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    train_dataset_metadata_df = train_dataset_metadata_df.head(1) # For testing
    # train_dataset_metadata_df = train_dataset_metadata_df[train_dataset_metadata_df['raw_label'].str.len() == 4]

    train_dataset = CaptchaDataset(train_dataset_metadata_df, vocab)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)

    T = 16      # Input sequence length
    C = 3      # Number of classes (including blank)
    N = 1      # Batch size
    S = 6      # Target sequence length of longest target in batch
    S_min = 4  # Minimum target length, for demonstration purposes
    
    model = ConvRNN()
    loss_func = nn.CTCLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.1)


    inputs_len = tuple([16 for n in range(N)])    
    total_steps = len(train_dataset_loader)
    epochs = 100
    for epoch_num in range(epochs):
        for i, (img_batch, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            fw_pass_output = model(img_batch)

            targets = torch.cat(labels)
            targets_len = tuple([len(l) for l in labels])
            print("targets = ", targets)

            loss = ctc_loss(outputs, targets, inputs_len, targets_len)
            loss.backward()
            optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_num+1, epochs, i+1, total_steps, loss.item()))



# import torchvision
# import numpy as np
# import matplotlib.pyplot as plt
# def imshow(inp, title=None):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated

# inputs, classes = next(iter(train_dataset_loader))
# classes_as_str = []
# for c in classes:
#     classes_as_str.append(lc.decode(c))

# out = torchvision.utils.make_grid(inputs, scale_each = True, normalize = True, padding = 4, nrow=4)
# imshow(out, title=classes_as_str)
# # imshow(out, title=[letters_for_model[x] for x in classes])