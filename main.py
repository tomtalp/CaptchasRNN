import string
from captcha_dataset import get_metadata_df, CaptchaDataset
from utils import custom_collate_func, LabelConverter, model_output_to_label, compare_tensors
from model import ConvRNN
import torch
import copy
import time

if __name__ == "__main__":
    # 01 vocab
    # vocab = "01"
    # train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/generated_images_1590229754'

    # Digits vocab
    # vocab = string.digits
    # train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/generated_images_1591000952'

    vocab = string.ascii_lowercase
    train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/local_train_lowercase_ascii'
    
    # vocab = string.ascii_lowercase + string.digits
    # train_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/local_train_lowercase_ascii'
    

    lc = LabelConverter(vocab)

    claptcha_test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/claptcha_test'
    claptcha_test_dataset_metadata_df = get_metadata_df(claptcha_test_dataset_path)
    claptcha_test_dataset_metadata_df = claptcha_test_dataset_metadata_df.head(2)
    claptcha_test_dataset = CaptchaDataset(claptcha_test_dataset_metadata_df, vocab)
    claptcha_test_dataset_loader = torch.utils.data.DataLoader(claptcha_test_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)



    train_dataset_metadata_df = get_metadata_df(train_dataset_path)
    # train_dataset_metadata_df = train_dataset_metadata_df.head(20) # For testing
    # train_dataset_metadata_df = train_dataset_metadata_df[train_dataset_metadata_df['raw_label'].str.len() == 5].head(2)

    train_dataset = CaptchaDataset(train_dataset_metadata_df, vocab)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)

    test_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/local_test_lowercase_ascii'
    test_dataset_metadata_df = get_metadata_df(test_dataset_path)
    test_dataset = CaptchaDataset(test_dataset_metadata_df, vocab)
    test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)

    val_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/validation_dataset_lowercase_ascii'
    val_dataset_metadata_df = get_metadata_df(val_dataset_path)
    val_dataset = CaptchaDataset(val_dataset_metadata_df, vocab)
    val_dataset_loader = torch.utils.data.DataLoader(val_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)

    kaggle_dataset_path = '/Users/tomtalpir/dev/tom/captcha_project/CaptchasRNN/kaggle_captcha/samples'
    kaggle_dataset_metadata_df = get_metadata_df(kaggle_dataset_path)
    kaggle_dataset = CaptchaDataset(kaggle_dataset_metadata_df, vocab, is_external_img=True)
    kaggle_dataset_loader = torch.utils.data.DataLoader(kaggle_dataset, batch_size=200, shuffle=True, collate_fn=custom_collate_func)


    train_dataset = claptcha_test_dataset
    train_dataset_loader = claptcha_test_dataset_loader

    dataloaders = {
        'train': train_dataset_loader,
        'val': test_dataset_loader
    }

    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(test_dataset)
    }

    T = 9     # Input sequence length
    C = len(vocab) + 1 # Number of classes. Add 1 for a blank label
    N = len(train_dataset) # Batch size
    S = 6      # Target sequence length of longest target in batch
    S_min = 4  # Minimum target length, for demonstration purposes
    
    model = ConvRNN(target_size=C)
    loss_func = nn.CTCLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0000001)


    inputs_len = tuple([T for n in range(N)]) # This is because basically a `N`-sized Tensor, with `T` as all it's values.
    total_steps = len(train_dataset_loader)
    epochs = 300
    for epoch_num in range(epochs):
        for i, (img_batch, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            fw_pass_output = model(img_batch)
            # print(fw_pass_output[:5])
            targets = torch.cat(labels)
            targets_len = tuple([len(l) for l in labels])
            # print("targets = ", targets)
        #     break
        # break
            loss = ctc_loss(fw_pass_output, targets, inputs_len, targets_len)
            loss.backward(retain_graph=True)
            optimizer.step()

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch_num+1, epochs, i+1, total_steps, loss.item()))
            prediction = []
            for b in range(N):
                single_element_pred = []
                for t in fw_pass_output.data:
                    _, pred = torch.max(t[b].data, 0)
                    single_element_pred.append(int(pred))
                prediction.append(single_element_pred)

            for i, pred in enumerate(prediction):
                prediction_tensor = model_output_to_label(pred)
                model_pred = lc.decode(prediction_tensor)
                real_label = lc.decode(labels[i])
                print("{i}. Pred = {p}, real = {r} ".format(i=i+1, p=model_pred, r=real_label))


prediction = []
for b in range(N):
    single_element_pred = []
    for t in fw_pass_output.data:
        _, pred = torch.max(t[b].data, 0)
        single_element_pred.append(int(pred))
    prediction.append(single_element_pred)

for i, pred in enumerate(prediction):
    prediction_tensor = model_output_to_label(pred)
    model_pred = lc.decode(prediction_tensor)
    real_label = lc.decode(labels[i])
    print("{i}. Pred = {p}, real = {r} ".format(i=i+1, p=model_pred, r=real_label))


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
# inputs, classes = next(iter(val_dataset_loader))
inputs, classes = next(iter(claptcha_test_dataset_loader))
inputs = inputs[:8, :, :, :]
classes = classes[:8]
classes_as_str = []
for c in classes:
    classes_as_str.append(lc.decode(c))

out = torchvision.utils.make_grid(inputs, scale_each = True, normalize = True, padding = 4, nrow=4)
imshow(out, title=classes_as_str)
# # imshow(out, title=[letters_for_model[x] for x in classes])

# def train_model(model, criterion, optimizer, num_epochs=25):
T = 9      # Input sequence length
C = len(vocab) + 1 # Number of classes. Add 1 for a blank label
S = 6      # Target sequence length of longest target in batch
S_min = 4  # Minimum target length, for demonstration purposes

model = ConvRNN(target_size=C)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00000001)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9, weight_decay=0.0000001)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CTCLoss()

num_epochs = 500
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
for epoch in range(num_epochs):
    since = time.time()
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            # continue
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        # running_corrects = 0

        # Iterate over data.
        N = dataset_sizes[phase] # Batch size
        inputs_len = tuple([T for n in range(N)]) # This is because basically a `N`-sized Tensor, with `T` as all it's values.

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            
            targets = torch.cat(labels)
            targets = targets.to(device)
            targets_len = tuple([len(l) for l in labels])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                # _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, targets, inputs_len, targets_len)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
        
        # print("RANDOM OUTPUTS PRINT - ")
        # print(outputs[:1])
        # print(labels[:1])

        prediction = []
        # for b in range(5): # Print the first 5 predictions
        for b in range(N):
            single_element_pred = []
            for t in outputs.data:
                _, pred = torch.max(t[b].data, 0)
                single_element_pred.append(int(pred))
            prediction.append(single_element_pred)

        for i, pred in enumerate(prediction):
            prediction_tensor = model_output_to_label(pred)
            model_pred = lc.decode(prediction_tensor)
            real_label = lc.decode(labels[i])
            print("{i}. Pred = {p}, real = {r} ".format(i=i+1, p=model_pred, r=real_label))

        epoch_loss = running_loss / dataset_sizes[phase]
        # epoch_acc = running_corrects.double() / dataset_sizes[phase]
        time_elapsed = time.time() - since
        print('Epooch took {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))

        # # deep copy the model
        # if phase == 'val' and epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())
    
    # model_name = '/content/drive/My Drive/Data science/captcha/resnet_18_deeper_ending_{e}.pth'.format(e=epoch)
    # print("Saving {m}".format(m=model_name))
    # torch.save(model.state_dict(), model_name)  

    model_for_eval = ConvRNN(target_size=C)
    model_for_eval.load_state_dict(torch.load("/Users/tomtalpir/Downloads/ctc_rnn_v1_47.pth", map_location=torch.device('cpu')))
    model_for_eval.eval()

    # total_validation_samples = len(val_dataset_loader)
    total_validation_samples = len(kaggle_dataset_loader)
    
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in kaggle_dataset_loader:
            targets = torch.cat(labels)
            outputs = model_for_eval(images)
            # break
            prediction = []
            for b in range(N):
                single_element_pred = []
                for t in outputs.data:
                    _, pred = torch.max(t[b].data, 0)
                    single_element_pred.append(int(pred))
                prediction.append(single_element_pred)

            for i, pred in enumerate(prediction):
                prediction_tensor = model_output_to_label(pred)
                model_pred = lc.decode(prediction_tensor)
                real_label = lc.decode(labels[i])
                total += 1
                # if compare_tensors(prediction_tensor, labels[i]):
                #     correct += 1
                print("{i}. Pred = {p}, real = {r} ".format(i=i+1, p=model_pred, r=real_label))
        
        print("{correct}/{total} = {val}%".format(correct=correct, total=total, val=float(correct) / total))

