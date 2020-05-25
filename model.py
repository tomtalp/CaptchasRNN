import torch
from torch import nn
import torchvision

x, labels = next(iter(train_dataset_loader))
x = x[:1, :, :, :]
labels = labels[:1]

resnet = torchvision.models.resnet18(pretrained=True)
modules = list(resnet.children())[:-2]
resnet = nn.Sequential(*modules)

conv_out = resnet(x) # This is of size batch_size * C * H * W . For Resnet18, C = 512
batch_size = conv_out.size(0)
features = conv_out.size(1)
conv_out = conv_out.view(batch_size, -1, features)

rnn = nn.RNN(512, 256, 1, batch_first=True) 

rnn_outputs, h = rnn(conv_out)

# Iterate over all of the RNN sequence outputs and execute a FC layer on them
rnn_seq_len = rnn_outputs.size(1)

fc = nn.Linear(256, 3)

fc_outputs = []
for seq_idx in range(rnn_seq_len):
    output = fc(rnn_outputs[:, seq_idx, :])
    fc_outputs.append(output)

# outputs = torch.stack(fc_outputs).permute(1, 0, 2) # Now outputs is of shape batch_size * seq_len * class_num
outputs = torch.stack(fc_outputs).log_softmax(2) # Outputs is of shape seq_len * batch_size * class_num (which is what CTC expects)

T = 16      # Input sequence length
C = 3      # Number of classes (including blank)
N = 1      # Batch size
S = 6      # Target sequence length of longest target in batch
S_min = 4  # Minimum target length, for demonstration purposes

inputs_len = tuple([16 for n in range(N)])
targets = torch.cat(labels)
targets_len = tuple([len(l) for l in labels])

ctc_loss = nn.CTCLoss()
loss = ctc_loss(outputs, targets, inputs_len, targets_len)


fake_input = torch.Tensor([
        [[-0.9348, -0.0279, -1.0001]],
        [[-0.8307, -0.0860, -1.0001]],
        [[-0.0001, -1.6588, -1.2896]],
        [[-0.9808, -0.0465, -1.0001]],
        [[-0.0001, -1.1834, -1.4781]],
        [[-0.6965, -0.0715, -1.0001]],
        [[-0.0001, -1.2597, -1.8087]],
        [[-0.9644, -0.0001, -1.3944]],
        [[-0.0001, -1.1862, -1.2642]],

        # [[5, -1.8495, -1.7262]],

        # [[5, -1.4327, -1.6968]],

        # [[5, -1.4702, -1.3724]],

        # [[5, -1.0238, -1.3466]],

        # [[5, -1.0890, -1.0840]],

        # [[5, -1.7125, -1.3317]],

        # [[5, -1.2762, -1.2437]]
        
        ]
    )

fake_inputs_len = tuple([1 for n in range(1)])
# fake_targets = torch.IntTensor([2, 2, 2, 1])
fake_targets = torch.IntTensor([[1, 1, 1, 1]])
# targets_len = tuple([len(l) for l in labels])
fake_targets_len = (4,)

ctc_loss = nn.CTCLoss()
loss = ctc_loss(fake_input, fake_targets, fake_inputs_len, fake_targets_len)

