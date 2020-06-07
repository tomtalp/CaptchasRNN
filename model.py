import torch
from torch import nn
import torchvision
import torch.nn.functional as F

class ConvRNN(nn.Module):
    def __init__(self, target_size):
        super(ConvRNN, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]

        self.resnet = nn.Sequential(*modules)
        
        #### Don't optimize resnet
        for p in self.resnet.parameters():
            p.requires_grad = False

        #### Optimize the last 2 blocks
        for c in list(self.resnet.children())[-2:]:
            for p in c.parameters():
                p.requires_grad = True


        self.cnv = nn.Conv2d(512, 512, 2)
        self.rnn = nn.RNN(512, 256, 1, batch_first=True) 
        
        # self.small_fc = nn.Linear(256, target_size)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, target_size)
        # self.fc4 = nn.Linear(64, target_size)

        # self.small_fc.bias.data.fill_(0)
        # self.small_fc.weight.data.uniform_(-0.1, 0.1)

        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-0.1, 0.1)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-0.1, 0.1)
        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-0.1, 0.1)
        # self.fc4.bias.data.fill_(0)
        # self.fc4.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        encoded_img = self.resnet(x)
        # encoded_img = resnet(x)
        encoded_img = F.max_pool2d(self.cnv(encoded_img), (2,2))
        # encoded_img = F.max_pool2d(cnv(encoded_img), (2,2))
        encoded_img = F.relu(encoded_img)

        batch_size = encoded_img.size(0)
        features = encoded_img.size(1)
        encoded_img = encoded_img.view(batch_size, -1, features)

        rnn_outputs, h = self.rnn(encoded_img)
        
        rnn_seq_len = rnn_outputs.size(1)

        fc_outputs = []
        for seq_idx in range(rnn_seq_len):
            
            # output = self.small_fc(rnn_outputs[:, seq_idx, :])
            output = self.fc1(rnn_outputs[:, seq_idx, :])
            output = F.relu(output)

            output = self.fc2(output)
            output = F.relu(output)

            output = self.fc3(output)
            # output = F.relu(output)

            # output = self.fc4(output)

            fc_outputs.append(output)

        outputs = torch.stack(fc_outputs).log_softmax(2) # Outputs is of shape seq_len * batch_size * class_num (which is what CTC expects)

        return outputs




# ######
# x, labels = next(iter(train_dataset_loader))
# x = x[:1, :, :, :]
# labels = labels[:1]

# resnet = torchvision.models.resnet18(pretrained=True)
# modules = list(resnet.children())[:-2]
# resnet = nn.Sequential(*modules)

# conv_out = resnet(x) # This is of size batch_size * C * H * W . For Resnet18, C = 512
# batch_size = conv_out.size(0)
# features = conv_out.size(1)
# conv_out = conv_out.view(batch_size, -1, features)

# rnn = nn.RNN(512, 256, 1, batch_first=True) 

# rnn_outputs, h = rnn(conv_out)

# # Iterate over all of the RNN sequence outputs and execute a FC layer on them
# rnn_seq_len = rnn_outputs.size(1)

# fc = nn.Linear(256, 3)

fc_outputs = []
for seq_idx in range(rnn_seq_len):
    output = fc1(rnn_outputs[:, seq_idx, :])
    output = fc2(output)
    fc_outputs.append(output)

# # outputs = torch.stack(fc_outputs).permute(1, 0, 2) # Now outputs is of shape batch_size * seq_len * class_num
# outputs = torch.stack(fc_outputs).log_softmax(2) # Outputs is of shape seq_len * batch_size * class_num (which is what CTC expects)

# T = 16      # Input sequence length
# C = 3      # Number of classes (including blank)
# N = 1      # Batch size
# S = 6      # Target sequence length of longest target in batch
# S_min = 4  # Minimum target length, for demonstration purposes

# inputs_len = tuple([16 for n in range(N)])
# targets = torch.cat(labels)
# targets_len = tuple([len(l) for l in labels])

# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(outputs, targets, inputs_len, targets_len)


# fake_input = torch.Tensor([
#         [[1, -1, -1]],
#         [[1, -1, -1]],
#         [[-1, 2, -1]], # 1
#         [[-1, 2, -1]], # _1
#         [[1, -1, -1]],
#         [[-1, 2, -1]], # 1
#         [[-1, 2.6, -1]], # -1
#         [[1, -1, -1]],
#         [[-1, -1, 3]], # 2
#         [[-1, -1, 2.7]], # _2
#         [[-1, -1, 4]], # _2
#         [[1, -1, -1]],
#         [[1, -1, -1]],
#         [[-1, -1, 3.5]], # 2
#         [[-1, -1, 3]], # _2
#         [[-1, -1, 3.1]], # -2
#         ]
#     )

# log_softmax_fake_out = fake_input.log_softmax(2)

# fake_inputs_len = tuple([1 for n in range(1)])
# # fake_targets = torch.IntTensor([2, 2, 2, 1])
# fake_targets = torch.IntTensor([[2, 2, 2, 2]])
# # targets_len = tuple([len(l) for l in labels])
# fake_targets_len = (4,)

# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(log_softmax_fake_out, fake_targets, fake_inputs_len, fake_targets_len)

