import torch

def model_output_to_label(prediction):
    """
    Receive a list which represents an output from our RNN, and convert it to an actual label.
    Our model has a lot of redundant outputs - The same characters repeated, and a lot of "blank" CTC labels.

    Our goal is to get rid of those characters and return the final label as a PyTorch Tensor

    Example - 
    Input = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0]
    Output = tensor([1, 1, 1, 2])

    Input = [2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0]
    Output = tensor([2, 2, 2, 1])
    """
    last_value = None
    result = []
    for x in prediction:
        if x != last_value and x > 0:
            result.append(x)
        last_value = x
    
    return torch.IntTensor(result)

def compare_tensors(t1, t2):
    """
    Check whether two tensors are identical
    """
    b = torch.all(t1.eq(t2))
    return bool(b)

def custom_collate_func(data):
    """
    Stack the images as one big tensor, and the labels as a list
    This is done because the labels are tensors of varying sizes, and we're forced to create them as a list instead
    of a tensor
    """
    imgs = torch.stack([x[0] for x in data], dim=0)
    labels = [x[1] for x in data]
    return [imgs, labels]

class LabelConverter():
    """
    A utility for encoding/decoding labels.
    A raw label might be "hello123", and we want to represent it as a Tensor
    """
    def __init__(self, vocab):
        self.vocab = vocab
        
        # The CTC loss function requires idx=0 to be reserved for the blank character, so indexing starts from 1
        self.offset = 1

        # Initialize an index for each char in our vocabulary, with the offset
        self.label_mapping = {}
        for i, v in enumerate(vocab):
            self.label_mapping[v] = i + self.offset
    
    def encode(self, label):
        encodings = []
        for char in label:
            encodings.append(self.label_mapping[char])
        
        return torch.IntTensor(encodings)

    def decode(self, labels_vector):
        txt_label = ""

        for v in labels_vector.tolist():
            if v == 0: # Skip CTC blank characters
                continue
            char = self.vocab[v - self.offset]
            txt_label += char
        
        return txt_label





