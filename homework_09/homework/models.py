from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


def get_seq_mask(input_seq_lens, max_seq_len):
    return torch.as_tensor(np.asarray(
        [[1 if j < input_seq_lens.data[i].item() else 0 for j in range(0, max_seq_len)] for i in
         range(0, input_seq_lens.shape[0])]), dtype=torch.float)  # .cuda()


class SeqPredictor:
    """
	Helper class to use within the SeqModel class to make sequential predictions.
	Not mandatory to use the class.
	However, this class can be helpful since the inputs are different during training and evaluation.
	During training, the whole sentence is passed at once to the network, while during evaluation the sequence is passed one action at a time.
	"""

    def __init__(self, model):
        self.model = model
        self.hidden = None

    def __call__(self, input):
        """
		@param input: A single input of shape (6,) indicator values (float: 0 or 1)
		@return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty)
		"""
        input1 = input.view(1, 6, 1)
        output, self.hidden = self.model(input1, self.hidden)

        return output.view(6)


class SeqModel(nn.Module):
    """
	Define your recurrent neural network here
	"""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(99, 2000)
        self.l2 = nn.Linear(2000, 6)

        self.rel = nn.ReLU()

        self.lstm = nn.LSTM(input_size=6, hidden_size=99, num_layers=1)

    def forward(self, input, hidden=None):
        """
		IMPORTANT: Do not change the function signature of the forward() function unless the grader won't work.
		@param input: A sequence of input actions (batch_size x 6 x sequence_length)
		@return The logit of a binary distribution of output actions (6 floating point values between -infty .. infty). Shape: batch_size x 6 x sequence_length
		"""
        # Initialize hidden state with zeros
        input1 = input.permute(2, 0, 1)

        output, hidden1 = self.lstm(input1, hidden)

        output = self.l1(output)

        output = self.rel(output)

        output = self.l2(output)

        output = output.permute(1, 2, 0)

        output = output.contiguous()
        output = output.view(*(input.size()))

        return output, hidden1

    def predictor(self):
        return SeqPredictor(self)
