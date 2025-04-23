"""
Implementations for:
"The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks"

This class will approximate exposure of different llms and visualize the results

import numpy as np
from scipy.stats import entropy
base = 2  # work in units of bits
qk = np.array([9/10, 1/10])  # biased coin
H = entropy(qk, base=base)
print(H)
H == -np.sum(qk * np.log(qk)) / np.log(base)
print(H)

Data Classes / Classes and functions
Log Perplexity:
Ranks:
Canaries: 
Format: 
Randomnessspace R
Guessing Entropy:

Grundlage der Metrik: 

Generative Sequence Models
A generative sequence model is a fundamental architecture
for common tasks such as language-modeling [4], translation
[3], dialogue systems, caption generation, optical character
recognition, and automatic speech recognition, among others.
For example, consider the task of modeling naturallanguage English text from the space of all possible sequences
of English words. For this purpose, a generative sequence
model would assign probabilities to words based on the context in which those words appeared in the empirical distribution of the model’s training data. For example, the model
might assign the token “lamb” a high probability after seeing
the sequence of words “Mary had a little”, and the token “the”
a low probability because—although “the” is a very common
word—this prefix of words requires a noun to come next, to
fit the distribution of natural, valid English.
Formally, generative sequence models are designed to generate a sequence of tokens x1...xn according to an (unknown)
distribution Pr(x1...xn). Generative sequence models estimate
this distribution, which can be decomposed through Bayes’
rule as Pr(x1...xn) = Πn
i=1Pr(xi
|x1...xi−1). Each individual
computation Pr(xi
|x1...xi−1) represents the probability of token xi occurring at timestep i with previous tokens x1 to xi−1.
Modern generative sequence models most frequently employ neural networks to estimate each conditional distribution.
To do this, a neural network is trained (using gradient descent to update the neural-network weights θ) to output the
conditional probability distribution over output tokens, given
input tokens x1 to xi−1, that maximizes the likelihood of the
training-data text corpus. For such models, Pr(xi
|x1...xi−1)
is defined as the probability of the token xi as returned by
evaluating the neural network fθ(x1...xi−1).
Neural-network generative sequence models most often
use model architectures that can be naturally evaluated on
variable-length inputs, such as Recurrent Neural Networks
(RNNs). RNNs are evaluated using a current token (e.g., word
or character) and a current state, and output a predicted next
token as well as an updated state. By processing input tokens
one at a time, RNNs can thereby process arbitrary-si
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import treebank
import pandas as pd
import random
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"There are {torch.cuda.device_count()} GPU(s) available.")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("No GPU available, training on CPU.")

# Now, 'device' variable holds either 'cuda' or 'cpu'

# Download necessary NLTK data if not already present
nltk.download('treebank')
nltk.download('punkt')
nltk.download('universal_tagset') # Optional, but good to have

# 1. Prepare the PTB Data (using a similar approach to your Pandas snippet for raw text)
def load_ptb_raw_text():
    text = ""
    for fileid in treebank.fileids():
        words = treebank.words(fileid)
        text += ' '.join(words).lower() + ' '
    return text.strip()

text = load_ptb_raw_text()
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for char, i in char_to_index.items()}
n_chars = len(chars)

seq_length = 10
embedding_dim = 50
hidden_units = 200
n_layers = 2
learning_rate = 0.0001
epochs = 10 # Adjust as needed
batch_size = 32
val_fraction = 0.1

def text_to_sequences(text, char_to_index, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(text) - seq_length):
        seq_in = text[i:i + seq_length]
        seq_out = text[i + seq_length]
        dataX.append([char_to_index[char] for char in seq_in])
        dataY.append(char_to_index[seq_out])
    return torch.tensor(dataX, dtype=torch.long), torch.tensor(dataY, dtype=torch.long)

input_seq, target_seq = text_to_sequences(text, char_to_index, seq_length)

# Split into training and validation sets
val_size = int(len(input_seq) * val_fraction)
train_input = input_seq[:-val_size]
train_target = target_seq[:-val_size]
val_input = input_seq[-val_size:]
val_target = target_seq[-val_size:]

# 2. Define the LSTM Model (same as before)
class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_units, n_layers, output_size):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, output_size)
        self.num_layers = n_layers
        self.hidden_units = hidden_units

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = output[:, -1, :] # Take the output of the last time step
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_units),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_units))

model = CharLSTM(n_chars, embedding_dim, hidden_units, n_layers, n_chars)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 3. Train the Model
print("Training on PTB Raw Text Dataset...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    n_batches = len(train_input) // batch_size
    total_loss = 0

    # Shuffle the training data
    permutation = torch.randperm(train_input.size(0))
    train_input_shuffled = train_input[permutation]
    train_target_shuffled = train_target[permutation]

    for i in range(n_batches):
        batch_in = train_input_shuffled[i * batch_size:(i + 1) * batch_size]
        batch_out = train_target_shuffled[i * batch_size:(i + 1) * batch_size]
        hidden = model.init_hidden(batch_in.size(0)) # Initialize hidden state for the batch
        output, hidden = model(batch_in, hidden)
        loss = criterion(output, batch_out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # Gradient clipping
        optimizer.step()
        total_loss += loss.item()

    # Evaluate on validation set
    model.eval()
    val_loss = 0
    val_batches = len(val_input) // batch_size
    with torch.no_grad():
        for i in range(val_batches):
            batch_in_val = val_input[i * batch_size:(i + 1) * batch_size]
            batch_out_val = val_target[i * batch_size:(i + 1) * batch_size]
            hidden_val = model.init_hidden(batch_in_val.size(0))
            output_val, _ = model(batch_in_val, hidden_val)
            loss_val = criterion(output_val, batch_out_val)
            val_loss += loss_val.item()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/n_batches:.4f}, Val Loss: {val_loss/val_batches:.4f}')

print("Training on PTB Raw Text Dataset Finished.")

# Next step: Augment with the canary sentence and continue training or evaluate.