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
from LSTM_Model import CharLSTM
import platform
import math

"""
Log Perplexity:
Ranks:
Canaries: 
Format: 
Randomnessspace R
Guessing Entropy:
"""

class ExposureMetric:

    def __init__(self, *args, **kwargs):

        self.log_perplexity = None
        self.rankList = None
        self.canaries = None
        self.format = None
        self.randomnessSpaceR = None
        self.guessingEntropy = None
        self.model = None
    
    def train_LLM(self):

        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
                print("MPS device not available, using CPU instead.")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("CUDA device not available, using CPU instead.")
        
        print(f"Verwendetes Device: {device}")

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

        self.model = CharLSTM(n_chars, embedding_dim, hidden_units, n_layers, n_chars)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 3. Train the Model
        print("Training on PTB Raw Text Dataset...")
        for epoch in range(epochs):
            self.model.train()
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
                hidden = self.model.init_hidden(batch_in.size(0)) # Initialize hidden state for the batch
                output, hidden = self.model(batch_in, hidden)
                loss = criterion(output, batch_out)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5) # Gradient clipping
                optimizer.step()
                total_loss += loss.item()

            # Evaluate on validation set
            self.model.eval()
            val_loss = 0
            val_batches = len(val_input) // batch_size
            with torch.no_grad():
                for i in range(val_batches):
                    batch_in_val = val_input[i * batch_size:(i + 1) * batch_size]
                    batch_out_val = val_target[i * batch_size:(i + 1) * batch_size]
                    hidden_val = self.model.init_hidden(batch_in_val.size(0))
                    output_val, _ = self.model(batch_in_val, hidden_val)
                    loss_val = criterion(output_val, batch_out_val)
                    val_loss += loss_val.item()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/n_batches:.4f}, Val Loss: {val_loss/val_batches:.4f}')

        print("Training on PTB Raw Text Dataset Finished.")

        # Example of getting logits for a sequence
        sample_sequence = text[:seq_length]
        logits = self.get_logits(sample_sequence)
        print(logits)
        print(f"Logits for sequence '{sample_sequence}': {logits.shape}, {logits}")
        torch.save(self.model.state_dict(), '.')

        # You can now use these 'logits' for further calculations within
        # the ExposureMetric class or elsewhere.
        # Next step: Augment with the canary sentence and continue training or evaluate.

    def get_logits(self, input_sequence):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Erstelle ein Mapping der eindeutigen Zeichen in der Eingabesequenz zu Indizes
            unique_chars = sorted(list(set(input_sequence)))
            char_to_index_local = {char: i for i, char in enumerate(unique_chars)}

            # Konvertiere die Eingabesequenz in numerische Indizes basierend auf dem lokalen Mapping
            input_tensor = torch.tensor([[char_to_index_local[char] for char in input_sequence]], dtype=torch.long).to(self.model.fc.weight.device)

            hidden = self.model.init_hidden(input_tensor.size(0))
            output, _ = self.model(input_tensor, hidden)
            # 'output' hat die Form (batch_size, seq_length, output_size)
            # Wir wollen die Logits für das nächste Zeichen *nach* der Sequenz,
            # was der Output am letzten Zeitschritt ist.

            # Da unser Modell mit einem festen Vokabular trainiert wurde,
            # müssen wir die Ausgabe des Modells auf die Größe dieses Vokabulars abbilden.
            # Die Größe des Ausgaberaums des Modells (output_size) entspricht der Anzahl
            # der eindeutigen Zeichen im Trainingsdatensatz.

            # Die Logits, die das Modell ausgibt, beziehen sich auf die Wahrscheinlichkeiten
            # des *nächsten* Zeichens im Vokabular des Trainingsdatensatzes.

            logits = output[-1, :].unsqueeze(0) # Füge die Batch-Dimension wieder hinzu
            return logits[0]
        
    #Die logits werden hier mit softmax umgewandelt und das ist die conditional probability von allen zeichen auf grundlage des bsiherigen modells
    def calculatePerplexity(self, sequence: str, device):

        #goal: calculate 
        self.model.eval()
        #Maybe usage of get_logits
        total_neg_logLikelihood = 0
        current_LogLikelihood = 0
        inputs = inputs.to(device) # You can move your input to gpu, torch defaults to cpu
        
        hidden = self.model.init_hidden(1)

        with torch.no_grad():

            pred = self.model(inputs, hidden)
            current_LogLikelihood = self.model.softmax(pred, dim=-1)
            total_neg_logLikelihood *= -math.log2(current_LogLikelihood)

    def calculateRankList(self):

        pass
    

#Ablauf für Secret Sharer: LLM trainieren und mit Formate zusätzlich trainieren und dann mit den generierten Tokens aus der erstellten Sequenz dann
#finetunen und mit den conditional probabilites des llm die perplexity 
def main():
    # Hier kommt die Hauptlogik deines Programms hin
    exposureMetric = ExposureMetric()
    exposureMetric.train_LLM()

if __name__ == "__main__":
    main()
