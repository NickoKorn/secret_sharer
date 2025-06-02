"""
Implementations for:
"The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks"
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
from canariesFabric import Canaries
import os
import torch.nn.functional as F

"""
Log Perplexity:
Ranks:
Canaries: 
Format: 
Randomnessspace R
Guessing Entropy:
"""

#Besser toDo: ExposureMetric als Data class und das training, validieren und testen innerhalb einer pipeline, die einfach 

#@dataclass(unsafe_hash=True)
class ExposureMetric:

    def __init__(self, *args, **kwargs):

        self.log_perplexity = None
        self.rankList = None
        self.guessingEntropy = None
        self.model = None
        self.device = None
        self.char_to_index = None
        self.chars = None
        self.text = None
        self.n_chars = None

    def chooseDevice(self):

        if platform.system() == "Darwin":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                print("MPS device not available, using CPU instead.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("CUDA device not available, using CPU instead.")
        
        print(f"Verwendetes Device: {self.device}")

    def loadPTBDataset(self, trainFormat: str):

        # Download necessary NLTK data if not already present
        nltk.download('treebank')
        nltk.download('punkt')
        nltk.download('universal_tagset') # Optional, but good to have

        # 1. Prepare the PTB Data (using a similar approach to your Pandas snippet for raw text)
        def load_ptb_raw_text():
            text = ""
            #In the beginning
            text+= ''.join(trainFormat).lower() + ' '
            for fileid in treebank.fileids():
                words = treebank.words(fileid)
                text += ' '.join(words).lower() + ' '
            return text.strip()

        self.text = load_ptb_raw_text()
        #In the end
        for i in range (0, 10, 1):
            self.text+= ''.join(trainFormat).lower() + ' '
        print(self.text)
        #print(type(self.text))

        self.chars = sorted(list(set(self.text)))
        self.char_to_index = {char: i for i, char in enumerate(self.chars)}
        self.n_chars = len(self.chars)
        #print('self_chars: ')
        #print(self.n_chars)

    def text_to_sequences_FastSlicing(self, text, seq_length):

            #print(len(text))
            #List compehension slicing
            dataX = [[self.char_to_index[char] for char in row] for row in [text[i:i+seq_length] for i in range(0, len(text)-seq_length)]]     
            dataY = [self.char_to_index[text[i+seq_length]] for i in range(0, len(text)-seq_length)]
            #dataX = [[char_to_index[char] for char in row] for row in dataX]
            
            #print(dataY[0]),#'\n',print(dataY)
            #print(dataY[1]),#'\n',print(dataY)
            #print(dataX[-2:]),#'\n',print(dataY)

            #print('DataX: ')
            #print(dataX)#'\n',print(dataY)
            return torch.tensor(dataX, dtype=torch.long), torch.tensor(dataY, dtype=torch.long)

    def train_LLM(self, trainFormat: str, numberOfCanaryInsertion: int, pushOrPopCanary: str, shuffle: int):

        self.chooseDevice()
        self.loadPTBDataset(trainFormat)
        index_to_char = {i: char for char, i in self.char_to_index.items()}
        #print(self.n_chars)

        seq_length = 5
        embedding_dim = 50
        hidden_units = 200
        n_layers = 2
        learning_rate = 0.0001
        epochs = 10 # Adjust as needed
        batch_size = 32
        val_fraction = 0.1

        input_seq, target_seq = self.text_to_sequences_FastSlicing(self.text, seq_length)

        # Split into training and validation sets
        val_size = int(len(input_seq) * val_fraction)
        train_input = input_seq[:-val_size]
        train_target = target_seq[:-val_size]
        val_input = input_seq[-val_size:]
        val_target = target_seq[-val_size:]

        self.model = CharLSTM(self.n_chars, embedding_dim, hidden_units, n_layers, self.n_chars)
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 3. Train the Model
        print("Training on PTB Raw Text Dataset...")
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            n_batches = len(train_input) // batch_size
            #print(n_batches)
            total_loss = 0

            # Shuffle the training data
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #if(shuffle = )
            permutation = torch.randperm(train_input.size(0))
            train_input_shuffled = train_input[permutation]
            #print('Traininput')
            #print(train_input)
            #print(len(train_input))
            #print(train_target)
            #print(train_target)
            #print(len(train_target))

            train_target_shuffled = train_target[permutation]
            #print('train_target_shuffled size: ')
            #print(len(train_target_shuffled))
            for i in range(n_batches):

                batch = train_input[i * batch_size:(i + 1) * batch_size]
                batch_in = train_input_shuffled[i * batch_size:(i + 1) * batch_size]
                batch_out = train_target_shuffled[i * batch_size:(i + 1) * batch_size]

                batch_in.to(self.device)
                batch_out.to(self.device)
                hidden = self.model.init_hidden(batch_in.size(0), self.device) # Initialize hidden state for the batch
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
                    
                    batch_in_val.to(self.device)
                    batch_out_val.to(self.device)

                    hidden_val = self.model.init_hidden(batch_in_val.size(0), self.device)
                    output_val, _ = self.model(batch_in_val, hidden_val)
                    loss_val = criterion(output_val, batch_out_val)
                    val_loss += loss_val.item()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/n_batches:.4f}, Val Loss: {val_loss/val_batches:.4f}')

        print("Training on PTB Raw Text Dataset Finished.")

        # Example of getting logits for a sequence
        sample_sequence = self.text[:seq_length]
        logits = self.get_logits(sample_sequence)
        print(logits)
        print(f"Logits for sequence '{sample_sequence}': {logits.shape}, {logits}")
        
        #Linux:
        torch.save(self.model.state_dict(), 'charLSTM.pth')

    def trainWithCanaries(self):
        
        #model = CharLSTM()
        self.model.load_state_dict(torch.load('charLSTM.pth', weights_only=True))
        pass

    def get_logits(self, input_sequence):
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            # Erstelle ein Mapping der eindeutigen Zeichen in der Eingabesequenz zu Indizes
            unique_chars = sorted(list(set(input_sequence)))
            char_to_index_local = {char: i for i, char in enumerate(unique_chars)}
            # Konvertiere die Eingabesequenz in numerische Indizes basierend auf dem lokalen Mapping

            input_tensor = torch.tensor([[char_to_index_local[char] for char in input_sequence]], dtype=torch.long).to(self.model.fc.weight.device)

            hidden = self.model.init_hidden(input_tensor.size(0), self.device)
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
    def calculatePerplexity(self, trainFormat: str, testFormats: list):

        print(trainFormat)
        self.chooseDevice()
        self.loadPTBDataset(trainFormat)

        print(f"Verwendetes Device: {self.device}")

        embedding_dim = 50
        hidden_units = 200
        n_layers = 2

        print(testFormats)
            
        self.model = CharLSTM(self.n_chars, embedding_dim, hidden_units, n_layers, self.n_chars)
        hidden = self.model.init_hidden(26, self.device)
        self.model.load_state_dict(torch.load('charLSTM.pth', weights_only=True))
        self.model.to(self.device)
        print('Ist auf dem Device')

        #goal: calculate 
        batch_size = 26
        self.model.eval()
        #26 Batches

        sequencesX = []
        sequencesY = []
        sequenceX, sequenceY = self.text_to_sequences_FastSlicing(trainFormat.lower(), seq_length=5)
        for i in range(len(testFormats)):

            print(i)
            print(testFormats[i])
            sequenceXTest, sequenceYTest = self.text_to_sequences_FastSlicing(testFormats[i].lower(), seq_length=5)
            sequencesX.append(sequenceXTest)
            sequencesY.append(sequenceYTest)

        print('sequenceXList:')
        print(sequencesX)
        print('sequenceYList:')
        print(sequencesY)
        #Es braucht ein Batch, das das Trainigsformat ein mal testet und dann noch auch alle anderen, um die 
        #Log-Perplexity auszurechnen
        #Beispiel vom Trainigsformat, wenn nur The random number is: 281265017 in einem Btach ist:
        #Auffüllen mit Nullen den Rest
        i=0
        batch_in = sequenceX[i * batch_size:(i + 1) * batch_size]
        print(sequenceX)
        print(batch_in)
        print(sequenceY)

        #Training batch_in: (32,10), Test Batch_in: (21,10), Padder: (11, 10)
        padder = torch.zeros(batch_size-len(batch_in), 5)
        #print(padder.size())
        paddedTest = torch.cat([batch_in,padder], dim = 0) # Choose your desired dim
        
        #print('paddedTest:')
        #print(paddedTest)
        #print(paddedTest.size())
        batch_out = sequenceY[i * batch_size:(i + 1) * batch_size]
        #Maybe usage of get_logits
        total_neg_logLikelihood = 1
        current_LogLikelihood = 0
        #sequence = [self.char_to_index[char.lower()] for char in trainFormat]
        #print(sequence)
        batch_in.to(self.device) 
        sequenceX.to(self.device) 
        sequenceY.to(self.device)

        paddedTest = paddedTest.long()  # Or .int() if appropriate
        paddedTest.to(self.device)
        print(batch_in.size())

        print('Testformats: ')
        print(testFormats)

        ranks = dict()

        with torch.no_grad():

            output, hidden = self.model(sequenceX, hidden)

            # Berechne die Wahrscheinlichkeiten (optional, aber hilfreich zum Verständnis)
            probabilities = F.softmax(output, dim=-1)

            # Finde den Index des Tokens mit der höchsten Wahrscheinlichkeit
            _, predicted_token_index = torch.max(probabilities, dim=-1)

            print('Output (Logits):')
            print(output)
            print('Wahrscheinlichkeiten:')
            print(probabilities.size())
            print(probabilities)
            print('Index des vorhergesagten Tokens (mit höchster Wahrscheinlichkeit):')
            print(predicted_token_index)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(output, sequenceY)
            print('loss:')
            print(loss)

            # Korrekte Akkumulation des Losses für die Perplexitätsberechnung
            # total_neg_logLikelihood += loss.item()

            ranks[trainFormat]=loss

            #Now calculating for every other format, trainingFormat should be the highest rank.

            for i in range(len(testFormats)):

                output, hidden = self.model(sequencesX[i], hidden)

                # Berechne die Wahrscheinlichkeiten (optional, aber hilfreich zum Verständnis)
                probabilities = F.softmax(output, dim=-1)

                # Finde den Index des Tokens mit der höchsten Wahrscheinlichkeit
                _, predicted_token_index = torch.max(probabilities, dim=-1)

                print('Output (Logits):')
                print(output)
                print('Wahrscheinlichkeiten:')
                print(probabilities.size())
                print(probabilities)
                print('Index des vorhergesagten Tokens (mit höchster Wahrscheinlichkeit):')
                print(predicted_token_index)

                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, sequencesY[i])
                print('loss:')
                print(loss)

                # Korrekte Akkumulation des Losses für die Perplexitätsberechnung
                # total_neg_logLikelihood += loss.item()

                ranks[testFormats[i]]=loss

        print(total_neg_logLikelihood)
        
        # Sortiere die Dictionary-Items (Key-Value-Paare) nach dem Wert
        sortierte_items = sorted(ranks.items(), key=lambda item: item[1])

        # Nimm die ersten 5 Elemente (die kleinsten)
        top_5_kleinsten = sortierte_items[:5]

        # Gib die Top 5 aus
        print("Die Top 5 kleinsten Key-Value-Paare sind:")
        for key, value in top_5_kleinsten:
            print(f"Key: '{key}', Value: {value}")
    
    def calculateRankList(self):

        pass

#Ablauf für Secret Sharer: LLM trainieren und mit Formate zusätzlich trainieren und dann mit den generierten Tokens aus der erstellten Sequenz dann
#finetunen und mit den conditional probabilites des llm die perplexity 
def main():
    # Hier kommt die Hauptlogik deines Programms hin

    canary = Canaries()
    canary.randomGenerator()
    exposureMetric = ExposureMetric()
    numberOfCanaryInsertion = 3
    pushOrPopCanary = "push"
    shuffle = 0 
    print("Trainingsstart")
    #exposureMetric.train_LLM(canary.formatForTraining, numberOfCanaryInsertion, pushOrPopCanary, shuffle)
    exposureMetric.calculatePerplexity(canary.formatForTraining, canary.getFormats())

if __name__ == "__main__":
    main()
