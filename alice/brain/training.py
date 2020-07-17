import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from alice.brain.utility import tokenize, stem , bag_of_world
from alice.brain.neural_network import NeuralNetwork


class TrainingDataset(Dataset):
    def __init__(self, x_train, y_train):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


def train():

    num_epochs = 1000
    batch_size = 8
    learning_rate = 0.001
    hidden_size = 8

    with open('../alice/brain/intents.json', 'r') as f:
        intents = json.load(f)

    ignore_words = ['?', ',', '.', '!']
    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)

        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_word = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train, y_train = [], []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_world(pattern_sentence, all_words)
        x_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    input_size = len(x_train[0])
    output_size = len(tags)

    print(input_size, output_size)

    dataset = TrainingDataset(x_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(device)

            outputs = model(words)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f"final loss: {loss.item():.4f}")

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags,
    }

    FILE = "../alice/brain/data.pth"
    torch.save(data, FILE)

    print(f"training complete. file saved to {FILE}")


train()