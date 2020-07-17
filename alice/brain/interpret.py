import torch
import json
import random

from alice.brain.utility import tokenize, bag_of_world
from alice.brain.neural_network import NeuralNetwork


class Interpreter:

    def __init__(self):
        self.device = None
        self.all_words = None
        self.tags = None
        self.intents = None

        self.get_model()

    def get_model(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open('../alice/brain/intents.json', 'r') as json_data:
            self.intents = json.load(json_data)

        FILE = "../alice/brain/data.pth"
        data = torch.load(FILE)

        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']

        self.all_words = data['all_words']
        self.tags = data['tags']
        model_state = data['model_state']

        self.model = NeuralNetwork(input_size, hidden_size, output_size)
        self.model.load_state_dict(model_state)
        self.model.eval()

    def interpret(self, sentence):
        sentence = tokenize(sentence)

        x = bag_of_world(sentence, self.all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x).to(self.device)

        output = self.model(x)
        _, predicted = torch.max(output, dim=1)

        tag = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in self.intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(self.intent['responses'])

        return None
