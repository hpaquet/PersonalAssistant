from alice.brain.training import train
from alice.brain.brain import Brain
from alice.brain.interpret import Interpreter


class Alice:

    def __init__(self):
        self.name = "Alice"
        self.brain = Brain()

        self.interpreter = Interpreter()

    def listen(self):
        print(f"Bonjour, mon nom est {self.name}, comment puis-je vous aider ?\n")
        print("utilisé la cmd quit pour terminer l'application")

        while True:
            sentence = input(">> ")

            if sentence == 'quit':
                print("Au revoir !")
                break

            answer = self.interpreter.interpret(sentence)

            if answer:
                print(f"{self.name}: {answer}")
            else:
                print(f"{self.name}: Je n'ai pas bien comprit ...")

    def learn(self):
        pass