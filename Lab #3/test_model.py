import numpy as np
import torch
from dataset import LetterDictionary, WordDictionary
from torch import nn
from model import MyModel, initMyModel

if __name__ == '__main__':
    LENGTH = 100

    dictionary = WordDictionary()
    # dictionary.add_data("WAP_DATA.txt")
    dictionary.add_data("data/file.txt")
    print(dictionary.count_words())

    seq_length = 10
    model = initMyModel(len(dictionary))
    model.load_state_dict(torch.load('GRU_weights.pth'))
    model.eval()

    while True:
        start_word = input("\n Начать с: ")
        answer = [dictionary.word2index.get(ch, 0) for ch in start_word]
        hidden = None
        for _ in range(LENGTH):
            x = torch.tensor([answer[-seq_length:]], dtype=torch.long)
            with torch.no_grad():
                output, hidden = model(x, hidden)
            prob = torch.softmax(output[0, -1], dim=-1).numpy()
            next_char = np.random.choice(len(prob), p=prob)
            answer.append(next_char)

        print(''.join([dictionary.index2word[ch] for ch in answer]))