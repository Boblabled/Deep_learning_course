import re

import numpy as np
import torch
from dataset import LetterDictionary, WordDictionary
from torch import nn
from model import MyModel, initMyModel

if __name__ == '__main__':
    MAX_LENGTH = 1000

    dictionary = WordDictionary()
    # dictionary.add_data("WAP_DATA.txt")
    dictionary.add_data("data/file.txt")
    print(dictionary.count_words())

    seq_length = 10
    model = initMyModel(len(dictionary))
    model.load_state_dict(torch.load('GRU_weights.pth'))
    model.eval()

    # dictionary.index2word[1] = "\n"
    # dictionary.index2word[2] = "\n"
    dictionary.index2word[3] = "\n"

    i = 1
    while True:
        # start_word = input("\n Начать с: ")
        print(f"============================================[ Трек #{i} ]=======================================")
        start_word = "<BOS>"
        answer = [dictionary.word2index.get(start_word, 0)]
        hidden = None
        # for _ in range(LENGTH):
        #     x = torch.tensor([answer[-seq_length:]], dtype=torch.long)
        #     with torch.no_grad():
        #         output, hidden = model(x, hidden)
        #     prob = torch.softmax(output[0, -1], dim=-1).numpy()
        #     next_char = np.random.choice(len(prob), p=prob)
        #     answer.append(next_char)

        length = 0
        while dictionary.index2word[answer[-1]] != "<EOS>" or length > MAX_LENGTH:
            x = torch.tensor([answer[-seq_length:]], dtype=torch.long)
            with torch.no_grad():
                output, hidden = model(x, hidden)
            prob = torch.softmax(output[0, -1], dim=-1).numpy()
            next_word = np.random.choice(len(prob), p=prob)
            answer.append(next_word)
            length += 1

        answer =  re.sub(r'\s([,.!?])', r'\1', ' '.join([dictionary.index2word[ch] for ch in answer]))

        print(answer)
        print(f"=================================[ Длинна текста: {length} слов ]=======================================\n")
        i += 1
        input()