import re
import string

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class LetterDictionary:
    def __init__(self):
        self.index2letter = {0: " "}
        self.letter2index = {letter: idx for idx, letter in self.index2letter.items()}
        self.letter2count = {letter: 0 for idx, letter in self.index2letter.items()}
        self.n_letters = len(self.index2letter)
        self.delimiter = " "

    def add_line(self, sentence):
        for letter in sentence.lower().strip():
            self.add_letter(letter)

    def add_letter(self, letter):
        if letter not in self.letter2index:
            self.letter2index[letter] = self.n_letters
            self.letter2count[letter] = 1
            self.index2letter[self.n_letters] = letter
            self.n_letters += 1
        else:
            self.letter2count[letter] += 1

    def add_data(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                self.add_line(line)

    def save_dict(self, path="dict.txt"):
        with open(path, "w", encoding='utf-8') as file:
            for idx, letter in self.index2letter.items():
                file.write(f'{idx}{self.delimiter}{letter}\n')

    def load_dict(self, path="dict.txt"):
        with open(path, "r", encoding='utf-8') as file:
            for line in file:
                print(line)
                key_value = line.strip().split(self.delimiter)
                if len(key_value) == 2:
                    self.index2letter[int(key_value[0])] = key_value[1]

    def convert_line_to_index(self, line: str) -> list:
        return [self.letter2index.get(letter, 0) for letter in line.lower()]

    def __len__(self):
        return len(self.index2letter)


class WordDictionary:
    def __init__(self):
        self.index2word = {0: " "}
        self.word2index = {letter: idx for idx, letter in self.index2word.items()}
        self.word2count = {letter: 0 for idx, letter in self.index2word.items()}
        self.word_index = len(self.index2word)
        self.delimiter = " "

    def add_line(self, sentence):
        words = re.sub(r"([^a-zA-Zа-яА-Я])", r" \1 ", sentence.lower().strip())
        for word in words.split():
            self.add_word(word)
            self.add_word("\n")

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.word_index
            self.word2count[word] = 1
            self.index2word[self.word_index] = word
            self.word_index += 1
        else:
            self.word2count[word] += 1

    def add_data(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                self.add_line(line)

    def save_dict(self, path="dict.txt"):
        with open(path, "w", encoding='utf-8') as file:
            for idx, letter in self.index2word.items():
                file.write(f'{idx}{self.delimiter}{letter}\n')

    # def load_dict(self, path="dict.txt"):
    #     with open(path, "r", encoding='utf-8') as file:
    #         for line in file:
    #             print(line)
    #             key_value = line.strip().split(self.delimiter)
    #             if len(key_value) == 2:
    #                 self.index2word[int(key_value[0])] = key_value[1]

    def convert_line_to_index(self, line: str) -> list:
        encoded_line = []
        words = re.sub(r"([^a-zA-Zа-яА-Я\[\]])", r" \1 ", line.lower().strip())
        for word in words.split():
            if word in string.punctuation and len(encoded_line) > 0:
                encoded_line.pop()
            encoded_line.append(self.word2index.get(word, 0))
            encoded_line.append(0)
        encoded_line.append(self.word2index["\n"])
        return encoded_line

    def __len__(self):
        return len(self.index2word)

    def count_words(self):
        count = 0
        for _, words in self.word2count.items():
            count += words
        return count


# class MyDataset(Dataset):
#     def __init__(self, path, dictionary, seq_length, step=1):
#         self.path = path
#         self.dictionary = dictionary
#         self.seq_length = seq_length
#         self.step = step
#
#         self.len = self.__count_len()
#         self.file = open(self.path, "r", encoding='utf-8')
#
#     def __count_len(self):
#         size = 0
#         with open(self.path, "r", encoding='utf-8') as file:
#             for line in file:
#                 size += len(line)
#         return size
#
#     def __len__(self):
#         return self.len
#
#     def __getitem__(self, idx):
#         if idx == 0:
#             self.file.close()
#             self.file = open(self.path, "r", encoding='utf-8')
#
#         x_line = self.file.read(self.seq_length)
#         y_line = x_line[self.step:] + self.file.read(self.step)
#         x = self.dictionary.convert_line_to_index(x_line)
#         y = self.dictionary.convert_line_to_index(y_line)
#         # x = self.data[idx:idx + self.seq_length]
#         # y = self.data[idx + 1:idx + self.seq_length + 1]
#         x = F.pad(torch.tensor(x, dtype=torch.long), (0, self.seq_length - len(x)), value=0)
#         y = F.pad(torch.tensor(y, dtype=torch.long), (0, self.seq_length - len(y)), value=0)
#         print(idx, len(x), len(y))
#         return x, y

class MyDataset(Dataset):
    def __init__(self, path, dictionary, seq_length, step=1):
        self.dictionary = dictionary
        self.data = self.__load_data(path)
        self.seq_length = seq_length
        self.step = step

    def __load_data(self, path):
        data = []
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                data += self.dictionary.convert_line_to_index(line)
        return data

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

if __name__ == '__main__':
    dictionary = LetterDictionary()
    dictionary.add_data("data.txt")
    dictionary.save_dict("dict.txt")
    dictionary.load_dict("dict.txt")