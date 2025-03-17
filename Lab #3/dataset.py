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


# class WordDictionary:
#     def __init__(self):
#         self.index2word = {0: " "}
#         self.word2index = {letter: idx for idx, letter in self.index2word.items()}
#         self.word2count = {letter: 0 for idx, letter in self.index2word.items()}
#         self.word_index = len(self.index2word)
#         self.delimiter = " "
#
#     def add_line(self, sentence):
#         # words = re.sub(r"([^a-zA-Zа-яА-Яеё])", r" \1 ", sentence.lower().strip())
#         words = re.sub(r'(?<=[а-яА-Яa-zA-ZёЁ])([.,!?;:])', r' \1', sentence.lower().strip())
#         for i, word in enumerate(words.split()):
#             if re.match(r"^[а-яА-Яa-zA-ZёЁ\-]+$", word) and i > 0:
#                 self.add_word(" ")
#             self.add_word(word)
#         self.add_word("\n")
#
#     def add_word(self, word):
#         if word not in self.word2index:
#             self.word2index[word] = self.word_index
#             self.word2count[word] = 1
#             self.index2word[self.word_index] = word
#             self.word_index += 1
#         else:
#             self.word2count[word] += 1
#
#     def add_data(self, path):
#         with open(path, 'r', encoding='utf-8') as file:
#             for line in file:
#                 self.add_line(line)
#
#     def save_dict(self, path="dict.txt"):
#         with open(path, "w", encoding='utf-8') as file:
#             for idx, letter in self.index2word.items():
#                 file.write(f'{idx}{self.delimiter}{letter}\n')
#
#     def convert_line_to_index(self, line: str) -> list:
#         encoded_line = []
#         words = re.sub(r'(?<=[а-яА-Яa-zA-ZёЁ])([.,!?;:])', r' \1', line.lower().strip())
#         for i, word in enumerate(words.split()):
#             if re.match(r"^[а-яА-Яa-zA-ZёЁ\-]+$", word) and i > 0:
#                 encoded_line.append(self.word2index.get(" ", 0))
#             encoded_line.append(self.word2index.get(word, 0))
#         encoded_line.append(self.word2index["\n"])
#         return encoded_line
#
#     def __len__(self):
#         return len(self.index2word)
#
#     def count_words(self):
#         count = 0
#         for _, words in self.word2count.items():
#             count += words
#         return count


class WordDictionary:
    def __init__(self):
        self.index2word = {0: "<UNK>", 1: "<BOS>", 2: "<EOS>", 3:"<SEP>"}
        self.word2index = {letter: idx for idx, letter in self.index2word.items()}
        self.word2count = {letter: 0 for idx, letter in self.index2word.items()}
        self.word_index = len(self.index2word)
        self.delimiter = " "

    def add_line(self, sentence):
        # words = re.sub(r"([^a-zA-Zа-яА-Яеё])", r" \1 ", sentence.lower().strip())
        new_sentence = sentence.lower().strip()
        if new_sentence.startswith("["):
            self.add_word(new_sentence)
        else:
            words = re.sub(r'(?<=[а-яА-Яa-zA-ZёЁ])([.,!?;:])', r' \1', new_sentence)
            for word in words.split():
                if word.upper() in self.word2index:
                    word = word.upper()
                self.add_word(word)
        self.add_word("<SEP>")

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

    def convert_line_to_index(self, line: str) -> list:
        new_line = line.lower().strip()
        encoded_line = []
        if new_line.startswith("["):
            encoded_line.append(self.word2index.get(new_line, 0))
        else:
            words = re.sub(r'(?<=[а-яА-Яa-zA-ZёЁ])([.,!?;:])', r' \1', new_line)
            for word in words.split():
                if word.upper() in self.word2index:
                    word = word.upper()
                encoded_line.append(self.word2index.get(word, 0))
        encoded_line.append(self.word2index.get("<SEP>", 0))
        return encoded_line

    def __len__(self):
        return len(self.index2word)

    def count_words(self):
        count = 0
        for _, words in self.word2count.items():
            count += words
        return count



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