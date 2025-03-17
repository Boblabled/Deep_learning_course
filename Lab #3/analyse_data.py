from dataset import WordDictionary
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dictionary = WordDictionary()
    # dictionary.add_data("WAP_DATA.txt")
    dictionary.add_data("KISH_DATA.txt")
    # print(dictionary.word2count)
    frec = dict(sorted(dictionary.word2count.items(), key=lambda item: item[1]))
    print(frec)
    print("Колличество слов: ", len(dictionary))
    plt.plot(range(len(frec)), list(frec.values()))
    plt.grid(True)
    plt.show()

