"""
Moshe Binieli
"""

import math
import sys
from collections import Counter


def lidstone_smooth(word_frequency, lamda, training_list_size, vocabulary_size):
    # The formula to calculate the Lidstone smoothing according to the lectures.
    return (word_frequency + lamda) / (training_list_size + lamda * vocabulary_size)


def heldout_smooth(ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, the_word_freq, vocabulary_size):
    # The formula to calculate the Heldout smoothing according to the lectures.
    # The formula notations and the code represent the same thing.
    # For example |H| is the size of heldout list.
    train_words_same_freq = []

    if the_word_freq > 0:
        for w, c in ho_training_list_cnt.items():
            if c == the_word_freq:
                train_words_same_freq.append(w)
        Nr = len(train_words_same_freq)
    else:
        train_words_same_freq = set(ho_heldout_list) - set(ho_training_list_cnt)
        Nr = vocabulary_size - len(ho_training_list_cnt)

    Tr = sum(ho_heldout_list_cnt[w] for w in train_words_same_freq)
    H = len(ho_heldout_list)
    return Tr / (Nr * H)


def perplexity_lidstone(validation_list_cnt, training_list_cnt, validation_list, training_list_size, vocabulary_size, lamda):
    if lamda == 0:
        return math.inf

    probability = 0
    for word, word_valid_cnt in validation_list_cnt.items():
        frequency = training_list_cnt.get(word, 0)
        px = lidstone_smooth(frequency, lamda, training_list_size, vocabulary_size)
        probability += math.log(px, 2) * word_valid_cnt
    return calc_perplexity(probability, len(validation_list))


def perplexity_heldout(test_word_list_cnt, test_word_list, ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, vocabulary_size):
    probability = 0
    for word, word_valid_cnt in test_word_list_cnt.items():
        px = heldout_smooth(ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, ho_training_list_cnt[word], vocabulary_size)
        probability += math.log(px, 2) * word_valid_cnt
    return calc_perplexity(probability, len(test_word_list))


def calc_perplexity(probability, size):
    return math.pow(2, (probability / -size))


def stats_table(min_lambda,training_list_size,vocabulary_size,ho_training_list_cnt,ho_heldout_list,ho_heldout_list_cnt,ho_training_list_size):
    heldout_counter = Counter(ho_training_list_cnt.values())

    table = []
    for i in range(10):
        f_l = lidstone_smooth(i, min_lambda, training_list_size, vocabulary_size) * training_list_size
        f_h = (heldout_smooth(ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, i, vocabulary_size) * ho_training_list_size)

        if i > 0:
            Nr = heldout_counter[i]
        else:
            Nr = vocabulary_size - len(ho_training_list_cnt)

        t_r = int(round(Nr * f_h))
        line_i = [round(xi, 5) for xi in (i, f_l, f_h, Nr, t_r)]
        table.append(line_i)
    return table


def load_words(filename):
    word_set = []

    with open(filename) as word_stream:
        for line in word_stream:
            if "<TRAIN" not in line and "<TEST" not in line and line != "\n":
                [word_set.append(word) for word in line.split()]
    word_stream.close()

    return word_set


def write_to_file(filename, outputs, table):
    output_string = ""
    output_string += "#Student\tMoshe Binieli\n"
    output_string += "\n".join([("#Output" + str(i) + "\t" +
                                 str(o).replace('e-', 'E-')) for i, o in outputs.items()])

    output_string += '\n'.join(["\n#Output29"] + ['\t'.join(str(x)
                                                            for x in line) for line in table])

    with open(filename, "w") as fp:
        fp.write(output_string)


def main():
    # A dictionary that will store all the outputs.
    outputs = dict([(key, None) for key in range(1, 29)])

    dev_set_file_name = "dataset/develop.txt" # sys.argv[1]
    test_set_file_name = "dataset/develop.txt" # sys.argv[2]
    input_word = "honduras" # sys.argv[3]
    output_filename = "output.txt" # sys.argv[4]

    # Boolean value to enable or disable the debug code section.
    debug_code = True
    # Given in the PDF File, the vocabulary is 300,000 length.
    vocabulary_size = 300000

    train_words = load_words(dev_set_file_name)
    test_word_list = load_words(test_set_file_name)

    outputs[1] = dev_set_file_name
    outputs[2] = test_set_file_name
    outputs[3] = input_word.lower()
    outputs[4] = output_filename
    outputs[5] = vocabulary_size
    outputs[6] = 1 / vocabulary_size
    outputs[7] = len(train_words)

    # Split the word set into a training set which contains 90% of the words.
    # The validation set will contain 10% of the words.
    threshold = round(0.9 * len(train_words))
    training_list = train_words[:threshold]
    validation_list = train_words[threshold:]

    training_list_cnt = Counter(training_list)
    validation_list_cnt = Counter(validation_list)

    training_list_size = len(training_list)

    outputs[8] = len(validation_list)
    outputs[9] = training_list_size
    outputs[10] = len(training_list_cnt)
    outputs[11] = training_list_cnt[input_word]
    outputs[12] = training_list_cnt[input_word] / training_list_size
    # "unseen-word" will refer to a word that never appears in the dataset.
    outputs[13] = training_list_cnt['unseen-word'] / training_list_size
    outputs[14] = lidstone_smooth(training_list_cnt[input_word], 0.1, training_list_size, vocabulary_size)
    # "unseen-word" will refer to a word that never appears in the dataset.
    outputs[15] = lidstone_smooth(training_list_cnt['unseen-word'], 0.1, training_list_size, vocabulary_size)

    outputs[16] = perplexity_lidstone(validation_list_cnt, training_list_cnt, validation_list, training_list_size, vocabulary_size, lamda=0.01)
    outputs[17] = perplexity_lidstone(validation_list_cnt, training_list_cnt, validation_list, training_list_size, vocabulary_size, lamda=0.1)
    outputs[18] = perplexity_lidstone(validation_list_cnt, training_list_cnt, validation_list, training_list_size, vocabulary_size, lamda=1.0)

    perplexities = [(perplexity_lidstone(validation_list_cnt,training_list_cnt,validation_list,training_list_size,vocabulary_size,lamda=L / 100,), L / 100,) for L in range(0, 201)]

    min_preplexity, min_lambda = min(perplexities)
    outputs[19] = min_lambda
    outputs[20] = min_preplexity

    # Split the training set and heldout list to half (50%) and half(50%).
    threshold = round(0.5 * len(train_words))
    ho_training_list = train_words[:threshold]
    ho_heldout_list = train_words[threshold:]
    ho_training_list_cnt = Counter(ho_training_list)
    ho_heldout_list_cnt = Counter(ho_heldout_list)
    ho_training_list_size = len(ho_training_list)

    outputs[21] = len(ho_training_list)
    outputs[22] = len(ho_heldout_list)

    outputs[23] = heldout_smooth(ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, ho_training_list_cnt[input_word], vocabulary_size)
    outputs[24] = heldout_smooth(ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, 0, vocabulary_size)

    if debug_code:
        total_words_cnt = Counter(train_words)
        unseen_events_size = vocabulary_size - len(total_words_cnt)
        debug_validation_list = ["word1", "word1", "word2"]

        s1 = sum(lidstone_smooth(total_words_cnt[w], min_lambda, len(train_words), vocabulary_size) for w in total_words_cnt)
        lidston_debug_check = (lidstone_smooth(total_words_cnt['unseen-word'], min_lambda, len(train_words), vocabulary_size) * unseen_events_size + s1)

        s2 = sum(heldout_smooth(total_words_cnt, Counter(debug_validation_list), debug_validation_list, total_words_cnt[w], vocabulary_size,)for w in total_words_cnt)
        heldout_debug_check = (heldout_smooth(total_words_cnt, Counter(debug_validation_list), debug_validation_list, 0, vocabulary_size) * unseen_events_size + s2)

        if round(lidston_debug_check, 4) != 1.0:
            print("Lidstone doesn't sum up to 1, it sums up to " + str(lidston_debug_check))
        else:
            print("Lidstone sum up to 1.")

        if round(heldout_debug_check, 4) != 1.0:
            print("Heldout doesn't sum up to 1, it sums up to " + str(heldout_debug_check))
        else:
            print("Heldout sum up to 1.")
    else:
        print("Debug checks have been skipped.")

    test_word_list_cnt = Counter(test_word_list)
    outputs[25] = len(test_word_list)

    lidstone_test_perplexity = perplexity_lidstone(test_word_list_cnt, training_list_cnt, test_word_list, training_list_size, vocabulary_size, lamda=min_lambda)
    heldout_test_perplexity = perplexity_heldout(test_word_list_cnt, test_word_list, ho_training_list_cnt, ho_heldout_list_cnt, ho_heldout_list, vocabulary_size)
    outputs[26] = lidstone_test_perplexity
    outputs[27] = heldout_test_perplexity
    outputs[28] = 'L' if lidstone_test_perplexity < heldout_test_perplexity else 'H'

    table = stats_table(min_lambda,training_list_size,vocabulary_size,ho_training_list_cnt,ho_heldout_list,ho_heldout_list_cnt,ho_training_list_size,)
    write_to_file(output_filename, outputs, table)


if __name__ == "__main__":
    main()
