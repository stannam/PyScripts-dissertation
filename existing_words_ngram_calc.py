# It calculates trigram prob for Sino and non-Sino stratum, respectively.
# Also, input a word (sequence of segments) and get two prob, each the likelihood for the words to be Sino or non-Sino
#
# the function strata_ngram_likelihood(word) calculates the trigram likelihood.

import csv
import os.path
import pickle
from collections import Counter
from itertools import product
from math import log, exp
from tqdm import tqdm


def load_corpus(directory="resources", filename="kor_corpus.tsv"):
    # input: directory and filename for a tsv file
    # in which two columns each for Korean word and its prescriptive etymology (sino/native/mix/foreign...)
    # output: list of dictionary objects
    path = os.path.join(directory, filename)

    corpus = []  # list to hold each words as the dictionary object
    with open(path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')

        # loop through each row in the file and append it to the list of dictionaries
        for row in reader:
            corpus.append(row)
    return corpus


def read_corpus_get_ngrams(dir='resources',  f_name='kor_corpus.tsv'):
    # load external tsv file, divide it into sino and non-sino, get ngrams and then pickle everything
    # dir: str. directory that contains a corpus file like kor_corpus.tsv
    # f_name: str. file name. e.g., kor_corpus.tsv

    # Part 1: load corpus
    corpus = load_corpus(directory=dir, filename=f_name)   # 'corpus': a list of dictionary objects
    sino = []  # list to hold sino words
    non_sino = []  # list to hold non sino words

    for word in corpus:
        if word['etymological_group'] == 'sino':
            sino.append(word['ipa'])
        else:
            non_sino.append(word['ipa'])
          
    # Part 2: calculate trigram of sino and non-sino, respectively
    trigram_counter_sino = ngram_counter(3, sino, phonemes=None, smoothing='laplace')
    trigram_counter_nonsino = ngram_counter(3, non_sino, phonemes=None, smoothing='laplace')

    # Optional: bigram of sino and non-sino respectively
    bigram_counter_sino = ngram_counter(2, sino, phonemes=None, smoothing='laplace')
    bigram_counter_nonsino = ngram_counter(2, non_sino, phonemes=None, smoothing='laplace')

  
    # Part 3: pickle the results:

    # trigram... is a tuple of Counter and dict. the Counter is the trigram counter with the occurrence of each trigram
    # the dict contains each word in the lexicon with its trigram.

    dump_ngram(trigram_counter_sino, 'sino_trigram.pkl')
    dump_ngram(bigram_counter_sino, 'sino_bigram.pkl')

    dump_ngram(trigram_counter_nonsino, 'nonsino_trigram.pkl')
    dump_ngram(bigram_counter_nonsino, 'nonsino_bigram.pkl')

    print("[INFO] Pickled files under the 'resources' directory")

def ngrams(n: int, word: str, sep=' ') -> list:
    # generate a list of ngram tuples from a single word
    # n: int. n in ngram
    # word: str. a word in the language
    # sep: str. string that separates segments

    res = []  # list. container for the return. individual ngrams will be contained here.

    word = ['<s>'] * (n-1) + [w for w in word.split(sep)] + ['<e>'] * (n-1)  # <s> is the special character for 'START' and <e> is the special character for 'END'

    for i in range(n-1, len(word)):
        ngram_token = []  # individual ngram
        for j in range(n):
            ngram_token.insert(0, word[i-j])
        res.append(tuple(ngram_token))

    return res

def pos_n(n, phonemes=None, smoothing = 'laplace'):
    # generate possible ngrams
    # n: int. the n in ngram
    # phonemes: list of str. list of phonemes in string
    if phonemes is None:
        print("no phonemes")
        return
    if smoothing != 'laplace':
        return     # smoothing other than laplace is not implemented.
    possible_ngrams = list(product(phonemes, repeat=n))

    if n == 1:
        return possible_ngrams
    else:
        n -= 1
        mgram = pos_n(n, phonemes=phonemes, smoothing = smoothing)
        for m in mgram:
            m = list(m)
            possible_ngrams.append(tuple(['<s>'] + m))
            if m[-1] != "<e>":                                     # END means END! (no sequences like s <e> <e>)
                possible_ngrams.append(tuple(list(m)+['<e>']))

    possible_ngrams = list(set(possible_ngrams))        # remove duplicates
    return possible_ngrams


def get_phonemes(lexicon, sep=" "):
    # extract the list of phonemes from words in the lexicon
    # sep: segment seperator
    phonemes = []
    for word in lexicon:
        p = list(set(word.split(sep)))
        phonemes.extend(p)
        phonemes = list(set(phonemes))
    return phonemes

def ngram_counter(n, lexicon, phonemes=None, smoothing=None):
    # generate ngram counter
    # n: int. the n in ngram
    # lexicon: list of str. list of words
    # phonemes: list of str (optional). list of phonemes in string
    # smoothing: str (optional). smoothing method. Default to None. only laplace implemented.

    ngram_list = []                 # container for lexicon level ngram tuples
    word_ngram_dict = {}            # container for word-level ngram key-value pairs. key=word, value=list of ngram tutples

    for word in lexicon:
        ngrams_res = ngrams(n, word)
        ngram_list.extend(ngrams_res)
        word_ngram_dict[word] = ngrams_res

    if smoothing == 'laplace' or smoothing == 'Laplace':
        # laplace smoothing!
        if phonemes is None:
            phonemes = get_phonemes(lexicon)
        possible_ngrams = pos_n(n, phonemes=phonemes)            # list of possible ngrams
        # and then, merge 'possible_ngram' (possible ngrams) and 'ngram_list' (observed ngrams)
        # in effect, adds one to all observed and unobserved datapoints
        ngram_list = possible_ngrams + ngram_list

    return Counter(ngram_list), word_ngram_dict


def ngram_word_likelihood(word, lexicon, mlexicon):
    # words: str. word (sequence of segments) to evaluate.
    # lexicon: Counter. a counter of ngrams
    # mlexicon: Counter. a counter of mgrams (where m = n - 1)
    lexicon_ngram = lexicon[0]
    lexicon_mgram = mlexicon[0]
    n = len(lexicon_ngram.most_common(1)[0][0])  # n is the 'n' in ngram
    m = len(lexicon_mgram.most_common(1)[0][0])  # m = n - 1

    if m != n - 1:
        print("ngram and mgram do not match")
        return

    ngram_tuples = ngrams(n, word, sep=' ')

    likelihood = 0
    for ng_tp in ngram_tuples:
        mg_tp = ng_tp[:m]
        if lexicon_ngram[ng_tp] == 0:
            return -99
        try:
            likelihood += log(lexicon_ngram[ng_tp] / lexicon_mgram[mg_tp])
        except ZeroDivisionError:
            likelihood += log(lexicon_ngram[ng_tp] / len(lexicon[1].keys()))
    return likelihood


def dump_ngram(ngram_obj, filename):
    with open(os.path.join('resources', filename), 'wb') as file:
        pickle.dump(ngram_obj, file)

def unpickler(filename):
    with open(os.path.join('resources', filename), 'rb') as file:
        res_object = pickle.load(file)
    return res_object


def strata_ngram_likelihood(word='',
                            trigram_counter_sino='',
                            bigram_counter_sino='',
                            trigram_counter_nonsino='',
                            bigram_counter_nonsino=''):
    # calculate ngram likelihood for a list of nonce words given sino vs nonsino stratum
    # word: list or str. list of nonce words
    if type(word) == list:
        # check if the resource directory exists if not, run pickler
        if not os.path.isdir('resources'):
            read_corpus_get_ngrams()

        # check if the pkl files exist if not, run pickler
        if not os.path.isfile(os.path.join('resources', 'sino_bigram.pkl')):
            read_corpus_get_ngrams()

        # if all required files exist, get trigram and bigram objects
        trigram_counter_sino = unpickler('sino_trigram.pkl')
        bigram_counter_sino = unpickler('sino_bigram.pkl')
        trigram_counter_nonsino = unpickler('nonsino_trigram.pkl')
        bigram_counter_nonsino = unpickler('nonsino_bigram.pkl')

        res_list = []
        for w in tqdm(word):
            res_list.append(strata_ngram_likelihood(w,
                                                    trigram_counter_sino,
                                                    bigram_counter_sino,
                                                    trigram_counter_nonsino,
                                                    bigram_counter_nonsino))
        return res_list

    nonce = word

    log_likelihood_sino = ngram_word_likelihood(nonce, trigram_counter_sino, bigram_counter_sino)  # in the base e log
    log_likelihood_nonsino = ngram_word_likelihood(nonce, trigram_counter_nonsino,
                                               bigram_counter_nonsino)  # in the base e log

    likelihood_sino = exp(log_likelihood_sino) / (exp(log_likelihood_sino) + exp(log_likelihood_nonsino))
    likelihood_nonsino = exp(log_likelihood_nonsino) / (exp(log_likelihood_sino) + exp(log_likelihood_nonsino))

    return likelihood_sino, likelihood_nonsino


def main():
    # calculate likelhood of a nonce word to be Sino or non-Sino:
    # nonce = 't ɑ s t ɑ ŋ'
    filenames = ["resV C C V l T V C.txt"]
    for filename in filenames:
        with open(f'nounce_word_generator\\{filename}', 'r', encoding='utf-8') as f:
            wordlist = [line.strip() for line in f]

        res = strata_ngram_likelihood(wordlist)

        with open(f'guess_strata {filename}', 'w+') as f:
            for nonce in res:
                nonce = str(nonce)
                f.write(nonce + '\n')





if __name__ == "__main__":
    # main()

    trigram_counter_sino = unpickler('sino_trigram.pkl')
    bigram_counter_sino = unpickler('sino_bigram.pkl')
    trigram_counter_nonsino = unpickler('nonsino_trigram.pkl')
    bigram_counter_nonsino = unpickler('nonsino_bigram.pkl')
    # while True:
    with open('ipalist.txt','r',encoding='utf-8') as iinput:
        while True:
            raw_input = input("??? ")
            iinput = [raw_input, f'h {raw_input}', f'l {raw_input}']
            for usrinput in iinput:
                usrinput = usrinput.strip()
                p1,p2 = strata_ngram_likelihood(usrinput,
                                             trigram_counter_sino=trigram_counter_sino,
                                             bigram_counter_sino=bigram_counter_sino,
                                             trigram_counter_nonsino=trigram_counter_nonsino,
                                             bigram_counter_nonsino=bigram_counter_nonsino)
#
                print(f'{usrinput}\t{p1}\t{p2}')
                print(p1)
                print(p2)
#
#
