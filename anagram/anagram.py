"""
File: anagram.py
Name:
----------------------------------
This program recursively finds all the anagram(s)
for the word input by user and terminates when the
input string matches the EXIT constant defined
at line 19

If you correctly implement this program, you should see the
number of anagrams for each word listed below:
    * arm -> 3 anagrams
    * contains -> 5 anagrams
    * stop -> 6 anagrams
    * tesla -> 10 anagrams
    * spear -> 12 anagrams
"""

import time  # This file allows you to calculate the speed of your algorithm

# Constants
FILE = 'dictionary.txt'  # This is the filename of an English dictionary
EXIT = '-1'  # Controls when to stop the loop

dictionary = {}


def main():
    """
    (1) about dealing with dictionary.txt: use dict (data type {alphabet: [all the words start with the same alphabet]})
    dict is faster than list!!!
    (2) about dealing with input word: at first, I tried to use the the same kind of back tracking as 'adv_permutaion',
    the ex in the class, however it can not work well when there are >1 same chs in a word (ex: contains, 2 'n').
    Therefore, I tried calculating the number of each ch in a word and put them into a dict (ex: {n : 2})
    """
    print('Welcome to stanCode \"Anagram Generator\" (or -1 to quit)')
    while True:
        input_word = input('Find anagrams for: ')
        start = time.time()
        if input_word == EXIT:
            break
        else:
            read_dictionary()
            ans = find_anagrams(input_word)
            print(f'{len(ans)} anagrams: {ans}')
            end = time.time()
            print('----------------------------------')
            print(f'The speed of your anagram algorithm: {end - start} seconds.')


def read_dictionary():
    with open(FILE, 'r') as f:
        for line in f:
            # key in dict
            if line[0] in dictionary:
                # 原本用strip，但發現每個list的第一個單字依然有換行符號，故改成下列方式
                dictionary[line[0]].append(line[:-1])
            else:
                dictionary[line[0]] = [line[:-1]]


    return dictionary


def find_anagrams(s):
    """
    :param s:
    :return:
    """

    ch = {}
    for sub in s:
        if sub in ch:
            ch[sub] += 1
        else:
            ch[sub] = 1

    return_lst = []
    find_anagrams_helper(s, '', ch, return_lst)
    return return_lst


def find_anagrams_helper(word_str, curr_str, input_word_dict, return_lst):
    """

    :param word_str: str, the word users input
    :param curr_str: str, produce every time by the recursion process
    :param ans_len: int, the num of the ch of the input word
    :param input_word_dict: dict, {character: number}
    :param return_lst: list, the ans after searching anagrams
    """
    if sum(input_word_dict.values()) == 0 and curr_str in dictionary[curr_str[0]] and curr_str not in return_lst:
        print('Searching...')
        print(curr_str)
        return_lst.append(curr_str)

    else:
        for ele in word_str:
            # if two words are anagrams, the num of each ch should be the same
            if input_word_dict[ele] != 0:
                # choose
                curr_str += ele
                input_word_dict[ele] -= 1

                if has_prefix(curr_str):
                    # explore
                    find_anagrams_helper(word_str, curr_str, input_word_dict, return_lst)

                # un-choose
                curr_str = curr_str[:-1]
                input_word_dict[ele] += 1


def has_prefix(sub_s):
    """
    :param sub_s: str, produce every time by the recursion process
    :return: bool, whether there are words starting with sub_s
    """
    # since we have index to loop over the dictionary, it saves a lot of time
    for word in dictionary[sub_s[0]]:
        if word.startswith(sub_s):
            return True


if __name__ == '__main__':
    main()
