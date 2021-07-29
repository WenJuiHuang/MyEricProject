"""
File: boggle.py
Name: Eric
----------------------------------------
TODO: The following program will present the boggle game.
        Thoughts:
            1. For every letter on the board, search other eight directions (excluding several conditions). (for loop)
            2. search eight directions -> move on to the letter : recursive backtracking
                note: the searching process cannot search the previous letter or used letter
                        (use set() to memorize the location that we have used)
            3. to enhance the efficiency of recognizing a word: use dict to deal with dictionary.txt and has_prefix()
            4. output: List[str] that found by the program
"""

import time

# This is the file name of the dictionary txt file
# we will be checking if a word exists by searching through it
FILE = 'dictionary.txt'

dictionary = {}


def main():
    # input
    board = make_a_board()
    # if the input is not correct, the make_a_board() function will return none
    if board is None:
        return

    start = time.time()
    # deal with the dictionary.txt
    read_dictionary()

    found = []

    for r in range(len(board)):
        for c in range(len(board[0])):
            search(board, r, c, board[r][c], found, {(r,c)})

    print("There are " + str(len(found)) + " words in total.")
    end = time.time()
    print('----------------------------------')
    print(f'The speed of your boggle algorithm: {end - start} seconds.')


def make_a_board():
    """
    input: str: four rows of letter, each letter should only be one letter and be separated by a blank
    :return: List[list] : board
    """

    board = []
    for i in range(4):
        s = input(str(i + 1) + ' row of letters: ')
        if len(s) == 7:
            # case insensitive
            s = s.lower()
            new_lst = s.split()
        else:
            print('The input is not in correct format!')
            return
        board.append(new_lst)
    return board


def search(board, r, c, letter, found, path):

    # base case
    if len(letter) >= 4 and letter not in found and letter in dictionary[letter[0]]:
        print('Found: ' + letter)
        found.append(letter)
        # search one more time after the base case ex: room, roomy
        search(board, r, c, letter, found, path)

    else:
        # eight directions for every letter
        directions = [[-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0]]
        for dir in directions:
            new_r = r + dir[0]
            new_c = c + dir[1]

            if 0 <= new_r <= 3 and 0 <= new_c <= 3:
                if (new_r, new_c) not in path:
                    # choose
                    letter = letter + board[new_r][new_c]
                    path.add((new_r, new_c))
                    # explore
                    if has_prefix(letter):
                        search(board, new_r, new_c, letter, found, path)
                    # un-choose
                    letter = letter[:-1]
                    path.remove((new_r, new_c))

    return found


def read_dictionary():
    """
    This function reads file "dictionary.txt" stored in FILE
    and appends words in each line into a Python dict
    """

    with open(FILE, 'r') as f:
        for line in f:
            if line[0] in dictionary:
                dictionary[line[0]].append(line[:-1])
            else:
                dictionary[line[0]] = [line[:-1]]


def has_prefix(sub_s):
    """
    :param sub_s: (str) A substring that is constructed by neighboring letters on a 4x4 square grid
    :return: (bool) If there is any words with prefix stored in sub_s
    """

    for word in dictionary[sub_s[0]]:
        if word.startswith(sub_s):
            return True


if __name__ == '__main__':
    main()
