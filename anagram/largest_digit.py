"""
File: largest_digit.py
Name: Eric
----------------------------------
This file recursively prints the biggest digit in
5 different integers, 12345, 281, 6, -111, -9453
If your implementation is correct, you should see
5, 8, 6, 1, 9 on Console.
"""


def main():
    print(find_largest_digit(12345))  # 5
    print(find_largest_digit(281))  # 8
    print(find_largest_digit(6))  # 6
    print(find_largest_digit(-111))  # 1
    print(find_largest_digit(-9453))  # 9


def find_largest_digit(n):

    num = 0
    return helper(num, abs(n))


def helper(num, n):
    # if num is bigger than the last ch, num will be the ans
    if num > n:
        return num

    else:
        # find ways to slice the whole number one ch by one ch
        num = max(num, n % 10)
        return helper(num, n // 10)


if __name__ == '__main__':
    main()
