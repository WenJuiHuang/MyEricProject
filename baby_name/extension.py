"""
File: extension.py
--------------------------
This file collects more data from
https://www.ssa.gov/oact/babynames/decades/names2010s.html
https://www.ssa.gov/oact/babynames/decades/names2000s.html
https://www.ssa.gov/oact/babynames/decades/names1990s.html
Please print the number of top200 male and female on Console
You should see:
---------------------------
2010s
Male Number: 10890537
Female Number: 7939153
---------------------------
2000s
Male Number: 12975692
Female Number: 9207577
---------------------------
1990s
Male Number: 14145431
Female Number: 10644002
"""

import requests
from bs4 import BeautifulSoup


def main():
    for year in ['2010s', '2000s', '1990s']:
        print('---------------------------')
        print(year)
        url = 'https://www.ssa.gov/oact/babynames/decades/names' + year + '.html'
        source_code = requests.get(url)
        html = source_code.text
        soup = BeautifulSoup(html, features="html.parser")
        # find out that the information we need is between <td>
        # 'table', {'class':'t-stripe'}.tbody.find_all('tr')
        items = soup.find_all('td')

        lst = []
        for item in items:
            item_string = item.string
            if item_string is not None and not item_string.isdigit() and not item_string.isalpha():
                item_int = int(item_string.replace(',', ''))
                lst.append(item_int)

        # index odd and index even
        print('Male Number: ' + str(sum(lst[::2])))
        print('Female Number: ' + str(sum(lst[1::2])))


if __name__ == '__main__':
    main()
