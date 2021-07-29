"""
File: add2.py
Name: Eric
------------------------
TODO: The program will sum two linked lists and output one new linked list. The linked lists are stored in
        reverse order. EX: l1: 2->4->3 (342) l2: 5->6->4(465) output: 7->0->8(807)
"""

import sys


class ListNode:
    def __init__(self, data=0, pointer=None):
        self.val = data
        self.next = pointer


def add_2_numbers(l1: ListNode, l2: ListNode) -> ListNode:
    #######################
    # TODO:
    #       (1) 邊traversal 邊將l1,l2 的val 直接相加
    #       (2) 考量進位: 因為輸出也是反向輸出，所以進位的部分可以照listnode的方向直接進位。創造carry_num處理進位
    #       (3) 處理當l1, l2長度不一樣的情況，一樣同時traversal相加，空格當0
    #       (4) 處理當l1, l2加完後進位多出來的一個數字，直接加一個node到l3
    #
    #
    #######################

    # dummy 的概念
    ans_node = ListNode(0)
    cur = ans_node
    carry_num = 0

    while l1 is not None and l2 is not None:
        cur.next = ListNode((l1.val + l2.val + carry_num) % 10)
        carry_num = (l1.val + l2.val + carry_num) // 10
        l1 = l1.next
        l2 = l2.next
        cur = cur.next

    if l1 is not None and l2 is None:
        while l1 is not None:
            cur.next = ListNode((l1.val + carry_num) % 10)
            carry_num = (l1.val + carry_num) // 10
            l1 = l1.next
            cur = cur.next

    if l2 is not None and l1 is None:
        while l2 is not None:
            cur.next = ListNode((l2.val + carry_num) % 10)
            carry_num = (l2.val + carry_num) // 10
            l2 = l2.next
            cur = cur.next

    if carry_num == 1:
        cur.next = ListNode(1)

    return ans_node.next


####### DO NOT EDIT CODE BELOW THIS LINE ########


def traversal(head):
    """
    :param head: ListNode, the first node to a linked list
    -------------------------------------------
    This function prints out the linked list starting with head
    """
    cur = head
    while cur.next is not None:
        print(cur.val, end='->')
        cur = cur.next
    print(cur.val)


def main():
    args = sys.argv[1:]
    if not args:
        print('Error: Please type"python3 add2.py test1"')
    else:
        if args[0] == 'test1':
            l1 = ListNode(2, None)
            l1.next = ListNode(4, None)
            l1.next.next = ListNode(3, None)
            l2 = ListNode(5, None)
            l2.next = ListNode(6, None)
            l2.next.next = ListNode(4, None)
            ans = add_2_numbers(l1, l2)
            print('---------test1---------')
            print('l1: ', end='')
            traversal(l1)
            print('l2: ', end='')
            traversal(l2)
            print('ans: ', end='')
            traversal(ans)
            print('-----------------------')
        elif args[0] == 'test2':
            l1 = ListNode(9, None)
            l1.next = ListNode(9, None)
            l1.next.next = ListNode(9, None)
            l1.next.next.next = ListNode(9, None)
            l1.next.next.next.next = ListNode(9, None)
            l1.next.next.next.next.next = ListNode(9, None)
            l1.next.next.next.next.next.next = ListNode(9, None)
            l2 = ListNode(9, None)
            l2.next = ListNode(9, None)
            l2.next.next = ListNode(9, None)
            l2.next.next.next = ListNode(9, None)
            ans = add_2_numbers(l1, l2)
            print('---------test2---------')
            print('l1: ', end='')
            traversal(l1)
            print('l2: ', end='')
            traversal(l2)
            print('ans: ', end='')
            traversal(ans)
            print('-----------------------')
        elif args[0] == 'test3':
            l1 = ListNode(0, None)
            l2 = ListNode(0, None)
            ans = add_2_numbers(l1, l2)
            print('---------test3---------')
            print('l1: ', end='')
            traversal(l1)
            print('l2: ', end='')
            traversal(l2)
            print('ans: ', end='')
            traversal(ans)
            print('-----------------------')
        else:
            print('Error: Please type"python3 add2.py test1"')


if __name__ == '__main__':
    main()
