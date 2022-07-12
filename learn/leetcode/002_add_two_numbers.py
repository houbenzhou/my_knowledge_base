# 给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字
# 请你将两个数字相加，并以相同形式返回一个表示和的链表
# 你可以假设除了数字0之外，这两个数都不会以0开头

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        carry = 0
        # dummy head
        head = curr = ListNode(0)
        while l1 or l2:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            curr.next = ListNode(val % 10)
            curr = curr.next
            carry = val / 10
        if carry > 0:
            curr.next = ListNode(carry)
        return head.next

if __name__ == '__main__':
    # begin
    # list_node1 = ListNode()
    # list_node2 = ListNode()
    s = Solution()
    print (s.addTwoNumbers([2,4,3], [5,6,4]))












