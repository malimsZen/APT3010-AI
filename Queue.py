from Node import Node
from LinkedList import LinkedList

class Queue:
    def __init__(self):
        self.list = LinkedList()

    def enqueue(self, new_item):
        new_node = Node(new_item)
        self.list.append(new_node)

    def dequeue(self):
        dequeued_item = self.list.head.data
        self.list.remove_after(None)
        return dequeued_item
