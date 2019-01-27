from heapq import heappush, heappop, heapify

class EmptyQueueException(Exception):
    def __str__(self):
        return "Cannot pop an empty queue"

# Implementation of a priority queue.
# elements will be sorted in descreasing order.
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.dup = set()

    def __str__(self):
        return '[' + ' '.join([str(i) for i in self.queue]) + ']'

    # to check if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # push more than an element
    def pushAll(self, l):
        for a in l:
            self.push(a)

    # for inserting an element in the queue
    def push(self, e):
        if e.status != "int":
            heappush(self.queue, e)

        elif e not in self.dup:
            heappush(self.queue, e)
            self.dup.add(e)

    def __len__(self):
        return len(self.queue)

    # for popping an element based on Priority
    def pop(self):
        if self.isEmpty():
            raise EmptyQueueException()
        else:
            return heappop(self.queue)

    def __iter__(self):
        """
        Do ascending iteration for TreeSet
        """
        for element in self.queue:
            yield element

    def clear(self):
        self.queue = []
