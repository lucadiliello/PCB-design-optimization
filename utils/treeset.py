#-------------------------------------------------------------------------------
# treeset.py
#
#
# Copyright (C) 2016, Ryosuke Fukatani
# License: Apache 2.0
# Modified by Luca Di Liello on 12/01/2019
#-------------------------------------------------------------------------------

import bisect

class TreeSet(object):
    """
    Binary-tree set like java Treeset.
    Duplicate elements will not be added.
    When added new element, TreeSet will be sorted automatically.
    """
    def __init__(self):
        self._treeset = []
        self.pop_number = 0
        self.push_number = 0

    def isEmpty(self):
        return len(self) == 0

    def addAll(self, elements):
        for element in elements:
            if element in self: continue
            self.add(element)

    def add(self, element):
        #print "Adding ", element
        if self.isEmpty() or element not in self:
            #print "Pos: ", bisect.bisect_left(self._treeset, element)
            bisect.insort_right(self._treeset, element)
            self.push_number += 1
        #print "After Adding: ", [str(s) for s in self._treeset]

    def add_high_low(self, element):
        #print "Adding ", element
        if element not in self:
            #print "Pos: ", bisect.bisect_left(self._treeset, element)
            index = bisect.bisect_right(self._treeset, element)
            self._treeset.insert(index, element)

            low = self._treeset[index - 1] if index > 0 else None
            high = self._treeset[index + 1] if index < len(self._treeset) - 1 else None
            self.push_number += 1
            return low, high

        #print "After Adding: ", [str(s) for s in self._treeset]
        return None

    def push(self, element):
        self.add(element)

    def pushAll(self, l):
        for a in l:
            self.push(a)

    def pop(self):
        if not self.isEmpty():
            res = self._treeset[0]
            self._treeset = self._treeset[1:]
            return res
        return None

    # Returns the least element in this set strictly greater than the given element,
    # or None if there is no such element.
    def higher(self, e):
        index = bisect.bisect_right(self._treeset, e)
        if index < self.__len__():
            return self[index]
        else:
            return None

    # Returns the greatest element in this set strictly less than the given element,
    # or None if there is no such element.
    def lower(self, e):
        index = bisect.bisect_left(self._treeset, e)
        if index > 0:
            return self[index - 1]
        else:
            return None

    def swap(self, e1, e2):
        i1 = self._treeset.index(e1)
        i2 = self._treeset.index(e2)
        self._treeset[i1] = e2
        self._treeset[i2] = e1

    def __getitem__(self, num):
        return self._treeset[num]

    def __len__(self):
        return len(self._treeset)

    def clear(self):
        """
        Delete all elements in TreeSet.
        """
        self._treeset = []

    def remove(self, element):
        """
        Remove element if element in TreeSet.
        """
        try:
            self._treeset.remove(element)
        except ValueError:
            return False
        return True

    def __iter__(self):
        """
        Do ascending iteration for TreeSet
        """
        for element in self._treeset:
            yield element

    def __str__(self):
        return '[' + ' '.join([str(i) for i in self._treeset]) + ']'

    def __eq__(self, target):
        if isinstance(target, TreeSet):
            return self._treeset == target._treeset
        elif isinstance(target, list):
            return self._treeset == target

    def __contains__(self, e):
        """
        Fast attribution judgment by bisect
        """
        try:
            return e == (self._treeset[bisect.bisect_left(self._treeset, e)])
        except:
            return False
