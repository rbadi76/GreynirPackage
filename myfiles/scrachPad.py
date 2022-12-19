'''from collections import defaultdict, OrderedDict

class Person:

    def __init__(self):
        self.name = "Róbert"
        self.age = 46
        self.properties: dict[int, str] = { 0: "This", 1: "is", 2: "a", 3: "dictionary"}


    def __str__(self):
        return "Hi, my name is " + self.name + " and I am " + str(self.age) + " old!"

myVar = Person()
print(myVar)
print(myVar.properties)'''

from islenska import Bin
b = Bin()
myvar = b.lookup("vera")
print(myvar)