class A:
    def __init__(self):
        pass
    def sayHi(self):
        print "Hello, I'm A"
    def greet(self):
        self.sayHi()

class B(A):
    def __init__(self):
        pass
    def sayHi(self):
        print "Hello, I'm B"

a = A()
a.sayHi()
a.greet()
b = B()
b.greet()
print "a"+[5,4]