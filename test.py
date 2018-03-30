
class Base:

    def a(self):
        self.b()

    def b(self):
        print 'Base:b'

class Derive(Base):

    def b(self):
        print 'Derive:b'

if __name__ == '__main__':
    d = Derive()
    d.a()

    b = Base()
    b.a()


