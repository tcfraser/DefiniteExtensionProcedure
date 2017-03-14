from collections import Counter

Counter = Counter

class Foocounter(Counter):
    capture_methods = [
        '__isub__', '__and__', '__iand__', '__or__', 'copy', '__ior__', '__sub__',
        '__add__', '__sub__', # TODO ensure I have all of the creation methods
    ]

    def __init__(self, *args, **kwargs):
        super(Counter, self).__init__(*args, **kwargs)

class Fooset(set):
    capture_methods = [
        '__ror__', 'difference_update', '__isub__',
        'symmetric_difference', '__rsub__', '__and__', '__rand__', 'intersection',
        'difference', '__iand__', 'union', '__ixor__',
        'symmetric_difference_update', '__or__', 'copy', '__rxor__',
        'intersection_update', '__xor__', '__ior__', '__sub__',
    ]

    def __init__(self, *args, **kwargs):
        super(Fooset,self).__init__(*args, **kwargs)

class Foofrozenset(frozenset):
    capture_methods = [
        '__ror__', 'symmetric_difference', '__rsub__', '__and__', '__rand__', 'intersection',
        'difference', 'union', '__or__', 'copy', '__rxor__', '__xor__', '__sub__',
    ]

    def __new__(cls, *args, **kwargs):
        return super(Foofrozenset,cls).__new__(cls,*args, **kwargs)

def capture_methods(cls, target_super, names):
    def capture_method_closure(name):
        super_function = getattr(target_super, name)
        def inner(self, *args):
            return cls(super_function(self, *args))
        inner.fn_name = name
        setattr(cls, name, inner)
    for name in names:
        capture_method_closure(name)

capture_methods(Fooset,         set,        Fooset.capture_methods)
capture_methods(Foofrozenset,   frozenset,  Foofrozenset.capture_methods)
capture_methods(Foocounter,     Counter,    Foocounter.capture_methods)

if __name__ == '__main__':
    A = Fooset(('A', 'B', 'C'))
    B = Fooset(('B', 'C'))
    a = Foofrozenset(('a', 'b', 'c'))
    b = Foofrozenset(('b', 'c'))
    print('A', A)
    print('B', B)
    print('a', a)
    print('b', b)
    print('A - B', A - B)
    print('a - b', a - b)

    T = Foocounter({'A':2, 'B':3})
    G = Foocounter({'B':2, 'C':3})
    print('G', G)
    print('T', T)
    T['A'] += 1
    print('T[\'A\'] += 1')
    print('T', T)
    print('T - G', T - G)
    print('T & G', T & G)

    try:
        ST = Foocounter({A:1, B:1})
    except TypeError as e:
        print('except', e)

    ST = Foocounter({a:1, b:1})
    print("ST", ST)
