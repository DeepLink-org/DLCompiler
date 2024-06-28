from z3 import Int, Solver, sat, unsat

x = Int('x')
y = Int('y')
z = Int('x')

s = Solver()
s.add(x > 2)
s.add(y < 10)
s.add(x + 2 * y == 7)

# equality test doesn't work
print(x == y)
print(x == z)

print(x is y)  # False
print(x is z)  # False

a = {x: 1}
print(x in a)  # True
print(z in a)  # True

print()

print(s.check())
print(s.check() == sat)
print(s.check() == unsat)

print()
print(s.model())
model = s.model()
for d in model:
    print(type(d), type(model[d]))
    print("%s -> %s" % (d, model[d]))
    print(model[d].is_int(), model[d].as_long(), type(model[d].as_long()))
