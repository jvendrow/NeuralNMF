import numpy as np
from time import time

x = np.random.rand(100)

z = np.zeros(100, dtype=np.bool)
convert = np.arange(100)

z[convert%2==1] = np.ones(50, dtype=np.bool)

P = convert[z]

a = time()
f1 = x[P]
b = time()
f2 = x[z]
c = time()
f3 = x[~z]
d = time()

print(b-a)
print(c-b)
print(d-c)
