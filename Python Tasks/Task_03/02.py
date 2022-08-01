import cmath
import math

AB = int(input())
BC = int(input())

print (str(int(round(math.degrees(cmath.phase(complex(BC,AB))))))+'\u00B0')