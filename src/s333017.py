import numpy as np

#problem 1:
#mean square error: 7.125940794232773e-34 # ok
def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

#problem 2:
#mean square error: 
def f2(x: np.ndarray) -> np.ndarray:
    return ...

#problem 3:
#mean square error: 10.117052287398375 # da riprovare
def f3(x: np.ndarray) -> np.ndarray:
    return np.add(np.add(np.multiply(x[0],x[0]),np.add(np.subtract(x[0],x[2]),np.subtract(np.multiply(x[0],x[0]),np.multiply(np.multiply(x[1],x[1]),x[1])))),np.subtract(np.subtract(4.2,x[2]),x[2]))

#problem 4:
#mean square error: 0.06882577028844528 # ok ma vediamo se può andare meglio
def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.add(np.add(np.add(np.sin(np.cos(-6)),np.cos(np.add(x[1],0))),np.cos(1)),np.add(np.add(np.cos(np.add(x[1],0)),np.cos(np.add(x[1],0))),np.cos(1))),np.add(np.add(np.add(np.sin(np.cos(-6)),np.cos(np.add(x[1],0))),np.cos(1)),np.add(np.add(np.cos(np.add(x[1],0)),np.cos(np.add(x[1],0))),np.cos(np.add(x[1],0)))))

#problem 5:
#mean square error:
def f5(x: np.ndarray) -> np.ndarray:
    return ...

#problem 6:
#mean square error: 2.9834998484575704e-05 # ok
def f6(x: np.ndarray) -> np.ndarray:
    return np.add(np.subtract(np.divide(np.add(np.sin(-1.9),np.subtract(np.subtract(np.subtract(x[1],np.subtract(x[0],x[1])),np.divide(x[1],np.sqrt(9.7))),np.divide(x[0],np.add(-6,np.sqrt(9))))),np.sin(np.abs(-1.7))),np.sin(-1.9)),np.multiply(np.multiply(np.multiply(np.multiply(1.3,1.3),np.multiply(np.log(np.sin(np.abs(8))),x[0])),np.multiply(1.3,np.sin(np.abs(8)))),np.sin(np.multiply(9.3,np.sin(np.abs(-9.9))))))

#problem 7:
#mean square error: 
def f7(x: np.ndarray) -> np.ndarray:
    return ...

#problem 8:
#mean square error: 29932.624601487092 # buono, da riprovare
def f8(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.multiply(np.multiply(np.multiply(np.abs(x[5]),x[5]),np.multiply(5.0,x[5])),np.multiply(np.abs(x[5]),x[5])),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9)),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9))
