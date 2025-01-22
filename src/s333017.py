import numpy as np

#problem 1:
#mean square error: 7.125940794232773e-34 # ok
def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

#problem 2:
#mean square error: 14839666398288.217 # da vedere bene
def f2(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.add(np.add(np.add(np.multiply(9,9),x[0]),np.divide(np.sqrt(np.sin(x[1])),x[0])),np.divide(9.9,np.sqrt(x[0]))),np.multiply(np.add(np.add(np.multiply(3.1,8),np.divide(9.9,np.sqrt(x[0]))),np.divide(9.9,np.sqrt(x[0]))),np.multiply(np.subtract(np.add(np.divide(x[0],x[0]),np.multiply(x[0],-5)),np.add(np.multiply(x[0],9.1),np.multiply(x[0],4.5))),np.add(np.multiply(x[1],x[2]),np.multiply(4.3,-5)))))

#problem 3:
#mean square error: 10.117052287398375 # da riprovare
def f3(x: np.ndarray) -> np.ndarray:
    return np.add(np.add(np.multiply(x[0],x[0]),np.add(np.subtract(x[0],x[2]),np.subtract(np.multiply(x[0],x[0]),np.multiply(np.multiply(x[1],x[1]),x[1])))),np.subtract(np.subtract(4.2,x[2]),x[2]))

#problem 4:
#mean square error: 0.06882577028844528 # ok ma vediamo se può andare meglio
def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.add(np.add(np.add(np.sin(np.cos(-6)),np.cos(np.add(x[1],0))),np.cos(1)),np.add(np.add(np.cos(np.add(x[1],0)),np.cos(np.add(x[1],0))),np.cos(1))),np.add(np.add(np.add(np.sin(np.cos(-6)),np.cos(np.add(x[1],0))),np.cos(1)),np.add(np.add(np.cos(np.add(x[1],0)),np.cos(np.add(x[1],0))),np.cos(np.add(x[1],0)))))

#problem 5:
#mean square error: 1.3022611986665231e-18 # ok, magari migliorabile
def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.multiply(np.sqrt(np.divide(np.sqrt(np.divide(0,x[0])),np.cos(np.cos(9)))),np.cos(np.divide(np.sqrt(x[1]),np.cos(-3.2)))),np.add(np.divide(np.sqrt(x[1]),np.cos(np.divide(np.sqrt(x[1]),np.cos(-3.2)))),np.multiply(np.subtract(x[0],np.sqrt(np.cos(x[0]))),np.add(np.divide(np.multiply(x[1],x[1]),np.cos(7)),np.cos(x[1])))))

#problem 6:
#mean square error: 2.9834998484575704e-05 # ok
def f6(x: np.ndarray) -> np.ndarray:
    return np.add(np.subtract(np.divide(np.add(np.sin(-1.9),np.subtract(np.subtract(np.subtract(x[1],np.subtract(x[0],x[1])),np.divide(x[1],np.sqrt(9.7))),np.divide(x[0],np.add(-6,np.sqrt(9))))),np.sin(np.abs(-1.7))),np.sin(-1.9)),np.multiply(np.multiply(np.multiply(np.multiply(1.3,1.3),np.multiply(np.log(np.sin(np.abs(8))),x[0])),np.multiply(1.3,np.sin(np.abs(8)))),np.sin(np.multiply(9.3,np.sin(np.abs(-9.9))))))

#problem 7:
#mean square error: 60.94333832775311 # da riprovare
def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(np.multiply(np.multiply(np.multiply(5.9,np.multiply(np.add(x[1],x[0]),x[0])),np.log(np.subtract(x[1],x[0]))),x[1]))

#problem 8:
#mean square error: 29932.624601487092 # buono, da riprovare
def f8(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.multiply(np.multiply(np.multiply(np.abs(x[5]),x[5]),np.multiply(5.0,x[5])),np.multiply(np.abs(x[5]),x[5])),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9)),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9))