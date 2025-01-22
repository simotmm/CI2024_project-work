import numpy as np

#problem 1:
#mean square error: 7.125940794232773e-34 # ok
def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])

#problem 2:
#mean square error: 15744936786290.686 # da vedere bene
def f2(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.add(np.add(np.add(np.add(x[0],x[0]),np.add(x[0],np.abs(-2))),-9.9),np.add(np.add(np.tan(-4.7),-1.3),np.add(np.add(np.tan(-4.7),-1.3),np.add(np.cos(-9),np.multiply(x[0],x[1]))))),np.multiply(np.add(np.add(np.add(np.tan(-4.7),-4.7),-4.7),np.multiply(np.sqrt(x[2]),np.multiply(x[1],np.add(x[2],x[2])))),np.multiply(np.subtract(np.cos(-2),x[0]),np.add(np.add(np.tan(-4.7),np.sqrt(7)),np.sqrt(np.sqrt(x[2]))))))

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
#mean square error: 0.0004160407569370655 # ok
def f6(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.add(np.subtract(np.add(x[1],np.multiply(np.multiply(-0.1,7.2),np.subtract(x[0],x[1]))),-0.1),np.divide(np.cos(np.tan(5)),np.add(8.8,x[1]))),np.cos(np.abs(np.tan(np.tan(np.tan(5))))))

#problem 7:
#mean square error: 60.94333832775311 # da riprovare
def f7(x: np.ndarray) -> np.ndarray:
    return np.abs(np.multiply(np.multiply(np.multiply(5.9,np.multiply(np.add(x[1],x[0]),x[0])),np.log(np.subtract(x[1],x[0]))),x[1]))

#problem 8:
#mean square error: 29932.624601487092 # buono, da riprovare
def f8(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.multiply(np.multiply(np.multiply(np.abs(x[5]),x[5]),np.multiply(5.0,x[5])),np.multiply(np.abs(x[5]),x[5])),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9)),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9))