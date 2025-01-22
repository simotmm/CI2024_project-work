import numpy as np

#problem 1:
#mean square error: 7.125940794232773e-34 # ok
def f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0])


#problem 2: 13987717013674.275 # da migliorare
#mean square error: 
def f2(x: np.ndarray) -> np.ndarray:
    return np.add(np.subtract(np.multiply(np.multiply(np.multiply(np.multiply(np.subtract(9,-6),np.multiply(np.subtract(9,-6),x[0])),np.multiply(9,np.subtract(9,-6))),-6),-6),np.multiply(np.multiply(np.multiply(np.multiply(np.subtract(9,-6),np.multiply(np.subtract(9,-6),x[0])),np.multiply(9,np.subtract(9,-6))),-6),np.multiply(np.add(x[1],np.sin(x[2])),np.sin(np.subtract(np.subtract(9.9,x[2]),np.subtract(x[0],x[2])))))),np.multiply(np.multiply(np.multiply(np.multiply(np.sqrt(np.add(np.abs(8),1e-40)),x[2]),x[1]),np.subtract(8,np.multiply(np.multiply(np.tan(8),7),np.multiply(np.multiply(-7,x[0]),np.add(8,7))))),8))


#problem 3:
#mean square error: 5.306486117932578e-29 # ok
def f3(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.subtract(np.subtract(4,np.multiply(np.multiply(x[1],x[1]),x[1])),np.subtract(x[2],np.multiply(x[0],x[0]))),np.subtract(x[2],np.multiply(x[0],x[0]))),np.subtract(x[2],np.multiply(x[2],-0.5)))


#problem 4:
#mean square error: 0.022938281157499518 # ok
def f4(x: np.ndarray) -> np.ndarray:
    return np.add(np.divide(np.add(np.cos(x[1]),x[0]),np.add(-7.4,1e-40)),np.add(np.divide(np.multiply(np.add(np.cos(x[1]),np.log(np.add(np.abs(np.subtract(-9,-7.4)),1e-40))),np.sqrt(np.add(np.abs(-7.4),1e-40))),np.add(-7.4,1e-40)),np.multiply(np.sqrt(np.add(np.abs(-7.4),1e-40)),np.multiply(np.add(np.cos(x[1]),np.log(np.add(np.abs(np.subtract(-9,-7.4)),1e-40))),np.sqrt(np.add(np.abs(-7.4),1e-40))))))


#problem 5:
#mean square error: 7.804262963600968e-19 # ok
def f5(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.subtract(np.add(np.add(np.add(x[1],x[1]),np.abs(x[0])),np.cos(np.subtract(x[0],-10.0))),np.abs(np.multiply(np.log(np.add(np.abs(np.abs(x[0])),1e-40)),np.multiply(np.multiply(x[0],x[1]),np.add(-2.0,x[1]))))),np.multiply(np.sqrt(np.add(np.abs(np.sqrt(np.add(np.abs(np.tan(np.subtract(x[0],x[0]))),1e-40))),1e-40)),np.sqrt(np.add(np.abs(np.log(np.add(np.abs(np.abs(x[0])),1e-40))),1e-40))))


#problem 6:
#mean square error: 1.4783925196927175e-05 # ok
def f6(x: np.ndarray) -> np.ndarray:
    return np.add(np.multiply(np.divide(np.tan(7.1),np.add(np.add(np.divide(np.add(2.3,np.multiply(np.abs(8),-2.1)),np.add(np.multiply(np.sin(8),x[1]),1e-40)),np.multiply(np.multiply(np.sin(8),0),np.multiply(np.sin(8),0))),1e-40)),np.abs(np.sin(np.tan(np.tan(7.1))))),np.multiply(np.divide(np.add(np.divide(np.divide(np.add(x[0],x[1]),np.add(np.add(0,-8),1e-40)),np.add(np.sin(np.tan(np.tan(7.1))),1e-40)),np.add(np.add(np.add(x[1],x[1]),np.multiply(-0.5,x[0])),np.divide(np.add(x[0],x[1]),np.add(np.add(-5,-8),1e-40)))),np.add(np.sin(8),1e-40)),np.sin(np.tan(np.tan(7.1)))))


#problem 7: 
#mean square error: 32.653988150939874 # ok
def f7(x: np.ndarray) -> np.ndarray:
    return np.multiply(np.sqrt(np.add(np.abs(np.log(np.add(np.abs(np.log(np.add(np.abs(np.subtract(x[0],x[1])),1e-40))),1e-40))),1e-40)),np.abs(np.multiply(np.multiply(x[1],np.log(np.add(np.abs(np.subtract(x[0],x[1])),1e-40))),np.multiply(-6,np.multiply(x[0],np.add(x[0],x[1]))))))


#problem 8:
#mean square error: 29932.624601487092 #
def f8(x: np.ndarray) -> np.ndarray:
    return np.subtract(np.subtract(np.multiply(np.multiply(np.multiply(np.abs(x[5]),x[5]),np.multiply(5.0,x[5])),np.multiply(np.abs(x[5]),x[5])),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9)),np.multiply(np.multiply(np.multiply(np.abs(x[4]),x[4]),x[4]),9))
