# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. .Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: E Kamalesh
RegisterNumber:  212222100019
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
    m=len(y) 
    h=X.dot(theta) 
    square_err=(h-y)**2
    return 1/(2*m)*np.sum(square_err) 

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) 

def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    J_history=[] #empty list
    for i in range(num_iters):
        predictions=X.dot(theta)
        error=np.dot(X.transpose(),(predictions-y))
        descent=alpha*(1/m)*error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def predict(x,theta):
    predictions=np.dot(theta.transpose(),x)
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For Population = 35000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For Population = 70000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

### Profit prediction:
![270279988-72c751de-9d20-419c-908d-f9d68e91d28e](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/f6b7b7cd-2a20-4269-9399-4f4fde26a45a)
### Function:
![270279726-5440dbe0-743e-4f21-9b15-c6750af2f4e1](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/5b1bb252-f2d9-42b8-bfb7-9f0416071511)
### GRADIENT DESCENT:
![270279708-aef30837-1496-469f-97f8-2a7a64ca63b7](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/33e5fb27-af2b-446d-b91c-c2c87b384e4a)
### COST FUNCTION USING GRADIENT DESCENT:
![270279961-17ee5b6b-631f-475a-9c3f-4e0a5e3a7b2e](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/930fc5cd-dfa2-425d-a888-7df9c3790947)
### LINEAR REGRESSION USING PROFIT PREDICTION:
![270279896-59d451d6-9bc7-4a74-aca4-e5c6e1790efb](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/ab6f1e43-c98a-486e-b754-55b7a1dc2209)
### PROFIT PREDICTION FOR A POPULATION OF 35000:
![270279656-024ca885-488e-400f-a698-300882d60a1a](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/a78e38ef-2d84-4789-8ca1-6b6ccd7eacad)
### PROFIT PREDICTION FOR A POPULATION OF 70000:
![270279641-b1150597-9c47-4e21-874e-7714e24aad18](https://github.com/kamalesh2509/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120444689/4d50aa24-ae63-4192-be42-841f97e55da5)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
