import pandas as pd
import matplotlib.pyplot as plt 
data=pd.read_csv('studytime_score_dataset_large.csv')
def loss_function(m,b,points):
    total_error=0
    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        total_error+=(y-(m*x+b ))**2
    return total_error/float(len(points)) 
def gradient_descent(m,b,points,learning_rate):
    m_gradient=0
    b_gradient=0
    N=float(len(points))
    for i in range(len(points)):
        x=points.iloc[i].studytime
        y=points.iloc[i].score
        m_gradient+=-(2/N)*x*(y-(m*x+b))
        b_gradient+=-(2/N)*(y-(m*x+b))
    new_m=m-(learning_rate*m_gradient)
    new_b=b-(learning_rate*b_gradient)
    return new_m,new_b
m=0
b=0
learning_rate=0.001
epochs=5000
for i in range(epochs):
    if i%100==0:
        print("Epoch:",i,"Loss:",loss_function(m,b,data))
    m,b=gradient_descent(m,b,data,learning_rate)
print("m:",m)
print("b:",b)
plt.scatter(data.studytime,data.score,color='blue')
plt.plot(list(range(0,10)),[m*x+b for x in range(0,10)],color='red')
plt.show()