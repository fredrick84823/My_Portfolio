import pandas as pd
import numpy as np

def bias_coef_update(m, b, X, Y, learning_rate):
    m_gradient = 0
    b_gradient = 0
    
    N = len(Y)
    m_grad_list = []
    b_grad_list = []

    # iterate over examples
    for idx in range(len(Y)):
        x = X[idx]
        y = Y[idx]
        
        # predict y with current bias and coefficient
        y_pred = (m * x) + b
        m_gradient += -(2/N) * x * (y - y_pred)
        b_gradient += -(2/N) * (y - y_pred)
        # print("m_g:", m_gradient)
        # print("b_g:", b_gradient)
        
    # use gradient with learning_rate to nudge bias and coefficient
    new_coef = m - (m_gradient * learning_rate)
    new_bias = b - (b_gradient * learning_rate)
    
    return new_coef, new_bias
  
def cost(x,y,m,b):
  y_pred = [(m*i) + b for i in x]
  return sum([abs(a-b) for a,b in zip(y_pred, y)]) / len(x)
  
def run(epoch_count=10000000):
  epochs = []
  costs = []
        
  m = 1.15
  b = 20
  learning_rate = 1e-9
  for i in range(epoch_count):
      m, b = bias_coef_update(m, b, x, y, learning_rate)
      C = cost(x,y,m,b)
      if i%100000==0:
        print(f"a={m}, b={b}, cost={C}")
        print()
      epochs.append(i)
      costs.append(C)
    
  return epochs, costs, m, b

# generate data
t = [[10342], 
     [41693],
     [27934], 
     [15294],]

df = pd.DataFrame(data=t, columns=['feature'])
df['label'] = df['feature']*2.5 + 500 # make label

x = df.feature.tolist()
y = df.label.tolist()

# run
epochs, costs, m, b = run()
