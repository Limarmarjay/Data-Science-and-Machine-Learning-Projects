# This project delves into the inner workings of 1D deep neural networks (Vectors), examining key mathematical concepts crucial to their operation. 
# This includes an exploration of functions used in Logistic Regression and Classification.
# Tools used: Python, Numpy, Matplotlib


import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-10, 10, 100)
m = 5
b = 4

# Convert x to 2D array
x_2D = x.reshape((10, 10))
print(x_2D)

# Function definitions

def linearFunc(x, m, b):
    y = (m * x) + b
    return y
def logisticsFunc(x):
    y = 1/1 + np.exp(-x)
    return y


# Make plots with matplotlib
y = logisticsFunc(x)

# create figure and axis
fig, ax = plt.subplots()
ax.plot(x, y, "r")

plt.savefig('DNN_project.png')

# for loop to create numerous graphs at once
for i in range(1):
    print(i+1)
    F1 = linearFunc(x, m, b)
    G1 = logisticsFunc(F1)
    F2 = linearFunc(G1, A, B)
    G2 = logisticsFunc(F2)


    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax.plot(x, F1, color = 'c')
    ax.grid()  # grid is used to display the gridlines of the graph
    ax2.plot(x, G1, 'c')
    ax2.grid()
    ax3.plot(x, F2, 'c')
    ax3.grid()
    ax4.plot(x, G2, 'c', label = 'Graph of Output')
    #plt.xlim(-2.5, 2.5)
    #ax4.plot(x, x, color = 'c', label = 'Graph of y = x')
    ax4.grid()
    ax4.legend()
    plt.show()

# The graphs keeps changing everytime due to the composition of the functions
# In all, there are noticeable changes in the y values on the vertical axis while the x values on the horizontal axis remains the same


# for loop to create all the layers' graphs at once
for i in range(1):


  # Input layer
  layer1 = linearFunc(x, a, b)
  layer1_activation = logisticsFunc(layer1) # G(F(x))


  # Hidden layer 1
  layer2 = linearFunc(layer1_activation, W, b)
  layer2_activation = logisticsFunc(layer2) # G(F(G(F(x))))


  # Hidden layer 2
  layer3 = linearFunc(layer2_activation, W, b)
  layer3_activation = logisticsFunc(layer3) # G(F(G(F(G(F(x))))))

  # Output layer
  output = linearFunc(layer3_activation, W, b)
  y = logisticsFunc(output)                # G(F(G(F(G(F(G(F(x))))))))



  fig, ax = plt.subplots()
  ax.plot(x, layer1_activation, label='Layer 1 Activation', color = 'r')
  ax.plot(x, layer2_activation, label='Layer 2 Activation', color = 'g')
  ax.plot(x, layer3_activation, label='Layer 3 Activation', color = 'c')
  ax.plot(x, y, label='Output', color = 'm')
  ax.set_xlabel('x values')
  ax.set_ylabel('Activation Function')
  ax.set_title('Neural Network Layers')
  ax.grid()
  ax.legend()

  plt.tight_layout()
  plt.show()
  
  # DECLARATION OF OTHER FUNCTIONS SIGNIFICANT IN DEEP NEURAL NETWORK PROCESSES
# Rectified Linear Unit activation function. Piecewise function whose output ranges between 0 and positive x values.
def ReLUfunc(x):
    r = np.maximum(0, x)
    return r

# Hyperbolic tangent activation function. Another S-Shaped squashing function with output ranges between 1 and -1
def tanhfunc(x):
    y = (np.exp(x) - np.exp(-x))/(np.exp(-x) + np.exp(x))
    return y

# Softmax activation Function. This outputs a probability distribution over a set of mutually exclusive categories. It can be used for Multi-class classification problems

def softmaxfunc(z):
    Z = np.exp(z - np.max(z))  # subtract max value for numerical stability
    sum = np.sum(Z)
    softmax_z = np.round(Z / sum,3)
    return softmax_z
