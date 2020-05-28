######### 1

import numpy as np
import matplotlib.pyplot as plt


def ring():
    n = 1000
    theta_out = np.random.uniform(low=0, high=2 * np.pi, size=n)
    noise_out = np.random.uniform(low=0.9, high=1.1, size=n)
    x_out = np.cos(theta_out) * noise_out
    y_out = np.sin(theta_out) * noise_out
    x_out = x_out.reshape(len(x_out),1)
    y_out = y_out.reshape(len(y_out),1)
    x_out = x_out/max(x_out)
    y_out = y_out/max(y_out)
    return x_out , y_out


######## 2

def moones():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=1000, noise=0.1)
    a = X[:, 0]
    b = X[:, 1]
    a = a.reshape(len(a), 1)
    b = b.reshape(len(b), 1)
    a = a/max(a)
    b = b/max(b)
    return a,b



######## 3


def four_circular_data():
    x=[]
    y=[]
    r = 1* np.sqrt(np.absolute(np.random.randn(250, 1)))
    teta =  np.pi * np.random.randn(250, 1)
    x.append((r * np.cos(teta) - 6))
    y.append(r * np.sin(teta)-4)
    x.append((r * np.cos(teta) + 6))
    y.append(r * np.sin(teta) + 4)
    x.append((r * np.cos(teta) + 6))
    y.append(r * np.sin(teta) - 4)
    x.append((r * np.cos(teta) - 6))
    y.append(r * np.sin(teta) + 4)
    x =np.array(x)
    y=np.array(y)
    # x = x.reshape(len(x[0]),1)
    y = y.transpose(2,0,1).reshape(1000,-1)
    x=x.transpose(2,0,1).reshape(1000,-1)
    x = x/max(x)
    y = y/max(y)
    return x , y

########## 4

def halazooni():
    import numpy as np
    import matplotlib.pyplot as plt

    def twospirals(n_points, noise=.5):
        """
         Returns the two spirals dataset.
        """
        n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
        d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
        return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))),
                np.hstack((np.zeros(n_points), np.ones(n_points))))

    X, y = twospirals(1000)

    a = X[y == 0, 0]
    b = X[y == 0, 1]
    a = a.reshape(len(a), 1)
    b = b.reshape(len(b), 1)
    a = a/max(a)
    b = b/max(b)
    return a,b

########## 5

def two_circular_data():
    x=[]
    y=[]
    r = 1* np.sqrt(np.absolute(np.random.randn(500, 1)))
    teta =  (np.pi) * np.random.randn(500, 1)
    x.append((10*r * np.cos(teta) - 6))
    y.append(r * np.sin(teta)-4)
    x.append((r * np.cos(teta) - 6))
    y.append(10*r * np.sin(teta) - 4)
    x =np.array(x)
    y=np.array(y)
    # x = x.reshape(len(x[0]),1)
    y = y.transpose(2,0,1).reshape(1000,-1)
    x=x.transpose(2,0,1).reshape(1000,-1)
    x = x/max(x)
    y = y/max(y)
    return x , y

############ 6
# def circular_data():
#     r = 3* np.sqrt(np.absolute(np.random.randn(1000, 1)))
#     teta =  np.pi * np.random.randn(1000, 1)
#     x = 10*r * np.cos(teta)
#     y = r * np.sin(teta)
#     return x , y
def line():
    X = 2 * np.random.rand(1000, 1)
    # X = np.linspace(-8,8,1000)
    y = 4 + 3 * X + np.random.randn(1000, 1)
    X = X/max(X)
    y = y/max(y)
    return X,y


########## 7

def sahmi_circular_data():
    x=[]
    y=[]
    r = 1* np.sqrt(np.absolute(np.random.randn(500, 1)))
    teta =  np.pi * np.random.randn(500, 1)
    x.append(r * np.cos(teta))
    y.append(10*r * np.sin(teta)+80)
    x =np.array(x)
    y=np.array(y)
    # x = x.reshape(len(x[0]),1)
    y = y.transpose(2,0,1).reshape(500,-1)
    x=x.transpose(2,0,1).reshape(500,-1)
    xx = 5 * np.random.randn(500, 1)
    yy = xx ** 2 + (8 * np.random.randn(500, 1))
    # x = np.c_[x,xx]
    # y = np.c_[y,yy]
    x=np.concatenate((x, xx))
    y=np.concatenate((y, yy))
    x = x/max(x)
    y = y/max(y)
    return x , y


############## 8

def darajese():
    X = 2 * np.random.randn(1000, 1)
    # X = 5* X
    # y = X ** 2 + (8 * np.random.randn(1000, 1))
    # X = 2 * np.random.randn(1000, 1)
    # X = 5 * X
    y = X ** 3 + (8 * np.random.randn(1000, 1))
    X = X / max(X)
    y = y / max(y)
    return X,y

def gausian():
    # defining the standard deviation
    mu = 0.5
    sigma = 0.1
    np.random.seed(0)
    X = np.random.normal(mu, sigma, (1000, 1))
    Y = np.random.normal(mu * 2, sigma * 3, (1000, 1))
    X = X/max(X)
    Y = Y/max(Y)
    return X , Y

def sin():
    r = 5 * np.sqrt(np.absolute(np.random.randn(1000, 1)))
    teta = 2 * np.random.randn(1000, 1)
    x = r * np.cos(teta)
    x = x/max(x)
    teta = teta/max(teta)
    return x,teta
#
# x , y = sahmi_circular_data()
# # # print(x)
# plt.plot(x,y,'.')
# plt.show()

# plt.figure(figsize=(20,40))
# plt.subplot(251)
# x , y = ring()
# plt.plot(x,y,'.')
# plt.subplot(252)
# x , y = moones()
# plt.plot(x,y,'.')
# plt.subplot(253)
# x , y = four_circular_data()
# plt.plot(x,y,'.')
# plt.subplot(254)
# x , y = halazooni()
# plt.plot(x,y,'.')
# plt.subplot(255)
# x , y = two_circular_data()
# plt.plot(x,y,'.')
# plt.subplot(256)
# x , y = line()
# plt.plot(x,y,'.')
# plt.subplot(257)
# x , y = sahmi_circular_data()
# plt.plot(x,y,'.')
# plt.subplot(258)
# x , y = sahmi()
# plt.plot(x,y,'.')
# plt.subplot(259)
# x , y = gausian()
# plt.plot(x,y,'.')
# plt.subplot(2,5,10)
# x , y = sin()
# plt.plot(x,y,'.')
#
# plt.show()
