import numpy as np
import matplotlib.pyplot as plt
import new_data


########################################## MAE
#
# def cal_cost(theta,X,y):
#     m=len(y)
#     predictions = X.dot(theta)
#     cost = (1/2*m)*np.sum(np.abs(predictions-y))
#     return cost
#
# def gradient_descent(X,y,theta,learning_rate,iterations):
#     m = len(y)
#     cost_history = np.zeros(iterations)
#     theta_history = np.zeros((iterations,2))
#     for it in range (iterations):
#         prediction = np.dot(X,theta)
#         # theta[0] = theta[0] - (1/2*m)*learning_rate * (X[:,1].dot(np.sign(prediction-y)))
#         # theta[1] = theta[1] - (1 /2* m) * learning_rate * (np.mean(np.sign(prediction - y)))
#         theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
#         theta_history[it,:]=theta.T
#         cost_history[it]=cal_cost(theta,X,y)
#     return theta, cost_history , theta_history

############################## MSE

def cal_cost(theta,X,y):
    m=len(y)
    predictions = X.dot(theta)
    cost = (1/2*m)*np.sum(np.square(predictions-y))
    return cost

def gradient_descent(X,y,theta,learning_rate,iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,2))
    for it in range (iterations):

        prediction = np.dot(X,theta)
        # theta[1] = theta[1] - (1 / m) * learning_rate * (X[:,1].dot((prediction - y)))
        # theta[0] = theta[0] -  learning_rate * (np.mean((prediction - y)))
        theta = theta - (1 / m) * learning_rate * (X.T.dot((prediction - y)))
        theta_history[it,:]=theta.T
        cost_history[it]=cal_cost(theta,X,y)
    return theta, cost_history , theta_history

############## UKE
# def cal_cost(theta,X,y):
#     m=len(y)
#     predictions = X.dot(theta)
#     cost = (1/2*m)*np.sum(predictions-y)
#     return cost
#
# def gradient_descent(X,y,theta,learning_rate,iterations):
#     m = len(y)
#     cost_history = np.zeros(iterations)
#     theta_history = np.zeros((iterations,2))
#     for it in range (iterations):
#         theta[1] = theta[1] - (1 / 2 * m) * learning_rate * np.sum(X[:, 1])
#         theta[0] = theta[0]
#         # theta = theta - (1/2*m)*learning_rate * np.sum(X.T,axis=1,keepdims=True)
#         theta_history[it,:]=theta.T
#         cost_history[it]=cal_cost(theta,X,y)
#     return theta, cost_history , theta_history


##################################################### enter input
x , y = new_data.halazooni()
m = x.mean()
v = x.std()
x = (x - m) / v
m = y.mean()
v = y.std()
y = (y - m) / v
X = np.c_[np.ones((1000, 1)), x]
#####################################################
# lr = 0.01
lr = 0.2
# lr = 0.5
# lr = 0.9
n_iter = 1000
theta,cost_history, theta_history = gradient_descent(X,y,np.array([-3,-7]).reshape(-1,1), lr,n_iter)
############################################
fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(2, 3, 3)
ax.plot(x,y,'.',color = 'purple')
plt.title('data random distibuted')
##################################################### plot best line
# the best theta
ax = fig.add_subplot(2, 3, 1)
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
ax.plot(x,y,'.',color = 'purple')
ax.plot(x,X.dot(theta_best),color = 'green',linewidth=2)
plt.title('the best line(bastekhat)')
###########################################################plot gradient descent lines
ax = fig.add_subplot(2, 3, 2)
ax.plot(x,y,'.',color = 'purple')
theta_first=np.array([0,-1]).reshape(-1,1)
ax.plot(x,X.dot(theta_first),color = 'orange')
for i in range(n_iter):
    if i%100 == 0:
        ax.plot(x,X.dot(theta_history[i]),color = 'blue')
ax.plot(x,X.dot(theta),color = 'green',linewidth=2)
plt.title('lines of gradient descent')
######################################################################contour

N = 9
theta1 = [np.array((0,0))]
J = [cost_history[0]]
for j in range(N-1):
    theta1.append(theta_history[j*100])
theta1.append(theta)

colors = ['b', 'g', 'm', 'c', 'orange','k','b', 'g', 'm', 'k']
ax = fig.add_subplot(2, 3, 4)
for j in range(1,N):
    ax.annotate('', xy=theta1[j], xytext=theta1[j-1],
                   arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                   va='center', ha='center')
ax.scatter(*zip(*theta1), c=colors, s=40, lw=0)

theta0_grid = np.linspace(-10,10,100)
theta1_grid = np.linspace(-10,10,100)

J_grid=[]
#
for i in range(100):
    for j in range(100):
        t = []
        t.append(theta0_grid[i])
        t.append(theta1_grid[j])
        t = np.array(t)
        a = t[:, np.newaxis]
        J_grid.append(cal_cost(a, X, y))


# A labeled contour plot for the RHS cost function
X1, Y1 = np.meshgrid(theta0_grid, theta1_grid)
J_grid = np.reshape(J_grid, X1.shape)

ax.contour(X1, Y1, J_grid, 30)
plt.title('2D contour of gradient descent')
theta_0 = theta_history[:,0]
theta_1 = theta_history[:,1]
ax = fig.add_subplot(2, 3, 5, projection='3d')
ax.plot_surface(X1, Y1, J_grid, rstride = 5, cstride = 5, cmap = 'jet', alpha=0.5)
ax.plot(theta_0,theta_1,cost_history, marker = '*', color = 'r', alpha = .4, label = 'Gradient descent')

ax.set_xlabel('theta 0')
ax.set_ylabel('theta 1')
ax.set_zlabel('Cost function')
ax.set_title('Gradient descent: Root at {}'.format(theta.ravel()))
ax.view_init(45, 45)
plt.title('3d contour of gradient descent')
###########################################################error
ax = fig.add_subplot(2, 3, 6)
def error(X,y):
    lr = 0.01
    n_iter = 1000
    theta, cost_history, theta_history = gradient_descent(X, y, np.array([-3, -7]).reshape(-1, 1), lr, n_iter)
    epoch = range(n_iter)
    ax.plot(epoch, cost_history, color='green')

    lr = 0.2
    theta, cost_history, theta_history = gradient_descent(X, y, np.array([-3, -7]).reshape(-1, 1), lr, n_iter)
    epoch = range(n_iter)
    ax.plot(epoch, cost_history, color='blue')

    lr = 0.5
    theta, cost_history, theta_history = gradient_descent(X, y, np.array([-3, -7]).reshape(-1, 1), lr, n_iter)
    epoch = range(n_iter)
    ax.plot(epoch, cost_history, color='purple')

    lr = 0.9
    theta, cost_history, theta_history = gradient_descent(X, y, np.array([-3, -7]).reshape(-1, 1), lr, n_iter)
    epoch = range(n_iter)
    ax.plot(epoch, cost_history, color='red')
    ax.set_xlabel(r'epoch')
    ax.set_ylabel(r'error')

error(X,y)
plt.suptitle('MSE cost function in 0.2 learning rate')
plt.show()
