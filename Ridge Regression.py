
# coding: utf-8

# # Ridge Regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\train.csv")
test=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\test.csv")


# In[2]:


def Jtheta(x,y,theta,lemda):
    y_pred=x.dot(theta)
    c=sum([(np.round(val,2)**2) for val in (y_pred-y)])
    t_theta=sum([v**2 for v in theta])
    cost=(c+(t_theta*lemda))/(2*len(x))
    return cost


# In[3]:


def gradient_descent(x,y,theta,alpha,lemda):
    y_pred=x.dot(theta)
    c=y_pred-y
    grad = (x.T.dot(c))/len(x)
    z=theta*(1-alpha)
    z[0]=0
    z=(z+alpha)*lemda
    temp = theta-(alpha*(grad+z))
    return temp


# In[17]:


def predict(theta,test):
    x0_test = np.ones((len(test),1))
    #test.insert(loc = 0,column='x0',value=x0_test)
    pred =test.dot(theta)
    return pred  


# In[5]:


theta=np.array([1,1,1,1,1])
iteration=100
alpha=0.000001
lemda=0.000001
elist=[]
m=train.shape[1]
x = train.iloc[:,0:4]
y = train.iloc[:,4]
x0 = np.ones((len(x),1))
x.insert(loc = 0,column='x0',value=x0)
for i in range(iteration):
    error=Jtheta(x,y,theta,lemda)
    if error<0.00001:
        break
    else:
        elist.append(error)
        theta=gradient_descent(x,y,theta,alpha,lemda)
        
#plt.plot(list(range(100)),elist)
#plt.xlabel('no. of iteration ')
#plt.ylabel('cost')
#plt.show()


# In[6]:


theta


# In[21]:


pred=predict(theta,test)
pred


# In[22]:


yd= test.iloc[:,4]


# In[ ]:


from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(yd, y_pred))

