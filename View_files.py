
# coding: utf-8

# In[4]:


import csv
import numpy as np


# In[5]:


data=[]
if __name__ == "__main__":
    csv_path = "demen.csv"
    with open(csv_path, "rt") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for line in reader:
            data.append(line)


# In[6]:


datax=[]
datay=[]
for line in data:
    datax.append(line[0:len(line)-1])
    datay.append(int(line[len(line)-1]))


x=np.array(datax)
x=x.astype(np.float)
y=np.array(datay)
y=y.astype(np.int)
print(x[0])


# In[ ]:




