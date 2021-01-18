import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys

COST_PATH = "./bin/cost.data"
TRAIN_ACC_PATH = "./bin/train_acc.data"

file = open(COST_PATH,"r")
cost_data = file.read().split(" ")[:-1]
file.close()

file = open(TRAIN_ACC_PATH,"r")
train_acc_data = file.read().split(" ")[:-1]
file.close()

cost = np.array(cost_data).astype(np.float)
train_acc = np.array(train_acc_data).astype(np.float)

cost = cost.reshape((1,-1)).squeeze()
train_acc = train_acc.reshape((1,-1)).squeeze()

index = np.arange(100,int(sys.argv[1])+100,100)

cost_series = pd.Series(data=cost,index=index,name="cost")
train_acc_series = pd.Series(data=train_acc,index=index,name="train_acc")

df = pd.concat([cost_series,train_acc_series],axis=1).reset_index()
df = df.set_index(['index'])
plt.figure(figsize=(15,9))
plt.plot(df["cost"],color="red",alpha=0.5,label="Cost")
plt.xlabel("Iterations",color="black",labelpad=30,fontsize=20)
plt.ylabel("Cost",color="black",labelpad=30,fontsize=20)
ticks = np.arange(100,int(sys.argv[1])+200,300)
plt.xticks(ticks=ticks,fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.title("Training Cost over "+sys.argv[1]+" iterations",pad=30,fontsize=15)
plt.savefig('cost.png', bbox_inches='tight')
# plt.show()
plt.figure(figsize=(15,9))
plt.plot(df["train_acc"],color="green",label="train_acc")
plt.xlabel("Iterations",color="black",labelpad=30,fontsize=20)
plt.ylabel("Accuracy",color="black",labelpad=30,fontsize=20)
ticks = np.arange(100,int(sys.argv[1])+200,300)
plt.xticks(ticks=ticks,fontsize=15)
plt.yticks(fontsize=15)
plt.legend()
plt.title("Training Accuracy over "+sys.argv[1]+" iterations",pad=30,fontsize=15)
plt.savefig('train_acc.png', bbox_inches='tight')
# plt.show()

os.remove(COST_PATH)
os.remove(TRAIN_ACC_PATH)