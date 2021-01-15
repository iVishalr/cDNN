import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = open("./bin/cost.data","rb")
cost = f.read().decode("utf-8")
f.close()
cost = cost.split(" ")[:-1]
cost_arr = np.array(cost).astype(np.float)

cost_df = pd.Series(cost_arr)

g = sns.lineplot(x=cost_df.index,y=cost_df.values)
g.figure.savefig("cost.png")
# plt.show()