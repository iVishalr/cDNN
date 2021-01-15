import numpy as np

a = np.ones((3,3))
a = a.flatten(order="F").reshape(1,-1).squeeze()

b = np.zeros((3,3))
b = b.flatten(order="F").reshape(1,-1).squeeze()

c = np.concatenate([a,b])

final = c.astype(np.float).tolist()
print(final)

f = open("test_file.data","a+")
for item in final:
  f.write("%s "%item)
f.close()