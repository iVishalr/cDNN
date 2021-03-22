import numpy as np

file = open("./src/test/TEST_DOT.txt","r", encoding="utf-8")
s = file.read()
file.close();

matrix = []
values = []
shapes = []
s = s.splitlines()
print(len(s))
for test_cases in s:
  values.append(test_cases.split(" ")[0])
  shapes.append(test_cases.split(" ")[1])

for i in range(len(values)):
  temp = []
  for j in values[i]:
    temp.append(int(j))
  values[i] = temp

for i in range(len(shapes)):
  shapes[i] = shapes[i][1:-1]

for i in range(len(shapes)):
  temp = []
  for j in shapes[i].split(","):
    temp.append(int(j))
  shapes[i] = temp

list_arr = []
for i in range(len(shapes)):
  list_arr.append(np.array(values[i]).reshape((shapes[i][0],shapes[i][1])))

dot_list = []
for arrays in list_arr:
  dot_list.append(np.dot(arrays,arrays.T))

writeFile_list = []
for dot in dot_list:
  writeFile_list.append(((dot.reshape(1,-1)).tolist())[0])

write_file_str = []
for i in range(len(writeFile_list)):
  temp = []
  for j in writeFile_list[i]:
    temp.append(str(j))
  write_file_str.append("".join(temp))

file = open("./src/test/_val_DOT.txt","a+")
for i in range(len(write_file_str)):
  file.write(write_file_str[i]+"\n")
file.close()