length = [150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, #normal
          
          150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, #obesity 1 step
          
          150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, #obesity 2 step
          
          150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0, #obesity 3 step
          
          150.0, 151.0, 152.0, 153.0, 154.0, 155.0, 156.0, 157.0, 158.0, 159.0, 
            160.0, 161.0, 162.0, 163.0, 164.0, 165.0, 166.0, 167.0, 168.0, 169.0,
            170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0  #obesity 4 step
          ]  #length(cm)

weight = [45.0, 45.9, 46.8, 47.7, 48.6, 49.5, 50.4, 51.3, 52.2, 53.1, 
            54.0, 54.9, 55.8, 56.7, 57.6, 58.5, 59.4, 60.3, 61.2, 62.1,
            63.0, 63.9, 64.8, 65.7, 66.6, 67.5, 68.4, 69.3, 70.2, 71.1, #normal
          
          52.0, 52.9, 53.8, 54.7, 55.6, 56.5, 57.4, 58.3, 59.2, 60.1, 
            61.0, 61.9, 62.8, 63.7, 64.6, 65.5, 66.4, 67.3, 68.2, 69.1,
            70.0, 70.9, 71.8, 72.7, 73.6, 74.5, 75.4, 76.3, 77.2, 78.1, #obesity 1 step
          
          59.0, 59.9, 60.8, 61.7, 62.6, 63.5, 64.4, 65.3, 66.2, 67.1, 
            68.0, 68.9, 69.8, 70.7, 71.6, 72.5, 73.4, 74.3, 75.2, 76.1,
            77.0, 77.9, 78.8, 79.7, 80.6, 81.5, 82.4, 83.3, 84.2, 85.1, #obesity 2 step
          
          66.0, 66.9, 67.8, 68.7, 69.6, 70.5, 71.4, 72.3, 73.2, 74.1, 
            75.0, 75.9, 76.8, 77.7, 78.6, 79.5, 80.4, 81.3, 82.2, 83.1,
            84.0, 84.9, 85.8, 86.7, 87.6, 88.5, 89.4, 90.3, 91.2, 92.1, #obesity 3 step
          
          73.0, 73.9, 74.8, 75.7, 76.6, 77.5, 78.4, 79.3, 80.2, 81.1, 
            82.0, 82.9, 83.8, 84.7, 85.6, 86.5, 87.4, 88.3, 89.2, 90.1,
            91.0, 91.9, 92.8, 93.7, 94.6, 95.5, 96.4, 97.3, 98.2, 99.1 #obesity 4 step
          ] #weight(kg), man standard
import numpy as np

obesity_rate_target = ([0] * 30) + ([1] * 30) + ([2] * 30) + ([3] * 30) + ([4] * 30) 
obesity_rate_data = np.column_stack((length, weight))

try:
  your_length = input("your length >").strip()
  your_weight = input("yout weight >").strip()
  yours = []
  yours.append(int(your_length))
  yours.append(int(your_weight))
except:
  print('please input int or float')

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    obesity_rate_data, obesity_rate_target, stratify = obesity_rate_target, random_state = 150
)

mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)

train_scaled = (train_input - mean) / std

from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target) #train start
kn.fit(train_scaled, train_target) 

test_scaled = (test_input - mean) /std

import matplotlib.pyplot as plt
new = (yours - mean) / std  #your length, weight
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1], color = 'b')
plt.scatter(new[0], new[1], marker = '^')
plt.scatter(train_scaled[indexes,0], train_scaled[indexes,1], marker = 'D', color = 'r')
plt.xlabel('length')
plt.ylabel('weight')
plt.plot([-1.5, -0.5, 0.5, 1.5, 2], [-2, -1.37, -0.74, -0.1, 0.2], color = 'g') #normal line 
plt.show()

#print your obesity step

a = [0 ,1, 2, 3 ,4]

if kn.predict([new]) == a[0]:
  print("normal")
elif kn.predict([new]) == a[1]:
  print("obesity 1 step")
elif kn.predict([new]) == a[2]:
  print("obesity 2 step")
elif kn.predict([new]) == a[3]:
  print("obesity 3 step")
elif kn.predict([new]) == a[4]:
  print("obesity 4 step")
