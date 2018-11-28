import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pylab import *

dataset = pd.read_csv("data_noah.csv")  #read .csv file
df_all = pd.DataFrame(dataset) 
df = df_all[ ['x', 'y'] ]
df_type = df_all[['pitch_type']]


#print(df['x'])
#plt.scatter(df['x'], df['y'], s = 7)
#plt.show()



# K means
K = 3
max_itr = 5
itr = 0

n_data = np.size(df,axis=0)
n_attr = np.size(df,axis=1)


Group = np.zeros([n_data, K], dtype=float)
Distance = np.zeros([n_data, K], dtype=float)

GroupCenters = df.sample(n=K).reset_index(drop=True)

#n_GroupCenters = np.size(GroupCenters, axis=0)   #equals to K
#print(Centers.values[1])

def Accuracy():   #figure out which index is FF CH CC and print accuracy
    GroupCenters.sort_values(by='y', ascending = False, inplace = True )
    d1 = df_type.replace(['FF', 'CH', 'CU'], GroupCenters.index.tolist())
    correct = 0
    for row in range(n_data):
        row_data = d1.values[row]
        if CenterIndex[row] == row_data:
            correct = correct + 1
    accuracy = correct / n_data
    print('Accuracy:')
    print(accuracy)

while 1:
    itr = itr + 1
    for row in range(n_data):
        row_data = df.values[row]

        #Distance = np.zeros(K, dtype=float)
        
        # Calculate the Euclidean distance with k center points
        for c in range(K):
            CenterPoint = GroupCenters.values[c]
            Distance[row, c] = np.linalg.norm(row_data - CenterPoint)  # calculate the euclidean distance (also can use scipy.euclidean)



        #print(np.amin(Distance))

        # the center which is closest to the row_data means that
        # the row_data belongs to the group where the center at. 

    CenterIndex = np.argmin(Distance, axis=1)  # row_data's group number
    print(CenterIndex)
    #Group[row, CenterIndex] = row_data

    #print(df.head(5))
    #print(df.values[[1, 2, 4], :])
    #plt.subplot(itr)
    flag = 0
    for i in range(K):
        print(i)
        new_avg = np.mean(df.values[CenterIndex == i, :], axis=0)
        if (abs(new_avg-GroupCenters.values[i])).all() == 0:
            flag = flag + 1
        print(new_avg)
        print(GroupCenters.values[i])
        GroupCenters.values[i, :] = new_avg
    if flag == K:
        break    #break when all the means do not move anymore
    
   
        
    print('\n')
        #print(df.values[CenterIndex == i, :])
plt.title("Total times:"+str(itr))        
plt.plot(df.values[CenterIndex == 0][:, 0], df.values[CenterIndex == 0][:, 1], 'r,', df.values[CenterIndex == 1][:, 0], df.values[CenterIndex == 1][:, 1], 'g,', df.values[CenterIndex == 2][:, 0], df.values[CenterIndex == 2][:, 1], 'b,')
plt.plot(GroupCenters.values[0, 0], GroupCenters.values[0, 1], 'ro', GroupCenters.values[1, 0], GroupCenters.values[1, 1], 'go', GroupCenters.values[2, 0], GroupCenters.values[2, 1], 'bo')
Accuracy()
plt.show()



        

