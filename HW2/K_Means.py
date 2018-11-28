import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dataset = pd.read_csv("data_noah.csv")  #read .csv file
df_all = pd.DataFrame(dataset) 
df = df_all[ ['x', 'y'] ]
df_type = df_all[['pitch_type']]

def Accuracy():   #figure out which index is FF CH CC and print accuracy
    df1 = GroupCenters.sort_values(by='y', ascending = False)
    d1 = df_type.replace(['FF', 'CH', 'CU'], df1.index.tolist())
    correct = 0
    for row in range(n_data):
        row_data = d1.values[row]
        if CenterIndex[row] == row_data:
            correct = correct + 1
    accuracy = correct / n_data
    print("Accuracy: %f\n" % accuracy)

# K means
K = 3
max_itr = 5
itr = 0

n_data = np.size(df,axis=0)
n_attr = np.size(df,axis=1)


Group = np.zeros([n_data, K], dtype=float)
Distance = np.zeros([n_data, K], dtype=float)

GroupCenters = df.sample(n=K).reset_index(drop=True)


while 1:
    itr = itr + 1
    for row in range(n_data):
        row_data = df.values[row]
        
        # Calculate the Euclidean distance with k center points
        for c in range(K):
            CenterPoint = GroupCenters.values[c]
            Distance[row, c] = np.linalg.norm(row_data - CenterPoint)  # calculate the euclidean distance (L2-Norm)

    # the center which is closest to the row_data means that: the row_data belongs to the group where the center at. 
    CenterIndex = np.argmin(Distance, axis=1)  # row_data's group number


    flag = 0
    for i in range(K):
        new_avg = np.mean(df.values[CenterIndex == i, :], axis=0)
        if (abs(new_avg - GroupCenters.values[i])).all() == 0:
            flag = flag + 1
        GroupCenters.values[i, :] = new_avg
    if flag == K:
        break    #break when all the means do not move anymore

if K >= 3:
    Accuracy()

plt.title("Total times:"+str(itr))
plt.xlabel("Horizontal Movement (Inches)")
plt.ylabel("Vertical Movement (Inches)")

x = np.arange(K)
ys = [i+x+(i*x)**2 for i in range(K)]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

#for y, c in zip(ys, colors):
#    plt.scatter(x, y, color=c)

for i,c in zip(range(K), colors):
    plt.scatter(df.values[CenterIndex == i][:, 0], df.values[CenterIndex == i][:, 1], s=1, color=c)
    plt.plot(GroupCenters.values[i, 0], GroupCenters.values[i, 1], 'o', color=c)
plt.show()




#================== figure out which index is FF CH CC  ===========================
