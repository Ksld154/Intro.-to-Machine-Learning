import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class tree(object):
    def __init__(self):
        self.parent = None
        self.left_bound = None
        self.right_bound = None
        self.left = None
        self.right = None
        self.data = None
        self.cut = None

def kd_tree (points, parent, cut):

    num_points = points['x'].count()
    if num_points == 0:
        return None

    #cut by x or cut by y
    if cut:
        points = points.sort_values(by = ['x']).reset_index(drop = True)
    else:
        points = points.sort_values(by = ['y']).reset_index(drop = True)
    new_tree = tree()
    new_tree.parent = parent
    new_tree.data = points.loc[int(num_points / 2)]
    new_tree.cut = cut
    #set bound
    if new_tree.parent is None:
        new_tree.left_bound = min-0.5
        new_tree.right_bound = max+0.5
    elif new_tree.parent is not None and new_tree.parent.parent is None:
        if cut:
            if new_tree.data[1] > new_tree.parent.data[1]:
                new_tree.left_bound = new_tree.parent.data[1]
                new_tree.right_bound = max+0.5
            else:
                new_tree.left_bound = min-0.5
                new_tree.right_bound = new_tree.parent.data[1]
        else:
            if new_tree.data[0] > new_tree.parent.data[0]:
                new_tree.left_bound = new_tree.parent.data[0]
                new_tree.right_bound = max+0.5
            else:
                new_tree.left_bound = min-0.5
                new_tree.right_bound = new_tree.parent.data[0]
    else:
        if cut:
            if new_tree.data[1] > new_tree.parent.data[1]:
                new_tree.left_bound = new_tree.parent.data[1]
                new_tree.right_bound = new_tree.parent.parent.right_bound
            else:
                new_tree.left_bound = new_tree.parent.parent.left_bound
                new_tree.right_bound = new_tree.parent.data[1]
        else:
            if new_tree.data[0] > new_tree.parent.data[0]:
                new_tree.left_bound = new_tree.parent.data[0]
                new_tree.right_bound = new_tree.parent.parent.right_bound
            else:
                new_tree.left_bound = new_tree.parent.parent.left_bound
                new_tree.right_bound = new_tree.parent.data[0]
    #draw point
    plt.plot(new_tree.data[0],new_tree.data[1],'ks',label='point')
    #draw line            
    if cut:
        plt.plot([new_tree.data[0],new_tree.data[0]],[new_tree.left_bound,new_tree.right_bound], color = 'b')
    else:
        plt.plot([new_tree.left_bound,new_tree.right_bound],[new_tree.data[1],new_tree.data[1]], color = 'r')

    #if num_points < 2 means there are no left and right
    if num_points < 2:
        return new_tree

    #seperate points into left and right
    points_left = points.loc[0 : int(num_points / 2) - 1]
    points_right = points.drop(points_left.index[:]).reset_index(drop = True)
    points_left = points_left.reset_index(drop = True)
    points_right = points_right.drop([0]).reset_index(drop = True)

    new_tree.left = kd_tree(points_left, new_tree, not cut)
    new_tree.right = kd_tree(points_right, new_tree, not cut)

    return new_tree

#main
points = []
file = open ("points.txt", "r")
for line in file:
    data = line.split()
    for i in range(len(data)):
        data[i] = int(data[i])
    points.append(data)

df = pd.DataFrame (points, columns = ['x', 'y'])
max = df['x'].max() if (df['x'].max() > df['y'].max()) else df['y'].max()
min = df['x'].min() if (df['x'].min() < df['y'].max()) else df['y'].min()
plt.xlim(min-0.5,max+0.5)
plt.ylim(min-0.5,max+0.5)

root = None

#calculate cut
cut = np.var(df['x'])>np.var(df['y'])

root = kd_tree (df, root, cut)
plt.show()