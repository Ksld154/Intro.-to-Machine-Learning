import pandas as pd
import numpy
import collections
import csv

def get_dict (attribute, csv_data):
    temp_dict = dict()
    for data_list in csv_data[attribute]:
        for string in numpy.array (data_list[2:-2].split('\', \'')):
            if string in temp_dict:
                temp_dict[string] += 1
            else:
                temp_dict[string] = 1
    temp_dict.pop ('')
    return temp_dict

def get_dict_ (attribute, csv_data):
    temp_dict = dict()
    for string in csv_data[attribute]:
        if string in temp_dict:
            temp_dict[string] += 1
        else:
            temp_dict[string] = 1
    return temp_dict

def get_id (dict_):
    id_ = dict()
    i = 1
    for string in dict_:
        id_[string] = i
        i += 1
    return id_


csv_data = pd.read_csv ('dataset_.csv')

genre_dict = get_dict ('genre', csv_data)
type_dict = get_dict_ ('type', csv_data)
studio_dict = get_dict ('studio', csv_data)
source_dict = get_dict_ ('source', csv_data)

type_id = get_id (type_dict)
studio_id = get_id (studio_dict)
source_id = get_id (source_dict)

data = pd.DataFrame (csv_data)

#add col for all genre
for key in genre_dict.keys():
    data[key] = 0

#add col for studio
data.insert(7, 'studio_id', 0)
data.insert(8, 'studio_num', 0)

#add col for time
data.insert(1, 'season', 0)
data.insert(2, 'year', 0)


for i in range(len(data.index)):
    #time pre
    if data['premiered'][i] != 'None':
        temp = data['premiered'][i].split()
        data.at[i, 'year'] = int(temp[1])
        season = temp[0]
        if season == 'Spring':
            data.at[i, 'season'] = 1
        elif season == 'Summer':
            data.at[i, 'season'] = 2
        elif season == 'Fall':
            data.at[i, 'season'] = 3
        elif season == 'Winter':
            data.at[i, 'season'] = 4

    #genre pre
    if len (data['genre'][i]) > 2:
        for string in numpy.array (data['genre'][i][2:-2].split('\', \'')):
            data.at[i, string] = 1

    #type pre
    data.at[i, 'type'] = type_id[data.at[i, 'type']]

    #studio pre
    studio = data['studio'][i][2:-2].split('\', \'')
    if studio[0] is not '':
        data.at[i, 'studio_id'] = studio_id[studio[0]]
        data.at[i, 'studio_num'] = studio_dict[studio[0]]

    #source pre
    data.at[i, 'source'] = source_id[data.at[i, 'source']]

data.drop (columns = 'name', inplace=True)
data.drop (columns = 'premiered', inplace=True)
data.drop (columns = 'genre', inplace=True)
data.drop (columns = 'producer', inplace=True)
data.drop (columns = 'licensor', inplace=True)
data.drop (columns = 'studio', inplace=True)

data.to_csv ('dataset_preprocessing.csv', index = False)
