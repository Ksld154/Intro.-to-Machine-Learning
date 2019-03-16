import requests
import json
import time
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

start_id = 0
end_id = 0
test = True

while test:
    start_id = input ("From: ")
    end_id = input ("To: ")
    # if start_id <= end_id:
    test = False
    # else:
    #     print ("input error\n")

start_crawler = time.time ()

index = 1

df = pd.DataFrame (columns = ['animeID', 'name', 'type', 'source', 'episodes', 
'rating', 'score', 'scoredBy', 'rank', 'popularity', 'members', 'favorites', 
'premiered', 'studio', 'genre'])

for i in range (int (start_id), int (end_id) + 1):
    start = time.time ()
    url = 'http://api.jikan.moe/anime/' + str (i)
    res = requests.get (url, timeout = None)
    
    data = json.loads (res.text)

    if 'error' in data:
        print ("%6d    %6f sec" %(i, time.time () - start), "    Didn't find.")
        continue
    
    new_list = []

    #mal_id
    if 'mal_id' in data:
        new_list.append (int (data['mal_id']))
    else:
        new_list.append (None)

    #title
    if 'title' in data:
        if data['title'] == "" or data['title'] == None:
            new_list.append ('None')
        else:
            new_list.append (data['title'])
    else:
        new_list.append ('None')

    #type
    if 'type' in data:
        if data['type'] == "" or data['type'] == None:
            new_list.append ('None')
        else:
            new_list.append (data['type'])
    else:
        new_list.append ('None')

    #source
    if 'source' in data:
        if data['source'] == "" or data['source'] == None:
            new_list.append ('None')
        else:
            new_list.append (data['source'])
    else:
        new_list.append ('None')

    #episodes
    if 'type' in data:
        if data['episodes'] == "" or data['episodes'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['episodes']))
    else:
        new_list.append (0)

    #rating
    if 'rating' in data:
        if data['rating'] == "" or data['rating'] == None:
            new_list.append ('None')
        else:
            new_list.append (data['rating'])
    else:
        new_list.append ('None')

    #score
    if 'score' in data:
        if data['score'] == "" or data['score'] == None:
            new_list.append (0)
        else:
            new_list.append (float (data['score']))
    else:
        new_list.append (0)

    #scored_by
    if 'scored_by' in data:
        if data['scored_by'] == "" or data['scored_by'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['scored_by']))
    else:
        new_list.append (0)

    #rank
    if 'rank' in data:
        if data['rank'] == "" or data['rank'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['rank']))
    else:
        new_list.append (0)

    #popularity
    if 'popularity' in data:
        if data['popularity'] == "" or data['popularity'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['popularity']))
    else:
        new_list.append (0)

    #members
    if 'members' in data:
        if data['members'] == "" or data['members'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['members']))
    else:
        new_list.append (0)

    #favorites
    if 'favorites' in data:
        if data['favorites'] == "" or data['favorites'] == None:
            new_list.append (0)
        else:
            new_list.append (int (data['favorites']))
    else:
        new_list.append (0)

    #premiered
    if 'premiered' in data:
        if data['premiered'] == "" or data['premiered'] == None:
            new_list.append ('None')
        else:
            new_list.append (data['premiered'])
    else:
        new_list.append ('None')

    #studio
    if 'studio' in data:
        if data['studio'] == "" or data['studio'] == []:
            new_list.append ('None')
        else:
            new_list.append (data['studio'][0]['name'])
    else:
        new_list.append ('None')
    
    #genre
    genre_list = []
    if 'genre' in data:
        if data['studio'] == "" or data['studio'] == []:
            new_list.append ('[]')
        else:
            for genre in data['genre']:
                genre_list.append (genre['name'])
            new_list.append (genre_list)
    else:
        new_list.append ('[]')

    df.loc [index] = [n for n in new_list]
    index += 1

    print ("%6d    %6f sec" %(i, time.time () - start), "    Done!")

df.to_csv ('dataset_' + str (start_id) + '_' + str (end_id) + '.csv', index = False)

print ("\nspend: %4f sec" %(time.time () - start_crawler))
print ("result: %d animes" %(index - 1))