# import json
from jikanpy import Jikan





if __name__ == "__main__":
    jikan = Jikan()

    anime = jikan(1)
    item = {
        'id': anime['mal_id'],
        'title': anime['title'],
        'type': anime['tyoe'], 
        'source': anime['source'], 
        'episodes': anime['episodes'], 
        'status': anime['status'], 
        'airing': anime['airing'], 
        # 'aired': {'from': '1998-04-03T00:00:00+00:00', 'to': '1999-04-24T00:00:00+00:00', 

        'duration': anime['duration'], 
        'rating': anime['rating'], 
        'score': anime['score'], 
        'scored_by': anime['score_by'], 
        'rank': anime['rank'], 
        'popularity_rank': anime['popularity'], 
        'members': anime['members'], 
        'favorites': anime['favorites'],
    }

    print(item)