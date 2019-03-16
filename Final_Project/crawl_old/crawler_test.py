import requests
from bs4 import BeautifulSoup




def MalCrawler():
    url = 'https://myanimelist.net/anime/10721'
    resp = requests.get(url) #回傳為一個request.Response的物件
    # print(resp.status_code)  #return 200 => Normal

    soup = BeautifulSoup(resp.text, 'html.parser')    
    left_column_container = soup.find(name='div',attrs={"class":'js-scrollfix-bottom'})

    # s = str(left_column_container)
    # left_col_string = s.encode("utf8").decode("cp950", "ignore")
    # f = open('left_col.html', 'w')
    # f.write(left_col_string)

    div_list = left_column_container.find_all(name='div')
    len_div = len(div_list)

    # print(div_list[12])
    print(div_list[12].span.text)
    print(div_list[12].a.text)
    i = 0
    for div in div_list:



        i = i + 1




def AttributesCrawler(left_column_container):
    pass

if __name__ == "__main__":
    MalCrawler()    