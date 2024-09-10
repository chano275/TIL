from pprint import pprint 
import requests
import csv


STR_ID = []
with open('movies.csv', 'r', encoding = 'utf-8') as file:
    content = csv.DictReader(file)
    for row in content:
        STR_ID.append((row['id']))

movie_reviews = []
fields = ['id', 'author', 'content', 'rating']

for i in range(len(STR_ID)):
    url = "https://api.themoviedb.org/3/movie/" + STR_ID[i] + "/reviews?language=en-US&page=1"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkZDQ3MDMyZTIyNTE4YWQ1MTRkOGVlYmI5NGZkODAxYyIsIm5iZiI6MTcyMTM1NDk0Ny42MzU1NzUsInN1YiI6IjY2OTljOTIwZTU0NTNmOWE1NzBjYWZmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tw8AJBiRkWm_s6GNUTi15aZVBbWqRH58rIStX93ZGso"
    }
    response = requests.get(url, headers=headers).json()

    for item in response['results']: # 해당 영화에 대한 리뷰 쫘르륵 
        ox_check = 0 # 선택하지 않는 케이스 고를 flag 변수 
        temp_item = {}
        temp_item['movie_id'] = int(STR_ID[i])  
        for key in fields:  # key 돌면서 내가 원하는 특성 가져올 것 / 받아오는 id가 리뷰어 id 라서, 이후에 리스트에 넣기 전에 추가해줘야 한다. 
                            # rating 은 author_details 의 rating 으로 들어가 있음 / dtype : float  nonetype / rating 5 이상 nonetype 아닌 요소를 선택 > 최우선적으로 체크해야 
            if key == 'rating' :    
                if ((item['author_details']['rating']) is not None) and (item['author_details']['rating'] >= 5) :  
                    temp_item[key] = item['author_details']['rating']
                else: 
                    ox_check = 1
                    break 
  
            else: 
                if key == 'content' and item['content'] == '': # 리뷰내용 없으면 리뷰없음으로 
                    temp_item[key] = '내용 없음'
                else: # 정상 
                    if key == 'id':
                        temp_item['review_id'] = item[key]
                    else:
                        temp_item[key] = item[key]
        
        if ox_check == 1: pass
        else: movie_reviews.append(temp_item)
    
                        
with open('movie_reviews.csv', 'w', newline='', encoding='utf-8') as file:
    fieldsname = ['review_id', 'movie_id', 'author', 'content', 'rating']
    content = csv.DictWriter(file, fieldnames=fieldsname)
    content.writeheader()
    for item in movie_reviews:
        content.writerow(item)
