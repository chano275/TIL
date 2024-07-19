from pprint import pprint 
import requests
import csv

STR_ID = []
with open('movies.csv', 'r', encoding = 'utf-8') as file:
    content = csv.DictReader(file)
    for row in content:        STR_ID.append((row['id']))

movie_cast = []
fields = ['id', 'name', 'character', 'order'] # id > cast_id로 / str_id 추가해야 

for i in range(len(STR_ID)):
    ox_check = 0 # 선택하지 않는 케이스 고를 flag 변수 
    url = "https://api.themoviedb.org/3/movie/"+ STR_ID[i] +"/credits?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkZDQ3MDMyZTIyNTE4YWQ1MTRkOGVlYmI5NGZkODAxYyIsIm5iZiI6MTcyMTM1NDk0Ny42MzU1NzUsInN1YiI6IjY2OTljOTIwZTU0NTNmOWE1NzBjYWZmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tw8AJBiRkWm_s6GNUTi15aZVBbWqRH58rIStX93ZGso"
    }
    response = requests.get(url, headers=headers).json()


#    pprint(response)
    who_ = []
    for item in response['cast']:
        temp_item = {}
        temp_item['movie_id'] = int(STR_ID[i])  

        for key in fields: 
            if key == 'id':
                temp_item['cast_id'] = item[key]
            else:
                # filter 1 
                if key == 'order' and item[key] > 10: # 출연 순서 10 이하만 
                    ox_check = 1
                    break

                # filter 2
                if key == 'name' and item[key] in who_: 
                    ox_check = 1
                    break         
                elif key == 'name': who_.append(item[key])
                # primary key 가 cast_id 인걸 생각해서 who_를 쓰는 방법을 다르게 조절해야 했을 것 같은데,
                # 시간 부족으로 인해 요구에 만족하지 못했다. 
                # 다음 코드 작성 시에는 각 테이블의 스키마를 보고 중요한 flow를 미리 생각하고 해야겠다. 

                # correction 1
                if (key == 'name' or key == 'character') and ('\n' in item[key]):
                    item[key] = item[key].replace('\n', ' ') 

                # correction 2
                if key == 'name' and item[key] == '': 
                    item[key] = '이름 없음'

                temp_item[key] = item[key] # 여기서 다 거른거 append 

        if ox_check == 1: pass
        else: movie_cast.append(temp_item)

#pprint(movie_cast)

with open('movie_cast.csv', 'w', newline='', encoding='utf-8') as file:
    fieldsname = ['cast_id', 'movie_id', 'name', 'character', 'order']
    content = csv.DictWriter(file, fieldnames=fieldsname)
    content.writeheader()
    for item in movie_cast:
        content.writerow(item)

# primary key 인 cast_id 에 중복이 있어서 csv file 을 넣는데 에러 발생. 