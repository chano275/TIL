from pprint import pprint 
import requests
import csv


STR_ID = []
with open('movies.csv', 'r', encoding = 'utf-8') as file:
    content = csv.DictReader(file)
    for row in content:
        STR_ID.append((row['id']))

movie_details = []
fields = ['id', 'budget', 'revenue', 'runtime', 'genres']

for i in range(len(STR_ID)):
    url = "https://api.themoviedb.org/3/movie/" + STR_ID[i] + "?language=en-US"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkZDQ3MDMyZTIyNTE4YWQ1MTRkOGVlYmI5NGZkODAxYyIsIm5iZiI6MTcyMTM1NDk0Ny42MzU1NzUsInN1YiI6IjY2OTljOTIwZTU0NTNmOWE1NzBjYWZmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tw8AJBiRkWm_s6GNUTi15aZVBbWqRH58rIStX93ZGso"
    }
    response = requests.get(url, headers=headers).json()

    temp_item = {} 
    for chk in range(len(fields)): 
        if fields[chk] == 'genres':
            genres_desc = ''
            genres_list = (response[fields[chk]]) 
            for j in range(len(genres_list)):
                genres_desc += genres_list[j]['name']
                if j != len(genres_list) - 1:
                    genres_desc += ', '
            temp_item[fields[chk]] = genres_desc
        elif fields[chk] == 'id':
            temp_item['movie_id'] = int(STR_ID[i])
        else: 
            temp_item[fields[chk]] = response[fields[chk]] 
            
    movie_details.append(temp_item)

with open('movie_details.csv', 'w', newline='', encoding='utf-8') as file:
    fieldsname = ['movie_id', 'budget', 'revenue', 'runtime', 'genres']
    content = csv.DictWriter(file, fieldnames=fieldsname)
    content.writeheader()
    for item in movie_details:
        content.writerow(item)