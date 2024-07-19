from pprint import pprint 
import requests
import csv

url = "https://api.themoviedb.org/3/movie/popular?language=en-US&page=1"
headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJkZDQ3MDMyZTIyNTE4YWQ1MTRkOGVlYmI5NGZkODAxYyIsIm5iZiI6MTcyMTM1NDk0Ny42MzU1NzUsInN1YiI6IjY2OTljOTIwZTU0NTNmOWE1NzBjYWZmMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Tw8AJBiRkWm_s6GNUTi15aZVBbWqRH58rIStX93ZGso"
}
response = requests.get(url, headers=headers).json()
movies = []
fields = ['id', 'title', 'release_date', 'popularity']
for item in response['results']:
    temp_item = {}
    for key in fields:
        temp_item[key] = item[key]
    movies.append(temp_item)

with open('movies.csv', 'w', newline='', encoding='utf-8') as file:
    content = csv.DictWriter(file, fieldnames=fields)
    content.writeheader()
    for item in movies:
        content.writerow(item)
