# 아래에 코드를 작성하시오.

movies = ['Inception', 'Interstellar', 'Dunkirk', 'Tenet']

def get_movie_recommendation(rating):
    if rating >= 9:
        return movies[0]
    elif rating >=8 and rating < 9:
        return movies[1]
    elif rating >=7 and rating < 8:
        return movies[2]
    else:
        return movies[3]

print('Enter your movie rating (0-10): ', end = '')
a = int(input())
print(f'Recommended movie: {get_movie_recommendation(a)}')
