movies = ['Inception', 'Interstellar', 'Dunkirk', 'Tenet']
ratings = [9, 8.5, 7.5, 6]

# 아래에 코드를 작성하시오.



def get_high_rated_movies(ans_, threshold):
    ret = []
    for i in range(len(ans_)):
        if ans_[i]['rating'] >= threshold:
            ret.append(ans_[i]['title'])
    return ret

############################################
ans = []
for i in range(len(movies)):
    dict = {}
    dict['title'] = movies[i]
    dict['rating'] = ratings[i]
    ans.append(dict)
print(ans)

# zip 써서 
# print("******")
# print(list(zip(movies, ratings)))
# for title, rating in zip(movies, ratings):
#     temp_dict = {'title':title, 'rating':rating}
#     list.append(temp_dict)

# print("******")

############################################

print('Enter the rating threshold:', end = ' ')
a = float(input())
li = (get_high_rated_movies(ans, a))
print(f'Movies with a rating of {(a)} or higher')
for j in range(len(li)):
    print(li[j])