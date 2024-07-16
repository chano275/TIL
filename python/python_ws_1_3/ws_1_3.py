users = [
    {"username": "alice", "age": 25, "is_active": True},
    {"username": "bob", "age": 17, "is_active": False},
    {"username": "charlie", "age": 30, "is_active": True},
    {"username": "david", "age": 22, "is_active": False},
    {"username": "eve", "age": 29, "is_active": True}
]

# 아래에 코드를 작성하시오.
def age_filter(people):
    ans = []
    for i in range(len(people)):
        if people[i]["age"] >= 18:
            ans.append(people[i])
    return ans



def tf_filter(people):
    ans = []
    for i in range(len(people)):
        if people[i]["is_active"] == True:
            ans.append(people[i])
    return ans


def age_tf_filter(people):
    ans = []
    for i in range(len(people)):
        if people[i]["is_active"] == True and people[i]["age"] >= 18:
            ans.append(people[i])
    return ans


if __name__ == '__main__':
    print(f'Adults: {age_filter(users)}')
    print(f'Active Users: {tf_filter(users)}')
    print(f'Adult Active Users: {age_tf_filter(users)}')