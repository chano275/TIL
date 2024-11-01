def format_string(text):# 문제 1-1: 문자열 포맷팅
    words = text.split()
    result = ''.join(word.capitalize() for word in words)
    return result
def select_even_index_elements(lst):return [lst[i] for i in range(len(lst)) if i % 2 == 0]# 문제 1-2: 짝수 인덱스 요소 선택
def remove_duplicates_and_sort(lst):return sorted(set(lst))# 문제 1-3: 중복 제거 및 정렬
print(format_string("hello world python"))  # "HelloWorldPython"
print(select_even_index_elements([1, 2, 3, 4, 5, 6]))  # [1, 3, 5]
print(remove_duplicates_and_sort([4, 2, 5, 2, 1, 4]))  # [1, 2, 4, 5]


def calculate_average(lst):  return sum(lst) / len(lst) if lst else 0# 리스트 컴프리헨션 
def is_prime(n):# 문제 2-2: 소수 판별
    if n < 2:return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:return False
    return True
def filter_primes(lst):return [num for num in lst if is_prime(num)]# 문제 2-3: 소수 필터링
print(calculate_average([1, 2, 3, 4, 5]))  # 3.0
print(is_prime(7))  # True
print(filter_primes([1, 2, 3, 4, 5, 6, 7]))  # [2, 3, 5, 7]

def remove_vowels(text):# 문제 1-1: 모음 제거
    vowels = "aeiouAEIOU"
    result = ""
    for char in text:
        if char not in vowels:result += char
    return result

def capitalize_words(text):# 문제 1-2: 단어 첫 글자 대문자 변환
    words = text.split()
    result = [word.capitalize() for word in words]
    return " ".join(result)

def count_word_occurrences(text, word):# 문제 1-3: 특정 단어 등장 횟수 세기
    words = text.split()
    count = 0
    for w in words:
        if w == word:count += 1
    return count
print(remove_vowels("hello world"))  # "hll wrld"
print(capitalize_words("hello world"))  # "Hello World"
print(count_word_occurrences("hello world, hello", "hello"))  # 2

def select_even(nums):return [num for num in nums if num % 2 == 0]# 문제 2-1: 짝수 선택
def remove_multiples_of_three(nums):return [num for num in nums if num % 3 != 0]# 문제 2-2: 3의 배수 제거
print(select_even([1, 2, 3, 4, 5, 6]))  # [2, 4, 6]
print(remove_multiples_of_three([1, 3, 6, 7, 9, 10]))  # [1, 7, 10]


def string_to_list(s):return list(s)# 문제 3-1: 문자열을 리스트로 변환
def list_to_string(lst):return ''.join(lst)# 문제 3-2: 리스트를 문자열로 변환
def split_words(s):return s.split()# 문제 3-3: 문자열을 공백 기준으로 분리
print(string_to_list("hello"))  # ['h', 'e', 'l', 'l', 'o']
print(list_to_string(['h', 'e', 'l', 'l', 'o']))  # "hello"
print(split_words("hello world"))  # ['hello', 'world']

def remove_key(d, key):# 문제 4-1: 딕셔너리에서 키 삭제
    if key in d:del d[key]
    return d
def add_key_value(d, key, value):# 문제 4-2: 딕셔너리에 키-값 쌍 추가
    d[key] = value
    return d
def filter_dict(d, threshold):return {k: v for k, v in d.items() if v >= threshold}  # 문제 4-3: 특정 기준 이상의 값만 선택
print(remove_key({'a': 1, 'b': 2, 'c': 3}, 'b'))  # {'a': 1, 'c': 3}
print(add_key_value({'a': 1}, 'b', 2))  # {'a': 1, 'b': 2}
print(filter_dict({'a': 1, 'b': 5, 'c': 10}, 5))  # {'b': 5, 'c': 10}

def intersection_set(set1, set2):return set1 & set2  # 문제 5-2: 교집합 계산
def union_set(set1, set2):return set1 | set2         # 문제 5-3: 합집합 계산
print(intersection_set({1, 2, 3}, {2, 3, 4}))  # {2, 3}
print(union_set({1, 2}, {3, 4}))  # {1, 2, 3, 4}
