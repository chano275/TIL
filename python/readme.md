### 파이참 :

[https://married-spot-253.notion.site/Pycharm-5e8fffe90ef4427fa7bdc5eb25aed056](https://www.notion.so/Pycharm-5e8fffe90ef4427fa7bdc5eb25aed056?pvs=21)

유용한 단축키

> 단축키는 외운다기 보다는 자주 사용하면 익숙해 집니다.
단축키를 계속 사용하다 보면 편리한 기능을 쉽게 사용할 수 있습니다.
> 

유형별 단축키, 단축키 변경법

- **편집**
    - `Ctrl + /` : 한 줄 혹은 여러 줄 주석 처리
    - `Ctrl + X` : 한 문장 잘라내기
    - `Ctrl + Y` : 한 문장 삭제
    - `Ctrl + D` : 한 문장 복제
    - `Ctrl + Alt + L` : 코드 자동 정렬
    - `Alt + 마우스 클릭` : 다중 커서 생성 (맥은 `Option + 마우스 클릭`)
- **보기**
    - `Ctrl + Space` : 현재 자동완성 가능한 기능 보여줌
    - `Ctrl + 마우스 클릭` : 함수, 클래스가 작성된 곳으로 들어감
    - `Ctrl + Q` : 해당 키워드의 사용법을 보여줌
    - `Ctrl + F12` : 함수, 클래스의 계층 구조 보여줌 (모달 창)
    - `Alt + 7` : 함수, 클래스의 계층 구조 보여줌 (왼쪽 사이드바)
- **찾기**
    - `Ctrl + Shift + F` : 프로젝트 전체에서 코드 찾기
- **기타**
    - `Ctrl + Shift + F10` : Python 파일 실행 (맥은 `Ctrl + Shift + R`)
    - `Ctrl + Alt + ←, →` : 키보드 커서 이전, 이후 위치로 이동 
    (파일 찾기로 다른 곳으로 이동한 후, 원래 위치로 돌아갈 때 유용)

---

- 데이터 분석 최적화 라이브러리 > 판다스, 넘파이, matplotlib
- 대용량 데이터 처리 / 시각화 용이
- ML DL 모델 구현 > scikit-learn / TF / keras
- 인터프리터에 무엇을 전달하든, 그것이 *메인* 모듈이 된다. 이름이 무엇이든 상관없다.

### **`__main__` 확인**

- 모듈이 메인 스크립트로 작동하는지 알아보는 관례적인 방법이다.
위 `if` 문에 속한 문장이 *메인* 프로그램이 된다.

```python
# prog.py
...
if __name__ == '__main__':
    # 메인 프로그램으로서 실행 ...
    문장
    ...
```

---

# Data types

- type(변수명) : 해당 변수의 데이터 타입 확인 가능
- numeric : int / float / complex
- text seq : str
- **sequence : list / tuple / range**
- **non-seq : set / dict**
- Boolean / None / Func

### 가변 VS 불변            &           순서 여부

| 가변(mutable) | list, **set**, dict |
| --- | --- |
| **불변**(immutable) | int, float, bool, tuple, string, unicode |

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/42590fef-336a-4258-bcbb-0aa991f44b7d/Untitled.png)

---

### 지수 표현 방식

```python
# 314 ∗ 0.01
number = 314e-2
```

---

## **Sequence Types :** String / LIST / TUPLE / RANGE

- 여러 개의 값들을 순서대로 나열해 저장 [ 정렬 X ]
- 반복 가능 [ Iteration ]
- 인덱싱 : 각 값에 고유한 인덱스 가짐 / 인덱스 사용해 특정 위치 값을 선택 가능 / 수정 가능 ( 튜플, 문자열, range X )
- 슬라이싱 : 인덱스 범위 조절해 부분적인 값 추출 가능 / 새로운 시퀀스를 생성 

num_list**[n:]                   :  앞의 n개 *빼고* 출력** 
print(my_str[2:4])           : 글자의 INDEX 보는 것이 아니라, 요소 사이의 빈칸을 자른다고 생각 / 요소로 쉽게 생각하려면 [START : END - 1]

*** START를 비운다면 [:3] >> 앞에 0이 생략된 것. 
*** END를 비운다면, 3부터 마지막까지 
**** STEP : [0:5:2] 마지막 요소에 JUMP 할 숫자 넣기. 이때는 index로 생각
***** [::-1] >> 뒤부터 출력 [ Negative INDEXING 생각 ]

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dc8aa68a-f836-4f3c-94ba-caa08f79613d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cdbd0eea-fc75-43ff-90e2-04a07aa55c46/Untitled.png)

---

### seq 1. str :

- 순서 O  /  **변경 ‘불가’ :** my_str[1] = 'z'  # TypeError: 'str' object does not support …
- String Interpolation : 문자열 내에 변수 / 표현식 삽입하는 방법 print( f ‘’ )

- .endswith() : true / false 반환

---

### seq 2. LIST :

- 여러 개의 값 순서대로 저장하는 ‘변경 가능’한 시퀀스 자료형 / 대괄호 [ ] 로 표기
- 어떤 자료형이든 저장 가능 ( 새로운 리스트도 가능 )  /  ‘+’ 연산자를 이용해 붙일 수 있음

```python
my_list = [1, 2, 3, 'Python', ['hello', 'world', '!!!']]
print(len(my_list))  # 5
print(my_list[4][-1])  # !!!
print(my_list[-1][1][0])  # w
```

- 이차원 리스트 : 이차원배열 photo에서 각 줄을 k라는 변수에 저장
해당 k에서 i라는 변수를 이용해 각 인자를 순서대로 방문

```jsx
    for k in photo:
        for i in k:
```

---

### seq 3. Tuple

- **원소 체크 << [idx] 형태로**
- **‘변경 불가’   /   소괄호 ( ) 로 표기**
- 모든 자료형 저장 가능

```python
my_tuple = (1, 'a', 3, 'b', 5) # ?? 
# TypeError: 'tuple' object does not support item assignment
my_tuple[1]  <<  'a'
```

- 파이썬 내부 동작할 때 사용하는 dtype

```python
x, y = (10, 20)
print(x)  # 10
print(y)  # 20
# 파이썬은 쉼표를 튜플 생성자로 사용하니 괄호는 생략 가능
x, y = 10, 20
```

- **빈 튜플 : ( )
튜플에 할당할 요소 값 1개 >> (1, )**
이유 : 파이썬이 튜플인지 아닌지 확인하려고 / ( ) 이 연산자인지 아닌지를 판가름하는 요소

---

### seq 4. range :

- **‘변경 불가’**
- 연속된 **정수** 시퀀스 생성 / range(n) : 0 ~ n-1 정수 시퀀스  /  range(n, m) : n ~ m-1 정수 시퀀스
- range( 시작값 , 끝값 , **step** )
- STEP [ X : 1씩 증가 / 음수 : 감소 / 0 : 에러 ]

```python
# 리스트로 형 변환 시 데이터 확인 가능
print(list(range(5)))   # [0, 1, 2, 3, 4]
print(list(range(1, 10)))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# 주로 반복문과 함께 활용
for i in range(1, 10):       print(i)  # 1 2 3 4 5 6 7 8 9
for i in range(1, 10, 2):    print(i)  # 1 3 5 7 9
```

---

## Non - seq

### non-seq 1. dict

- key - value 쌍으로 이루어짐  /  중괄호 { } 로 표기  /  변경 가능
- **순서 X  /  중복 X**
- **KEY : 유니크함 > 변경 불가능한 자료형만 사용 가능 { STR, INT, FLOAT, TUPLE, RANGE }**
VALUE : 모든 자료형 사용 가능
    - **딕셔네리명 [ 키값 ] 이용해 요소 추가 / 변경   >>   이외에는 밑의 메서드 참조**

```python
my_dict = {'apple': 12, 'list': [1, 2, 3]}
print(my_dict['apple'])  # 12
print(my_dict['list'])  # [1, 2, 3]
# 추가
my_dict['banana'] = 50
print(my_dict) # {'apple': 12, 'list': [1, 2, 3], 'banana': 50}
# 변경
my_dict['apple'] = 100
print(my_dict) # {'apple': 100, 'list': [1, 2, 3], 'banana': 50}
```

### non-seq 2. Set

- 순서 X / **중복 X  /**  변경 가능 / 수학에서의 집합과 동일한 연산 처리 가능
- **중괄호 {} 로 표기 ( == DICT )**
- ***** 공집합 :  set() 로 써야 오류 X  >   {} → X**

```python
my_set_1 = {1, 2, 3}
my_set_2 = {3, 6, 9}
# 합집합******************
print(my_set_1 | my_set_2)  # {1, 2, 3, 6, 9}
# 차집합
print(my_set_1 - my_set_2)  # {1, 2}
# 교집합8*******************
print(my_set_1 & my_set_2)  # {3}
```

---

## None :

- 파이썬에서 값이 없음을 표현  /  함수의 return이 없는 경우 None을 반환

```python
variable = None
print(variable)  # None

def func():
    print('aaa')
print(func())  # None
```

---

## Boolean

- False / True로 사용해야 인식  ( 맨 앞글자 대문자 )

---

# 복사

- **변경 가능한 dtype**의 복사  >  **동일한거 가리킴**

```python
a = [1, 2, 3, 4]
b = a
b[0] = 100
print(a)  # [100, 2, 3, 4]
print(b)  # [100, 2, 3, 4]
```

- **변경 불가한 dtype**의 복사  >  새로운 객체 생성 > **값만 복사**

```python
a = 20
b = a
b = 10
print(a)  # 20
print(b)  # 10
```

---

## 복사 유형

1. 할당  :  객체 참조를 복사 > 동일한 객체 바라본다 

```
original_list = [1, 2, 3]
copy_list = original_list
print(original_list, copy_list)  # [1, 2, 3] [1, 2, 3]
copy_list[0] = 'hi'
print(original_list, copy_list)  # ['hi', 2, 3] ['hi', 2, 3]
```

1. **shallow copy : * SLICING 활용**! : 새로운 시퀀스 생성하기 때문에 저장되는 메모리 영역도 아예 새로움  >  한계 : 2차원 리스트와 같이 리스트 안에 리스트 들어가있으면 가장 안쪽에 들어가 있는 요소는 할당 형식으로 복사되는 듯 .. ? << 리스트의 최상위 요소들에 대해서만 새로운 seq 생성한대 

```python
a = [1, 2, 3] 
b = a[:]
print(a, b)  # [1, 2, 3] [1, 2, 3]
b[0] = 100
print(a, b)  # [1, 2, 3] [100, 2, 3]

# 한계 : 
a = [1, 2, [1, 2]]
b = a[:]
print(a, b)  # [1, 2, [1, 2]] [1, 2, [1, 2]]
b[2][0] = 100
print(a, b)  # [1, 2, [100, 2]] [1, 2, [100, 2]] 
```

1. deep copy : 모듈 사용해서 아예 새로운 객체로 

```python
import copy
original_list = [1, 2, [1, 2]]
deep_copied_list = **copy.deepcopy**(original_list)
deep_copied_list[2][0] = 100
print(original_list)  # [1, 2, [1, 2]]
print(deep_copied_list)  # [1, 2, [100, 2]]
```

---

# 암시적 형변환

- 파이썬이 내부적으로 동작할 때에 자동으로 형변환 해주는 것

```python
print(3 + 5.0)  # 8.0 자연스럽게 float 
print(True + 3)  # 4 자연스럽게 true == 1
print(True + False)  # 1
```

# 명시적 형변환

- 개발자가 직접 형변환

```python
print(int('1'))  # 1
# ValueError: invalid literal for int() with base 10: '3.5'
print(int('3.5'))

print(int(3.5))  # 3
print(float('3.5'))  # 3.5

print(str(1) + '등')  # 1등
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/39dc1a2f-eaa5-4342-a483-f1e90edf397a/Untitled.png)

- **++, — 파이썬에서는 지원 X**             &            // 몫             &            ** 거듭제곱 ( 지수 )

---

- is

```python
# SyntaxWarning: "is" with a literal. Did you mean "=="?
# **==은 값(데이터)을 비교하는 것이지만 is는 레퍼런스(주소)를 비교**하기 때문
# 아래 조건은 항상 False이기 때문에 is 대신 ==를 사용해야 한다는 것을 알림
print(1 is True)  # False
print(2 is 2.0)  # False
```

---

## 단축평가

- 논리 연산에서 두번째 피연산자 평가하지 않고, 결과를 결정하는 동작

```python
vowels = 'aeiou'

# and는 둘다 true 일때 뒤에 요소를 반환 
print(('a' and 'b') in vowels)  # False
print(('b' and 'a') in vowels)  # True

**# or 는 앞에 요소만 보고 찍는다** 
print(('a' or 'b') in vowels)  # True
print(('b' or 'a') in vowels)  # False

# 정수 : [숫자 0, 음수] - false / [양] - true ( 그값을 찍음 ) 
# and 는 두개 다 맞으면 뒤에까지 보니까 뒤에 숫자를 찍음
print(3 and 5)  # 5
print(3 and 0)  # 0
print(0 and 3)  # 0
print(0 and 0)  # 0

# or는 앞이 true 면 바로 찍으니까 
print(5 or 3)  # 5
print(3 or 0)  # 3
print(0 or 3)  # 3
print(0 or 0)  # 0
```

---

## 멤버십 연산자 in & not in

- 특정 값이 시퀀스나 다른 컬렉션에 속하는지 여부 확인

```python
word = 'hello'
numbers = [1, 2, 3, 4, 5]
print('h' in word)  # True
print('z' in word)  # False
print(4 not in numbers)  # False
print(6 not in numbers)  # True
```

---

## 시퀀스형 연산자

- +, *

```python
# Gildong Hong
print('Gildong' + ' Hong')
# hihihihihi
print('hi' * 5)

# [1, 2, 'a', 'b']
**print([1, 2] + ['a', 'b'])**
# [1, 2, 1, 2]
**print([1, 2] * 2)**
```

## 연산자 우선순위

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dc6a6556-4328-4874-85a3-ace60e6c26b4/Untitled.png)

---

# 제어문

```python
# if / elif / else 한줄에 
def solution(angle):
answer = 0
return 1 if 0<angle<90 else 2 if angle==90 else 3 if 90<angle<180 else 4
```

## for :

- 반복 횟수 명확히 정해져 있을 경우 유용  /
- **이터러블 [[            seq [ 리스트 / 튜플 / 문자열 등 ] + DICT + SET               ]] 데이터 처리 시**
- for  변수 in 반복가능객체:

```python
items = ['apple', 'banana', 'coconut']
for item in items:
    print(item)
```

### 중첩리스트 순회 :

```python
elements = [['A', 'B'], ['c', 'd']]
for elem in elements:
    for item in elem:
        print(item)
```

## while :

- 반복 횟수 불명확하거나  //  특정 조건 따라 반복 종료해야 할 때 유용

---

## continue :

- 다음 반복으로 건너뜀

```python
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # 1 3 5 7 9
```

## ** List Comprehension :

- 간결하고 효율적인 리스트 생성법
- **[ expression for 변수 in iterable ]**
- **[ expression for 변수 in iterable if 조건식 ]**
- * 2차원 배열 생성 시 유용

```python
numbers = [1, 2, 3, 4, 5]

squared_numbers = [num**2 for num in numbers]
print(squared_numbers)  # [1, 4, 9, 16, 25]

data1 = [[0] * (10) for _ in range(10)]
# 또는
data2 = [[0 for _ in range(10)] for _ in range(10)]
```

---

# 함수 func

- 파이썬에서의 함수 : 일급 객체로 사용 가능
다른 변수에 할당될 수 있음 
다른 함수의 인자로 전달될 수 있음
다른 함수에 의해 반환될 수 있음
- return 은 하나의 객체만 반환! ( list, dict, set, int, str, … )
- 익명 함수로 사용 가능 [ 람다 표현식 ]

```python
def make_sum(pram1, pram2): # 매개변수 
    """이것은 두 수를 받아
    두 수의 합을 반환하는 함수입니다.
    >>> make_sum(1, 2)
    3
    """
    return pram1 + pram2
    
make_sum(a, b) # 인자 
```

---

## 매개변수 : Parameter

- 함수 내부에서 사용할 변수 / 함수 정의할 때에, 함수가 받을 값 나타내는 변수

---

## 인자 : Arguments

- 함수 호출할 때에, 실제로 전달되는 값
- **위치** 인자 : 함수 호출시 **인자의 위치**에 따라 전달되는 인자 / 반드시 값을 전달해야 함 
                  * 우리가 평소에 쓰는 함수 인자 *
- **기본 인자 값** : 함수 정의에서 매개변수에 기본 값 할당 / 인자 안오면 기본값 매개변수에 할당

```python
def greet(name, age=30):
    print(f'안녕하세요, {name}님! {age}살이시군요.')
greet('Bob')  # 안녕하세요, Bob님! 30살이시군요.
greet('Charlie', 40)  # 안녕하세요, Charlie님! 40살이시군요.
```

- **키워드** 인자 : 함수 **호출 시 인자의 이름과 함께 값 전달**하는 인자
매개변수와 인자 일치시키지 않고, 특정 매개변수에 값 할당 가능
인자의 **순서 중요 X, 인자의 이름 명시하여 전달**  BUT **호출 시 키워드 인자는 위치 인자 뒤에**

```python
def greet(name, age):
    print(f'안녕하세요, {name}님! {age}살이시군요.')
greet(**name='Dave'**, age=35)   # 순서 중요하지 않음! 

**greet(age = 35, 'Dave' ) # XXXXXXXX**
```

- **임의의 인자 목록** : **정해지지 않은 개수의 인자 처리**하는 인자 
함수 정의 시 **매개변수 앞에 *  > 여러개의 인자를 튜플로** 처리

```python
def calculate_sum(*args):
    print(args)
    total = sum(args)
    print(f'합계: {total}')
calculate_sum(1, 2, 3)
```

- 임의의 **키워드** 인자 목록 : 정해지지 않은 개수의 **키워드** 인자 처리하는 인자
함수 정의 시 **매개변수 앞에 ** / 여러개의 인자를 dict 로 묶어 처리**

```python
def print_info(**kwargs):
    print(kwargs)
print_info(name='Eve', age=30)  # **{'name': 'Eve', 'age': 30}**
```

## 함수 인자 권장 작성 순서

- **위치 > 기본 > 가변 > 가변 키워드**

```python
def func(**pos1, default_arg='default', *args, **kwargs**):
    print('pos1:', pos1)
    print('pos2:', pos2)
    print('default_arg:', default_arg)
    print('args:', args)
    print('kwargs:', kwargs)
func(1, 2, 3, 4, 5, 6, key1='value1', key2='value2')
```

---

## 패킹 :

- 여러개의 값을 하나의 변수에 묶어서 담는 것   /  변수에 담긴 값들은 튜플 형태로 묶임

```python
packed_values = 1, 2, 3, 4, 5
print(packed_values)  # (1, 2, 3, 4, 5)

def my_func(*objects):
    print (objects) # (1, 2, 3, 4, 5)
    print(type(objects)) # <class 'tuple'>
my_func(1, 2, 3, 4, 5)
# (1, 2, 3, 4, 5)
# <class 'tuple'>
```

- ***b는 남은 요소들을 리스트로 패킹하여 할당**

```
numbers = [1, 2, 3, 4, 5]
a, *b, c = numbers
print(a)  # 1
print(b)  # [2, 3, 4]
print(c)  # 5
```

## 언패킹 :

- 패킹된 변수의 값을 개별적 변수로 분리하여 할당
- 튜플이나 리스트 등의 객체의 요소들을 개별 변수에 할당

```python
packed_values = 1, 2, 3, 4, 5
**a, b, c, d, e = packed_values**
print(a, b, c, d, e)  # 1 2 3 4 5
```

- * 는 리스트의 요소를 언패킹하여 인자로 전달 : 
리스트 하나를 보냈는데 3개 들어가니까 언패킹!!!

```python
def my_function(x, y, z):
    print(x, y, z)

names = ['alice', 'jane', 'peter']
my_function(***names**) # alice jane peter
```

- ****는 dict의 키 - 값 쌍을 언패킹하여 함수의 키워드 인자로 전달**

```python
def my_function(x, y, z):
    print(x, y, z)
my_dict = {'x': 1, 'y': 2, 'z': 3}
my_function(**my_dict)  # 1 2 3
```

---

# *, **, 패킹, 언패킹 연산자 정리

- * 
패킹 > 여러개의 인자를 하나의 튜플로 묶음 
언패킹 > 시퀀스, 반복 가능한 객체를 각각의 요소로 언패킹해 함수의 인자로 전달
- ** 
**언패킹 > dict 의 키 - 값 쌍을 언패킹 > 함수의 키워드 인자로 전달**

---

## 메서드

- [ dtype 객체 . 메서드명() ]
- **객체에 속한 함수** / 객체의 상태를 조작하거나 동작 수행
- **클래스 내부에 정의되는 함수**

### 문자열 메서드 :

- ‘불변 특징’ > **원본 변경 X 변수에 할당**해주어야 함!
1. replace : 

```python
text = 'Hello, world!'
new_text = text.replace('world', 'Python')
print(new_text)  # Hello, Python!
```

1. strip : 좌우 공백 삭제  /  () 안에 문자 넣으면 해당 문자 제거 

```python
text = '  Hello, world!  '
new_text = text.strip()
print(new_text)  # 'Hello, world!'
```

1. split : 인자로 받은 문자열 기준 왼오 2개로 쪼갬  >>  리스트로 할당해서 사용 

```python
text = 'Hello, world!'
words = text.split(',')
print(words)  # ['Hello', ' world!']
```

- split(    )     : 인자(구분자) X  >>  모든 종류의 공백문자로 나눔
- split(" ") : \n \t (탭키) 나누지 못함, 공백문자가 연속으로 나오는 경우 빈 문자열 추가

1. join : 
- 구분자 + 메서드 >> 구분자를 기준으로 왼쪽 오른쪽에 문자열 붙임

```python
words = ['Hello', 'world!']
text = '-'.join(words)
print(text)  # 'Hello-world!'
```

1. capitalize : 문자열 가장 앞의 글자 대문자로 

---

### 리스트 메서드

1. append : 

```python
my_list = [1, 2, 3]
my_list.append(4)
print(my_list)  # [1, 2, 3, 4]
```

1. extend : 인자가 반복 가능하다면, 해당 객체가 가지고 있는 모든 요소 순회하며 리스트에 넣음 / 여러가지 인자 넣을때 사용  /  **+= 과 같은 기능** 

```python
my_list = [1, 2, 3]
my_list.extend([4, 5, 6])
print(my_list) # [1, 2, 3, 4, 5, 6]
```

1. pop : list 
- 인자 없으면 맨 마지막 pop  /  인자 있으면 해당 idx pop
- pop한 객체를 어떤 변수에 저장 가능

```python
my_list = [1, 2, 3, 4, 5]
item1 = my_list.pop()
item2 = my_list.pop(0)
print(item1)  # 5
print(item2)  # 1
print(my_list)  # [2, 3, 4]
```

1. reverse : 

```python
my_list = [1, 3, 2, 8, 1, 9]
my_list.reverse()
print(my_list)  # [9, 1, 8, 2, 3, 1]
```

1. sort : 
- .sort(key=len) : 키값에 따라 정렬

```python
my_list = [3, 2, 1]
my_list.sort()
print(my_list)  # [1, 2, 3]
my_list.sort(reverse=True)
print(my_list)  # [3, 2, 1]
```

---

### dict 메서드

1. get : 
- 키에 연결된 value 반환
- 대괄호로 dict 보는거랑 다른거 ? > 대괄호는 키가 없을 때에 에러 / **get**은 없으면 **none 값 반환**

```python
person = {'name': 'Alice', 'age': 25}
print(person.get('name'))  # 키에 연결된 값 반환 / 키 없으면 None
print(person.get('country', 'Unknown')) # 키 없으면 > none이 아니라 unknown 반환 두번째 인자 
```

1. keys  &   values :
- 해당 딕셔네리 키 / value  만 모아서 리스트로 반환

```python
person = {'name': 'Alice', 'age': 25}
print(person.keys())  # dict_keys(['name', 'age’])
for k in person.keys():
    print(k)

    
print(person.values())  # dict_values ([‘Alice’, 25])
for v in person.values():
    print(v)
```

1. items : 
- 키, 벨류 둘다 필요할때 / **튜플 형태로 반환**
- 반복문 통해서 주로 사용

```python
person = {'name': 'Alice', 'age': 25}
print(person.items())  # dict_items([('name', 'Alice'), ('age', 25)])
for k, v in person.items():
    print(k, v)
```

1. pop : dict

```python
person = {'name': 'Alice', 'age': 25}

print(person.pop('age'))  # 25 
# 해당하는 키값 제거 > 연결됐던 값 반환
print(person)  # {'name': 'Alice'}

print(person.pop('country', None))  # None
print(person.pop('country'))  # KeyError
# 키값 없으면 none 반환하라고 2번째 인자를 줘야 함 / 없으면 에러 발생
```

---

### Set 메서드

1. add : 
- 집합이기 때문에 중복은 의미가 없다

```python
my_set = {'a', 'b', 'c', 1, 2, 3}
my_set.add(4)
print(my_set)  # {1, 'b', 3, 2, 'c', 'd', 'a’}
my_set.add(4)
print(my_set)  # {1, 'b', 3, 2, 'c', 'd', 'a’}
```

1. remove : 
- 없는 요소 제거하면 키 에러

```python
my_set = {'a', 'b', 'c', 1, 2, 3}
my_set.remove(2)
print(my_set)  # {'b', 1, 3, 'c', 'a'}
my_set.remove(10)
print(my_set)  # KeyError
```

---

## map

- **map(func, iterable)**
- 이터러블한 데이터구조의 모든 요소에 함수 적용 > MAP OBJECT로 반환
- **list(map(int, input().split()))**

---

## zip(*iterables)

- 임의의 iterable 을 모아, **튜플을 원소로 하는 zip object 반환**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/09db2d33-5a59-4ef6-8121-6bde38b4f061/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/828d5a77-c6cc-42f3-9bc9-2bc63c014c2a/Untitled.png)
    
- list로 빼서 볼 수 있고, idx 안맞는거는 버림

---

# Lambda 표현식

- 설명 : 익명 함수를 만드는데 사용되는 표현식
- 기본 구조 : **[ lambda 매개변수 : 표현식 ]**   >  한줄로 간단한 함수 정의
- lambda 키워드 : 람다 함수 선언 위해 사용되는 키워드
- 매개변수 : 함수에 전달되는 매개변수들
                 여러 개의 매개변수가 있을 경우, 쉼표로 구분
- 표현식 : 함수의 실행되는 코드 블록 / 결과값을 반환하는 표현식으로 작성

- 람다 표현식 예시

```python
def addition(x, y):
	return x + y
result = addition(3,5)
print(result)

###

addition = lambda x, y: x + y
result = addition(3,5)
print(result)

## map 함수 응용
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x*x, numbers))
print(squares) # [1,4,9,16,25]

```

- cf>  sorted() 함수와 lambda 함수 사용

파이썬의 내장 함수인 sorted() 함수를 사용 > dict 키/값 정렬 가능 
sorted() 함수 : list / tuple / dict 같은 iterable 객체 받아 정렬된 리스트 반환

dict 정렬 : sorted() 함수에 dict의 items() 메소드를 이용 > (키, 값) 쌍의 list 만들고
정렬 기준으로 사용할 값을 선택하여 lambda 함수를 만들어 sorted() 함수의 key 인자로 전달

```python
d = {'apple': 3, 'banana': 2, 'cherry': 1}

# 키를 기준으로 정렬
sorted_d = sorted(d.items(), key=lambda x: x[0])
print(sorted_d)

# 값(value)를 기준으로 정렬
sorted_d = sorted(d.items(), key=lambda x: x[1])
print(sorted_d)
```

위 코드에서, lambda 함수는 (키, 값) 쌍을 인자로 받아 각각의 키 또는 값에 대한 정렬 기준을 반환 > 이 예제에서는 키를 기준으로 정렬한 뒤, 값을 기준으로 정렬한 결과를 출력

---

- 참고 : 파이썬 스타일 가이드 
1. 변수명 무엇을 위한 것인지 직관적으로 / 공백 4칸 들여쓰기 / 한줄 79자 
2, 함수/클래스 정의하는 블록 사이에는 빈 줄 추가

---

# Enumerate

- emumerate(iterable, start = 0) : 
iterable 객체의 각 요소에 대해 [[ **인덱스 ]] 와 함께 반환**하는 내장함수

```python
fruits = ['apple', 'banana', 'cherry']

for index, fruit **in enumerate(fruits):**
	print(f'인덱스 {index}: {fruit}')

'''
인덱스 0: apple
인덱스 1: banana
인덱스 2: cherry
'''
```

---

## global

- 변수의 스코프 전역 범위로 지정하기 위해 사용  /  보통 함수 내에서 전역변수 수정하려는 경우 사용
- 매개변수에 global 키워드 사용 불가

```python
num = 0
def inc():
	global num # 전역변수 선언
	num += 1

print(num) # 0
inc()
print(num) # 1 
```

- 변수 수명주기 : 
빌트인 scope : 파이썬 실행 이후부터 영원히 유지
global scope : 모듈 호출된 시점 / 인터프리터 끝날 떄까지 유지
local scope : 함수 생성시 / 종료시
- 이름 검색 규칙 : LEGB Rule << 이름공간 순서대로 찾아 나가는 것

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/75f701ac-ebb6-454c-881b-edb3e3516bf9/Untitled.png)

- **막줄 핵심! 저거때문에 함수에 인자 안넘겨도 참조할 수 있는 듯**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2d1c3260-731e-46dd-ba79-ff2e2b60cf24/Untitled.png)

---

# 객체 :

- 클래스에서 정의한 것 토대로 메모리에 할당된 것.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/77f5252e-66f5-4e89-a932-02a72b31308b/Untitled.png)

- **속성(정보 / 변수) + 행동(동작 / 메서드)**으로 구성된 모든 것 
속성(데이터) + 기능(메서드)
- **클래스로 만든 객체 == 인스턴스**

- ex> 가수의 속성(정보) - 직업, 생년월일, 국적 > 변수
                   행동(메서드) - 랩(), 댄스()             > 메서드 

가수(클래스) → 객체(아이유) / 아이유는 가수의 인스턴스다 ok / **아이유가 인스턴스다 X ????**

- **하나의 객체는 특정 type 의 인스턴스다.  == 변수 NAME 은 STR 클래스의 인스턴스이다**

- 객체 정리
1. 타입(type) : 어떤 연산자와 조작이 가능? 
2. 속성(attribute) : 어떤 상태(데이터)를 가지는가?
3. 조작법(method) : 어떤 행위(함수)를 할 수 있는가? == 기능

## 클래스 & 객체 :

### name = ‘Alice’

- 변수 name의 타입은 str 클래스다.
- 변수 name은 str 클래스의 객체(인스턴스)다

### ‘hello’.upper()

- 객체.행동()
- 인스턴스.메서드()

---

# 클래스 :

- 속성(변수) +  메서드(함수)
- print(help(str)) : class 라는걸 확인 가능  >>  **현재 만들어져 있는 타입 = 클래스**
- 파이썬에서 타입 표현하는 방법 / 클래스를 만든다 == 타입을 만든다
- 객체 생성 위한 설계도 / 데이터와 기능 함께 묶는 방법 제공

```python
class 클래스명: #클래스명 : 괄호 없이 class 이용하여 선언
	pass
	
### **Class 안의 함수 = 메서드**
class Person:
    blood_color = 'red'
    def __init__(self, name):
        self.name = name
    def singing(self):
        return f'{self.name}가 노래합니다.'
        
# singer1 이라는 Person class의 인스턴스 생성
# init에 name 줬으므로, singer1의 self.name = 'iu'
singer1 = Person('iu') 

# 메서드 호출 / 위에서 self.name 만들었으므로
# singer1.singing() 호출하면 iu가 노래합니다 return
print(singer1.singing())  

# 속성(변수) 접근도 가능 
print(singer1.blood_color)
```

---

## 클래스 구성요소

### 생성자 함수 : __init__ (self, name)

- 인스턴스 메서드와 동일하게 동작 > 첫번째 매개인자 self
- 인스턴스 객체를 **‘생성할 때’ 자동으로 호출**되는 특별한 메서드  /  init 메서드로 정의, **객체 초기화**
- **생성자 함수 통해 인스턴스 생성, 필요한 초기값 생성**

### 인스턴스 변수 : self.name

- 생성자 함수 안에 들어있던 변수 ?
- 인스턴스별로 유지되는 변수 / 인스턴스마다 독립적인 값, 인스턴스 생성될 때마다 초기화

### 클래스 변수 : blood_color

- 클래스 내에 있는 변수 / 클래스로 생성된 모든 인스턴스가 공유

>> 생성자 함수에 클래스 변수 +1을 놓으면, 몇개의 인스턴스가 생겼는지 count 가능
self.name = name 
**Person.count += 1**
- 단점 : 클래스변수 [Circle.p](http://Circle.py)i 변경 시 모든 인스턴스에서 해당 값이 변함 
** BUT c1.pi 같이 인스턴스에서 변경은 하지 못하고, 읽었던 값을 저장만 함 ??????

### 인스턴스 메서드 : singing(self)

- 각 인스턴스에서 호출 가능한 메서드 / 함수 만드는 것과 동일.
- ***** 반드시 첫 번째 매개변수로 인스턴스 자신(self)를 전달받음!**
- **인스턴스 변수에 접근하고, 수정**하는 등의 작업 수행

---

## 인스턴스와 클래스 간의 이름 공간

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d546c1bf-bd83-4c6c-8278-aef27da4f601/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/77ac1c34-5482-452c-98dc-067894417ff9/Untitled.png)

- 클래스 변수 인스턴스에서 컨트롤하면, 전체 클래스 변수 값이 바뀌는게 아니라
해당 인스턴스에만 그 값이 할당된다고 생각하면 될듯
- 인스턴스에서 클래스 이름공간을 **조회 가능 BUT , 수정 불가**
- func의 이름 공간과는 다름 < ??????????

---

### cf> 파이썬 끝나고 인스턴스 제거시 소멸자

def __ del __(self): print(’악’) # del 키워드로 강제 소멸 가능 

- dir ( ) 내장함수 사용해 해당 type이 가지고 있는 메서드, 속성 등등 확인 가능 / 매직메서드는 파이썬 코드가 실행될 때에 내부 설계적으로 실행에 필요한 로직들이 담겨있어 별도로 정의하지 않아도 class 정의시 기본 class 상속받아 기본적인 매직메서드는 모두 정의되어 있는 상태이다.

---

## ********* 클래스 메서드 :

- 클래스가 호출하는 메서드 :
- **인스턴스 메서드와의 차이? : 클래스 변수 조작, 클래스 레벨의 동작 수행 가능  

의미 :** 클래스 메서드를 사용하는 이유? >> 인스턴스 메서드로도 물론 클래스 변수와 같은 객체에 접근할 수 있지만, 클래스 메서드를 통해서 접근하는게 개발자가 나중에 코드를 봤을때 클래스의 요소들을 클래스 메서드에서 컨트롤하는게 보이니까 체크하기에 직관적이어서

근데 라고 하기에는 위에 img 보면 그것도 아닌거같은디…

#### 클래스 매서드 << 클래스 변수 컨트롤할때 어떤 식으로 하는지 찾아보기
cls.total_movies += 1 맞나 ?

- 함수 선언 : @classmethod                             # **데코레이터 필수** 
                     def class_method(cls, arg1, … ):   # **첫번째 매개변수 : cls**

```python
class Person:
    count = 0
    def __init__(self, name):
        self.name = name
        Person.count += 1
##################################################################
    @classmethod
    def number_of_population(cls):     # cls에는 Person 클래스가 전달된다. 
        print(f'인구수는 {**cls.count**}입니다.')
##################################################################

person1 = Person('iu')
person2 = Person('BTS')
```

---

## ********** 정적 메서드 :

- 클래스, 인스턴스 상관없이 독립적으로 동작하는 메서드 :
                                   주로 클래스와 관련 O / 인스턴스와 상호작용 필요하지 않을 때에 사용

- 함수 선언 : @staticmethod  # **데코레이터 사용**
                     def static_method(arg1, …)  # **필수 매개변수 X // 매개변수 받아도 된다!**
- 단순히 문자열 조작하는 기능 제공 ( 클래스에 대한 정의 )

```python
class StringUtils:
#############################################
    @staticmethod
    def reverse_string(string):
        return string[::-1]
        
    @staticmethod
    def capitalize_string(string):
        return string.capitalize()
#############################################

text = 'hello, world'
reversed_text = StringUtils.reverse_string(text)
print(reversed_text)  # dlrow ,olleh
capitalized_text = StringUtils.capitalize_string(text)
print(capitalized_text)  # Hello, world
```

---

# 상속

## 클래스 상속

- Person 이라는 클래스에 교수, 학생 같이 인스턴스 생성한다면, 교수만의 정보 담기 어려움
그렇다고 교수 / 학생 2개의 클래스 만들기에는 중복이 많음

- 이럴때, Person의 클래스에 공통적인 부분을 두고, 
Person을 상속받아 교수와 학생이라는 새로운 자식 class 생성

- 상속받을 class를 class 선언 시에 ( ) 안에 입력

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
        
    def talk(self):  # 상속받은 자식들이 사용할 메서드
        print(f'반갑습니다. {self.name}입니다.')
       
							  **##### Person 상속받음**
class Professor(Person): 
    def __init__(self, name, age, department):
        self.name = name
        self.age = age
        self.department = department # 교수만의 특별한 인스턴스 변수 
        
class Student(Person): # Person 상속받음 
    def __init__(self, name, age, gpa):
        self.name = name
        self.age = age
        self.gpa = gpa               # 학생만의 특별한 인스턴스 변수 

p1 = Professor('박교수', 49, '컴퓨터공학과')
s1 = Student('김학생', 20, 3.5)

#############################################################
# 부모 Person 클래스의 talk 메서드를 활용
**p1.talk()**  # 반갑습니다. 박교수입니다.
**s1.talk()**  # 반갑습니다. 김학생입니다.
#############################################################
```

---

## 다중 상속

- 2개 이상의 상위 class로 부터 여러 행동이나 특징 상속받을 수 있는 것
- 상속받은 모든 class의 요소 활용 가능
- **중복된 속성 / 메서드 >> 상속 순서에 의해 결정됨**

```python
class Person:
    def __init__(self, name):
        self.name = name
    def greeting(self):
        return f'안녕, {self.name}'
        
        
class Mom(Person):
    gene = 'XX'
    def swim(self):
        return '엄마가 수영'
        
        
class Dad(Person):
    gene = 'XY'
    def walk(self):
        return '아빠가 걷기'

class FirstChild(Dad, Mom): # **다중 상속 > mom 의 swin / dad의 walk 모두 사용 가능** 
    def swim(self):         # **겹치는 메서드 : 자식이 이김** 
        return '첫째가 수영'
    def cry(self):
        return '첫째가 응애'
        
        
baby1 = FirstChild('아가')
print(baby1.cry())  # 첫째가 응애
print(baby1.swim())  # 첫째가 수영
print(baby1.walk())  # 아빠가 걷기
print(baby1.gene)  # XY                

""" 
Mom, Dad 클래스의 동일한 이름의 클래스 변수 gene : 
FirstChild > Dad > Mom 순서로 gene 찾아봄 
FirstChild 찾았는데 없고, Dad에 있다 > Dad 의 gene 사용 

자식 > 상속된 순서로 요소 찾아봄 
"""
```

---

### 다이아몬드 문제 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a1d26258-9311-4aaa-a416-dc21a736d074/Untitled.png)

- 두 클래스 B, C가 A에서 상속     +     D가 B,C 모두에서 상속될 때에 발생하는 모호함
    - 파이썬에서의 해결책 : 
    
    **MRO(Method Resolution Order) 알고리즘** 사용 :
     1.  MRO 알고리즘 사용하여 클래스 목록 생성
     2. 부모 클래스로부터 상속된 속성들의 검색을 **DFS**로, 왼 → 오 로, 
                                                                                      계층 구조에서 겹치는 같은 클래스를 두번 검색 X
    
    ***** DFS라고 했지만 다이아몬드 구조에서 DBAC 가 아니라 DBCA 인 이유? 
    C의 다음 MRO가 A이면 ( B, C가 동일한 클래스에 상속된다면 )  DBCA**
- 그래서, Dad, Mom 의 gene : 왼쪽 > 오른쪽 순서

---

## **메서드 오버라이딩 (Method Overriding)**

- 의미 : **부모 클래스에서 정의된 메서드를 자식 클래스에서 재정의**하는 것
- 자식 클래스에서 부모 클래스의 메서드를 재정의 > 자식 클래스에 맞게 동작 변경 가능
- **오버라이딩된 메서드 > 부모 클래스의 메서드와 동일한 이름과 매개변수를 가져야**
- **super() 키워드**를 사용하여 **부모 클래스의 메서드를 호출 가능**

### super() :

- 부모 클래스 객체를 반환하는 내장 함수
- **다중 상속** 시 **MRO 기반**으로 현재 클래스가 상속하는 모든 부모 클래스 중 
**다음에 호출될 메서드 결정하여 자동 호출**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/225405fb-8fd2-401d-be18-e8646e55baf2/Untitled.png)

```python
"""

단일 상속 : 
2개 위의 코드에서 Person의 init에 받는 것을 Student도 동일한 코드로 그대로 받는데, 
Student가 Person을 상속 받고, init 내에 super().__init__(self 제외한 매개변수) 입력 +
Student에서 추가로 받는 변수 선언

"""

class Person:
    def __init__(self, name, age, number, email):
        self.name = name
        self.age = age
        self.number = number
        self.email = email

class Student(Person):
    def __init__(self, name, age, number, email, student_id):
        
        **##################################**
        **# Person의 init 메서드 호출 - 상속하고 있는거 한개이므로 MRO 생각 X** 
        super().__init__(name, age, number, email) # 동일한 이름, 매개변수 
        **##################################**
        
        self.student_id = student_id
        
   

##################################################################

# 다중 상속 : **MRO 상으로 ParentA가 우선 > ParentA 클래스의 init 메서드 호출**
class ParentA:
    def __init__(self):
        self.value_a = 'ParentA'
    def show_value(self):
        print(f'Value from ParentA: {self.value_a}')
        
        
class ParentB:
    def __init__(self):
        self.value_b = 'ParentB'
    def show_value(self):
        print(f'Value from ParentB: {self.value_b}')
        

class Child(ParentA, ParentB):
    def __init__(self):
        super().__init__() # ParentA 클래스의 __init__ 메서드 호출
        self.value_c = 'Child'
    def show_value(self):
        super().show_value() # ParentA 클래스의 show_value 메서드 호출
        print(f'Value from Child: {self.value_c}')

```

### SUPER 사용 사례

- 단일 상속 구조 : 명시적 이름 지정 X해도 부모클래스 참조 가능 > 코드 유지보수 용이
                               클래스 이름 변경, 부모 클래스 교체되어도 super() 사용 > 코드수정 적게 필요

- 다중 상속 구조 : MRO를 따른 메서드 호출
                              복잡한 다중 상속 구조에서 발생할 수 있는 문제 방지

---

### MRO 필요 이유? :

- 부모 클래스들이 여러번 엑세스되지 않도록 
 1. 각 클래스에서 지정된 왼 > 오 순서 보존하고,
 2. 각 부모를 오직 한번만 호출하고
 3. **부모들의 우선순위에 영향을 주지 않으면서 서브 클래스를 만드는 단조적 구조 형성**
- 프로그래밍 언어의 신뢰성 있고 확장성 있는 클래스 설계 가능
- 클래스 간의 메서드 호출 순서가 예측 가능하게 유지되며, 코드의 재사용성과 유지보수 향상

---

## 클래스 참고

### 메서드 주의사항 :

- 누가 어떤 메서드 사용 ? : 클래스      > 클래스 / 스태틱 메서드 
                                                인스턴스 > 인스턴스 메서드

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c1cc3ed6-72de-4e69-b521-0eed25bab3cd/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/79011cd3-e77a-4d10-96a7-9e0ed13a59e2/Untitled.png)

- 클래스 :      모든 메서드 호출 가능 BUT! **클래스 / 스태틱 메서드만** 사용하도록
- 인스턴스 : 모든 메서드 호출 가능 BUT! **인스턴스 메서드만** 사용하도록

---

### 매직 메서드 ( 스페셜 메서드 ) :

- 인스턴스 메서드 >> **self 인자로** 받음
- 특정 상황에 자동으로 호출된다 < 외에는 일반 함수와 동일하다 > 기본인자를 추가할 수도 있다
- **Double underscore __ 있는 메서드 : 특수한 동작 위해 만들어진 메서드**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e75d9153-90b8-41bf-86f0-00e282d2769c/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/aa62b8d9-5d42-456d-ab89-9fad042c2799/Untitled.png)

```jsx
class chano:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

c1 = chano('chano')
print(c1) # chano 
```

- 쉽게 말해서, 밑도끝도없이 print 들어가면 _ _ str _ _ 안에 들어가 있던 리턴값을 찍어 준다는 듯

---

### 데코레이터 ?????????????????

- **다른 함수의 코드 유지한 채로 수정 / 확장하기 위해 사용되는 함수**

```python
def my_decorator(func): # 꾸며줄 func 
    def wrapper():        
        print('함수 실행 전')  # 함수 실행 전에 수행할 작업
        
        result = func()        # 원본 함수 호출

        print('함수 실행 후')        # 함수 실행 후에 수행할 작업

        return result
    return wrapper

@my_decorator # 데코레이터 사용
def my_function():
    print('원본 함수 실행')
my_function()
"""
함수 실행 전
원본 함수 실행
함수 실행 후
"""
```

---

# 참고  for data

## 이터레이터 Iterator :

- 반복 가능한 객체의 요소를 하나씩 반환하는 객체
- 이터레이터 동작 직접 클래스로 정의 ? 
class 내에 __ iter __ (self) 매직 메서드 정의 
__ next __ 포함.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3d9cf454-33dd-49ee-9dcd-66620ce94689/Untitled.png)

### 파이썬 내부적 반복 동작원리 :

- 내부적으로 for 동작할 때에, 반복가능한 객체에 대해 iter() 호출
- iter 함수                : 메서드 __ next __ () 정의하는 이터레이터 객체 돌려줌
- __ next __ 메서드 : 반복 가능한 객체들의 요소들을 한번에 하나씩 접근
- 남은 요소 X          : StopIteraion 예외를 일으켜 for 반복 종료 알림

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/bcbe0992-395b-4a2c-a216-a8771f888e22/Untitled.png)

---

T XXXXXXXXXXXXX 

# 제네레이터

- 제네레이터 함수 통해서 이터레이터 객체 사용
- 이터레이터를 간단하게 만드는 함수
- 반복 가능한 객체 만들지 않으면, 선언과 동시에 메모리 소모
- 사용 이유? 
1. 메모리 효율성 : 
한번에 한개의 값만 생성 : 제너레이터는 값을 하나씩 생성하여 반환 
> 전체 시퀀스를 한번에 메모리에 로드하지 않아도 됨

대용량 데이터 처리 : 대용량 데이터 셋 처리할 때에, 메모리 사용 최소화
ex> 파일의 각 줄을 한번에 하나씩 읽어 처리 가능 

2. 무한 시퀀스 처리 : 
무한 시퀀스 생성 : 무한 루프를 통해서 무한 시퀀스 생성 가능 
> 끝이 없는 데이터 스트림 처리 시 유용 

3. 지연 평가 : 
필요할 때만 값 생성 : 제너레이터는 값이 실제로 필요할 때만 값 생성 
> 불필요한 계산 피하고, 성능 최적화 가능 

연산 지연 : 복잡한 연산 지연하여 수행 > 계산이 필요한 시점에만 연산 이루어짐
- 특징 : 
1. 클래스 기반이 이터레이터 생성 필요성 X  >  __ iter __ () // __ next __ () 메서드 저절로 생성
2. self.index / self .data 같은 인스턴스 변수 사용하는 접근법에 비해 함수 사용 용이, 명료
- 구조 : 
일반적인 함수처럼 작성 
yield 문 사용하여 값 반환

```python
def generate_numbers():
    for i in range(3):
        yield i
# 제네레이터 통해 반복 가능한 객체 생성하고 호출 > 다음 순서 기억

# 호출 전에는 모든 값 메모리에 올리지 않음 ( 지연 평가 )         
for number in generate_numbers():
    print(number)  # 0 1 2
############################################################
def reverse_generator(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]

for char in reverse_generator('abc'):
    print(char) # c b a
############################################################

```

## return, yield 차이

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/37f7ecd2-77c9-4bbf-825d-22013939c7cc/Untitled.png)

### 제네레이터 예시

```python
# 무한 시퀀스

def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
        
gen = infinite_sequence()
print(next(gen))  # 0 > 함수 종료 X 
print(next(gen))  # 1 > 함수 종료 X 
print(next(gen))  # 2 > 함수 종료 X 
############################################################
# 활용 
def fibonacci_generator():
    n1, n2 = 0, 1
    while True:
        yield n1
        n1, n2 = n2, n1 + n2
        
gen = fibonacci_generator()

# 첫 10개의 피보나치 수를 출력
for _ in range(10):
    print(next(gen))  # gen.__next__()와 동일
    
print(next(gen))  # 11번째 피보나치 수
print(next(gen))  # 12번째 피보나치 수

# 처음부터 계산하는게 아니라, 함수와 변수가 유지되고 있으므로 메모리 효율성 ㅁㅊ 
# 과거로 돌아가는건 불가능!!! 필요한게 있으면 타 변수에 저장해둬 

############################################################

```

### 제네레이터 사용한 경우

- 파일에서 한줄씩 읽어와 처리
- 메모리에 전체 파일 로드 X > 한줄씩 처리해 메모리 사용 절약

```python
# 대용량 데이터 처리 
def read_large_file_with_generator(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()
            
# 예제 파일 경로
file_path = 'large_data_file.txt'

# 제너레이터 사용하여 파일 읽기 및 처리
for line in read_large_file_with_generator(file_path):
    print(line)
```

### 제네레이터, 데이터 분석에서의 활용 :

- 추후 데이터 분석 시 큰 데이터 한번에 받아( 메모리에 올려 ) 처리할 수없는 경우 발생
- 데이터를 개발자가 원하는 양만큼만 가져와서 처리하고, 
다시 그 양만큼만 가져와 처리하는 행위 반복하여 데이터 분석 진행

### 제네레이터 표현식 :

- list comprehension 과 비슷 BUT 대괄호 대신 **소괄호 ()**
- 일반적으로, list comprehension 보다 메모리 덜씀
- 간결하지만, 가독성 떨어짐

### 제네레이터 주의사항 :

- 제네레이터 요소 모두 참조한 경우, 재사용 불가능
- 다시 제네레이터 함수 사용해 변수에 재할당 필요  >>  다시 함수 호출하라

---

# 에러, 예외

## 문법 에러 (syntax error)

- 프로그램 구문이 올바르지 않은 경우 발생  /  오타 / 괄호 / 콜론 누락 등 문법적 오류

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/60aca8d5-bed2-4656-a742-7c30236b5669/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c1b70492-c78b-4862-9b29-d7ee797d7352/Untitled.png)

## 예외 ( exception ) -

- 프로그램 실행 중 감지되는 에러
- 내장 예외 : 
1. ZeroDivisionError : 나누기 또는 모듈로 연산의 2번재 인자 0 일 때 발생
2. NameError : 지역 or 전역 이름 못찾을 때에 발생 
3. TypeError : 연산 or 인자 타입 불일치 / 함수 인자 누락 / 인자 초과  
4. ValueError : 연산, 함수에 문제 없지만 부적절한 값을 가진 인자 받았고, 
                          상황이 indexerror 처럼 더 구체적인 예외로 설명되지 않는 경우 발생 
                          ex> int(’1.5’)
5. IndexError : 시퀀스 idx가 범위 벗어날 때에 발생 
6. KeyError : dict 에 해당 key 존재하지 않을 때 
7. ModuleNotFoundError : 모듈 찾을 수 없을 때에 발생 
8. ImportError : import 하려는 이름 못찾을 때 발생 
9. KeyboardInterrupt : 사용자가 ctrl + c / delete 눌렀을 때 발생 ( 무한루프 시 강제 종료 ) 
10. IndentationError : 들여쓰기 관련 오류

---

## 예외 처리

```python
try: 
    x = int(input('숫자를 입력하세요: '))
    y = 10 / x
    
"""
첫번째 예외처리 : 내장 예외명으로 선언 : **0으로 나누면 이거 해** 

Exception-hierarchy ?
zerodivisionerror 걸리면 밑에 except 체크 못한다. 
>> 하위 예외 클래스부터 순차적으로 예외처리 해야 한다.
"""
except ZeroDivisionError: 
    print('0으로 나눌 수 없습니다.')    
except ValueError:        # int 로 형변환 안되는 input이면 이거 띄워줘 
    print('유효한 숫자가 아닙니다.')
    
else:                     # 에러 발생 안하면 해 
    print(f'결과: {y}')
finally:                  # 에러 뜨던 안뜨던 해
    print('프로그램이 종료되었습니다.')
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c3d7715c-350d-4521-acb7-f91078ceab8b/Untitled.png)

---

# 모듈

- 한 파일로 묶인 변수 / 함수 / 클래스의 모음  [ 특정한 기능 하는 코드 작성된 파이썬 파일 ]
- 코드 재사용성 + 유지보수 + 가독성 + 중복 -

- 모듈 가져오는 방법 ? 

- 아래에서 ‘ . ‘ 의 의미 ? :    온점의 왼쪽 객체에서 오른쪽 이름을 찾아라 
                                                 math 객체에서 sqrt 찾아라 

- 서로 다른 모듈, 동일한 함수 import ? :       마지막 import로 대체됨 

- 이를 해결하기 위해 아래와 같은 as (alias) 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a5722d69-19ba-4edb-b3f4-5b81a8c0b669/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/adee6130-ed02-4e37-b671-09992db2b845/Untitled.png)

- 모듈 / 패키지 / 라이브러리 ?
1. 모듈 : 파이썬 파일 하나

2. 패키지 : 연관된 모듈들 하나의 dir에 모아놓은 것 // [ 파이썬 파일 여러개 모여있는거 ]
from package import module
from package.module import func(혹은 변수) 

3. 라이브러리 : 패키지들로 이뤄진 기능 하나

cf> 프레임워크 : 특정한 기능 개발을 위해 모아놓은 틀  >  아예 다른 느낌 

---

# 정규 표현식  222 ~

- 문자열에서 특정 패턴 찾기 위해 사용되는 기법
- 복잡한 문자열 속 특정 규칙으로 된 문자열을 검색 / 치환 / 추출 등 간결하게 수행 가능

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7c32f7ca-adec-4904-bd99-64cf22a1ee56/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3e00699f-3fcf-48b0-9ea7-380723ae2483/Untitled.png)

- re 모듈 : 정규식 일치 연산 제공하는 파이썬 내장 모듈
- 정규표현식 문자열 앞에 ‘r’ 붙이는 이유 ? : 

문자열 raw string 으로 취급해 \ 를 escape 문자로 처리하지 않도록 
하지만? ‘r’ 붙이지 않아도 정상적 동작하지만 / 더 명확하고 안전하다

- re 관련 특수 문자 & 시퀀스 & 메서드  << 밑에서 예시

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f5ba4829-5772-4854-b6c7-62db71d59f89/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f233c369-308a-49c2-8f4e-7129d5583159/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3663e034-b23a-40f6-b5ab-427ec1bcb6a0/Untitled.png)

- 문자열의 시작, 끝 매칭

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/90dca04c-726c-4cbe-9408-d91ddbdacd4a/Untitled.png)

- 반복 패턴 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/51645f16-686e-4d3a-9d3a-9c043a336e9f/Untitled.png)

- 범위 판단 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f612bd93-1c22-4a76-8f01-c07eaa9c6ebe/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9777fe96-a3d0-4c7e-9375-88bf123a0a95/Untitled.png)

- 문자 매칭 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b897a779-fdc2-44ed-a152-a1de170a2309/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/caf3a5cf-84a5-4ee0-b365-a822a9af75d2/Untitled.png)

- 그룹화 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6e5b6791-02d4-44d6-87b5-e99d22ca2c5d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f58c1c48-8c79-4f64-a373-6fe31d88f8c0/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/bd0b4de7-2216-40fe-9835-57dc645cc3ed/Untitled.png)

- 문자열 바꾸기 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6a9990f6-0472-42af-b792-9c00aec07958/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4347adfc-3f1f-4e49-9828-0cf5a20c49b3/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ca6504dd-2222-45a9-b92d-ed572bc035bd/Untitled.png)

- 요약 : GPT ㄱㄱ

---

# 기본 입출력

## 입력

- input() 으로 받아오는 데이터 : 기본적으로 str
- a = list( map(변수 형태, input().split()) )
split 이용해 공백을 기준으로 쪼개고 
map : 앞의 함수를 뒤의 인자에 각각 주는 ?  > map object 로 return 되므로, list 형변환 해야
- 이차원 배열 input 받기 : 
`[list(map(int, input().split())) for _ in range(n)]`
# 붙어있는 list 같은 경우에는 split 빼주기

---

### 아스키 코드

- 65 ~ 90  A ~ Z
- 97 ~ 122 a ~ z
- chr(int)  =  문자
- ord(str)  =  숫자
- .upper() : 문자열 내에 모든걸 대문자로
- .lower()
- .isupper() : 문자열 내에 모든 문자 대문자인지 boolean
- .islower