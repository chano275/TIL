# Algorithm

[https://github.com/RecoRecoNi/Algorithm-Study/blob/main/✨ 효과 만점 코딩테스트 Cheat Sheet!/✏️ 파이썬에서 사용할 수 있는 유용한 구문 모음.md](https://github.com/RecoRecoNi/Algorithm-Study/blob/main/%E2%9C%A8%20%ED%9A%A8%EA%B3%BC%20%EB%A7%8C%EC%A0%90%20%EC%BD%94%EB%94%A9%ED%85%8C%EC%8A%A4%ED%8A%B8%20Cheat%20Sheet!/%E2%9C%8F%EF%B8%8F%20%ED%8C%8C%EC%9D%B4%EC%8D%AC%EC%97%90%EC%84%9C%20%EC%82%AC%EC%9A%A9%ED%95%A0%20%EC%88%98%20%EC%9E%88%EB%8A%94%20%EC%9C%A0%EC%9A%A9%ED%95%9C%20%EA%B5%AC%EB%AC%B8%20%EB%AA%A8%EC%9D%8C.md) 

[알고리즘 Ad(+) 학습 방법](https://www.notion.so/Ad-f26e3fe5cb1e4f2e88dfd4ba6f9957a3?pvs=21) 

---

# 알고리즘 기본

- 알고리즘 : 문제를 해결하기 위한 절차나 방법
                 컴퓨터가 어떤 일을 수행하기 위한 단계적 방법

### 알고리즘 표현 방법 :

- 의사 코드 - 언어 구애 X / 로직 간단, 명확하게
순서도 - 시각적, 직관적 
프로그래밍 언어

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8876a8bc-c429-4de0-a084-2a092e027e6c/Untitled.png)

### 알고리즘 성능 :

- 무엇이 좋은 알고리즘인가 ? :
1. 정확성 : 얼마나 정확하게 동작하는가                       - 모든 경우를 처리해야, 에러 커버해야 
2. 효율성 : 얼마나 최적화 되었는가                              - 실행시간, 메모리
3. 확장성 : 입력 크기에 상관없이 항상 성능 일정한가  - 빅오 
4. 단순성 : 얼마나 단순한가                                          - 유지보수, 디버깅
- 주어진 문제 해결 위한 여러가지 알고리즘 중, 어떤 일고리즘?
- 알고리즘 성능 분석 필요 : 많은 문제에서 성능 분석의 기준으로 ‘알고리즘의 작업량’ 비교
- 1 ~ 100 합 구할때 수식으로 구하면 훨씬 적은 연산으로 계산 가능

## 복잡도

1. 시간 복잡도 : 연산의 작업량, 수행 시간 [ 빅 오 ] 
. 최선의 경우 : 빅 오메가 표현법 사용 / 알고리즘 가장 빠르게 실행될 때의 성능
. 평균적 경우 : 빅 세타 표기법 사용 / 일반적인 알고리즘의 성능
. [ 최악의 경우 ] : 빅 오 표기법 사용 / 알고리즘 가장 느리게 실행될 때의 성능 
2. 공간 복잡도 : 메모리 사용량 - DB 인덱싱 

** trade-off : 시간 - 공간 복잡도는 하나가 좋아질수록 다른쪽이 안좋아질 수 있다.
                     둘 다 최적의 효율을 만족하는 복잡도를 찾아야 한다. 
 
- 복잡도의 점근적 표기 : 
1. 시간/공간 복잡도는 입력 크기에 대한 **함수**로 표기 / 주로 여러개의 항을 가지는 다항식
2. 이를 단순한 함수로 표현하기 위해 점근적 표기 ( Asymptotic Notation ) 사용 = 간단한 함수
3. 입력 크기 n 이 무한대로 커질 때의 복잡도를 표현하기 위해 사용하는 표기법
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/30fdcc55-d205-4d9d-8ab7-08c634ad61ea/Untitled.png)
    

### 빅 오 표기법 :          ***[ n  >  10억 n당  1초 ]***

- 시간 복잡도 함수 중 가장 큰 영향력 주는 n에 대한 항만을 표시 / 계수는 생략
- n개의 데이터 입력받아 저장 > 각 데이터에 1씩 증가 > 각 데이터를 화면에 출력 → 복잡도?
                                                                                              O(n)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4cb82cea-5476-4d75-90ce-ccffe442b0a4/Untitled.png)
    

- 대부분의 효율적 알고리즘 : **n logn 이 목표 ~ n^2 사이의 시간 복잡도**를 만족하도록 노력해야

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fb863384-3864-4cc4-94ca-c6f17a013cf8/Untitled.png)

---

# 재귀 호출

- 반복과 재귀 :  유사한 작업 수행 가능 
1. 반복 : 수행하는 작업 완료될 때까지 반복
2. 재귀 : 주어진 문제의 해 구하기 위해 동일하며 더 작은 문제의 해 이용하는 방법 / 함수로 구현
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/996705d2-d61a-44c8-9af5-dbf86fa77f06/Untitled.png)
    

## 재귀 함수 :

- 함수 내부에서 직/간접적으로 **자기 자신 호출하는 함수  /**  재귀적 정의 이용해 함수 구현
- **기본 부분(basis part, 기저조건) - 종료 조건!!                       (끝)
유도 부분(inductive part)          - 재귀함수 호출하는 부분 (시작)            으로 구성**
- 반복 구조에 비해 간결하고 이해하기 쉬움
- 함수 호출 : 프로그램 메모리 구조에서 스택 사용
 > 재귀 호출 = 반복적인 스택의 사용 의미 
 > 데이터가 쌓인다 > 메모리 및 속도에서 성능저하 발생
- 재귀함수 작성 팁 : 함수 실행 시 **실행을 결정하는 내용을 매개변수로 사용** 
                                                                                아래의 예시에서 리스트 

문제 풀 때에는 일단 다 매개변수로 넘기고 test 하는 방식

### 팩토리얼 재귀 함수 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/978d7d47-9a2f-4ceb-a4d3-e2faf1f8a016/Untitled.png)

```python
def fact(n):
    if n <= 1:    # Basis Rule: n이 1 이하인 경우 1을 반환합니다.
        return 1
    else:        # Inductive Rule: (n-1)로 자기 자신을 호출하는 재귀 케이스
        return n * fact(n - 1)

# 사용 예시
print(fact(5))  # 5*4*3*2*1을 계산하여 120을 출력합니다
```

- basis rule 을 종료조건으로 둔 모습
- 재귀적인 부분(n>1) :        (n-1)!         을 재귀적으로 호출
- Flow Chart :
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d665f92f-4610-4170-a156-1278f720e7d4/Untitled.png)
    

### 피보나치 수열 재귀 함수 :

- 메모이제이션 없는 **시간복잡도 : 2^n** - 값이 늘어날때마다 2개의 fibo 호출하므로
- 피보나치 : 이전에 두 수 합을 다음 항으로 하는 수열
- 피보나치 수열의 i번째 값 계산하는 함수 F에 대해
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7203f0c3-fb8d-49ca-89cd-8b5e699a660a/Untitled.png)
    

```python
def fibonacci(n): # Basic Rule : n이 0일 때, 0을 반환 / n이 1일 때, 1을 반환
	if n == 0:		
		return 0
		
	elif n == 1:	
		return 1
		
	else:			# Inductive Rule : n이 2 이상일 때, F(n-1) + F(n-2) 를 반환합니다
		return fibonacci(n-1) + fibonacci(n-2)
```

- 피보나치 재귀함수 문제점 ? : 엄청난 중복 호출 존재
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/30f661f8-bc2b-4ea3-9efd-5ce553ee8fc3/Untitled.png)
    

## 메모이제이션 :

- 의미 : 프로그램 실행 시 **이전 계산 값을 메모리에 저장** > 매번 계산 X > 전체적 실행속도 빠르게
- memoization 을 글자 그대로 해석 : ‘메모리에 넣기’ 라는 의미 / ‘기억되어야 할 것’

- **동적 계획법 [ DP, 다이나믹 프로그래밍 ] 핵심**- A형에 잘 안나옴 / 코테에 자주 나옴

- fibo1(n) 값 계산하자마자 저장 ( memoize )   >    **실행시간 O(n)** 으로 줄일 수 있다

```python
def fibo1(n):
	global memo
	
  # n이 2 이상이고 memo[n]이 아직 계산되지 않았다면 (0이면 아직 계산되지 않은 것)	
  # memo[n]에 fibo1(n-1)과 fibo1(n-2)의 합을 저장하여 재귀적으로 피보나치 수 계산

	if n >= 2 and memo[n] == 0:			# 재귀 돌리는 부분  
		memo[n] = fibo1(n-1) + fibo1(n-2) 	

	return memo[n]  # 현재 n에 대한 피보나치 수를 반환
	

# 피보나치 수를 계산하기 위한 메모이제이션을 사용하는 배열 초기화
# 메모이제이션은 이미 계산된 결과를 저장하여 중복 계산을 피함
num = 10
memo = [0] * (num + 1)# n+1 크기의 리스트를 생성하고 모든 값을 0으로 초기화
memo[0] = 0  # 피보나치 수열의 첫 번째 수는 0
memo[1] = 1  # 피보나치 수열의 두 번째 수는 1

result = fibo1(num) # n번째 피보나치 수 계산
print(f'피보나치 수열의 {num} 번째 수는 {result}") 
```

## 하노이의 탑 :

- **시간복잡도 : 2^n**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5048a138-71e4-4678-a16a-5921d4637374/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/76defb6e-a723-411b-9768-d281f4cda5a3/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/611cbe74-f9e8-4755-ab3c-39ff43c7677b/Untitled.png)

- 풀이

```python
"""
n : 이동할 디스크의 개수
source: 디스크가 처음에 위치한 기둥
target: 디스크를 이동시킬 목표 기둥
auxiliary: 보조 기둥 
"""

def hanoi(n, source, target, auxiliary):
    # n개의 디스크를 source에서 target으로 이동하는 함수
    if n > 0:
        # 1. n-1개의 디스크를 source에서 auxiliary로 이동 (재귀 호출)
        hanoi(n - 1, source, auxiliary, target)
        
        # 2. 하나의 디스크를 source에서 target으로 이동
        print(f"Move disk {n} from {source} to {target}")
        
        # 3. n-1개의 디스크를 auxiliary에서 target으로 이동 (재귀 호출)
        hanoi(n - 1, auxiliary, target, source)

# 3개의 원반을 옮기는 예시
n = 3  
hanoi(n, 'A', 'C', 'B')  # 'A'에서 'C'로 이동, 'B'는 보조
```

---

# 완전 탐색 : 브루트 포스 [ **순열 / 조합 / 부분집합 ]**

- 어떻게 해서든 무조건 해를 찾을 수 있다 (단, 빠른건 장담 X)
- 모든 경우의 수를 나열 → 확인
- 대부분의 문제 적용 가능 / 상대적으로 빠른 시간에 문제 해결 / 경우의 수 작다면 유용
- **순열(Permutation) / 조합(Combination) / 부분집합(Subset)** 과 같은 조합적 문제들과 연관
- TEST 등에서 우선 완전 검색으로 접근 > 최적화 진행하는게 유용
    - **순열, 조합 모두 동일한 수의 중복은 X ( 1, 1 X )
    ‘중복’ 붙으면 동일한 수의 중복 생각!!**
    
    ## 순열 : 순서 O   [  1 2 ≠ 2 1  ]                    **nPr = n! / (n - r)!**
    
    - 서로 다른 것 중 몇개를 뽑아 **순서대로 나열             <> 조합은 순서 고려 X**
    - 서로 다른 n개중 r개 택하는 순열 : **nPr** = n * n-1 * n-2 * … * (n-r+1)   /   npn = n!
    - 다수의 알고리즘 문제 : 순서화된 요소들의 집합에서 방법 찾는것과 관련
    - **n개의 요소에서 n! 개의 순열들 존재 : N ≤ 12 일때만 순열 사용** N > 12 이면 시간 복잡도 답없
    
    ```python
    for i in range(1, 4):         # 1 ~ 3
    
    	for j in range(1, 4):       # 1 ~ 3
    	
    		if j != i:                # **동일한 수 X**
    	
    			for k in range(1, 4):   # 1 ~ 3 
    	
    				if k != i and k != j: # **동일한 수 X**
    	
    					print(i, j, k) 
    ```
    
    ```python
    # 재귀를 통한 순열 생성        < **뒤에 코드가 더 낫다**
    # selected : 선택된 값 목록 / remain : 선택되지 않고 남은 값 목록 
    
    def perm(selected, remain):
    
    	if not remain:				print(selected) # remain 비어있으면 seleted 출력 
    	
    	else:
    		for i in range(len(remain)): 
    			select_i = result[i] # 요소 하나 선택 > 1 / 2 / 3 순서대로
    			remain_list = remain[:i] + remain[i+1:] 			
    			perm(selected + [select_i], remain_list) 
    			
    			"""
    			**내가 선택한 수 제외하고 리스트 다시 생성하는 법** > 방법을 기억하기 
    			
    			list[a:b]                 > a 포함, b 불포함 
    			
    			remain[:i] + remain[i+1:] > i 불포함, i+1 포함
    			0 ~ i-1      i+1 ~ end-1			
    			"""
    									
    			
    perm([], [1,2,3])
    ```
    
    ---
    
    ## 조합 : 순서 X   [  1 2 == 2 1  ]           **nCr = 순열 / r!**
    
    - 서로 다른 n개의 원소 중 r개를 순서 없이 골라낸 것    <> 순열 : 순서 O
    
     
    
    - 조합의 수식 :   nPr 의 분모에 r! 가 붙은 수식 
    r! : r개의 원소를 나열하는 방법의 수  >  그걸 나눔으로써 순서 사라진다 > 조합의 수가 구해진다
    
    - 재귀적 표현 이해 :
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/969a0fc7-6f23-4613-8b16-7183db53cdd5/Untitled.png)
        
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8b52789a-5cd7-4278-bdaf-851135fa2619/Untitled.png)
    
    - 반복문/재귀 통한 조합 생성
    
    ```python
    # 바로 우측 아이디어를 그대로 들고옴 
    # ( 밑줄친 애들만 살리기 ) 
    
    for i in range(1, 5):
        for j in range(i+1, 5):
            for k in range(j+1, 5):
                print(i, j, k)
    ```
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d11925fb-3fa0-4e00-bf18-a4e66221771f/Untitled.png)
    
    ### 본 코드가 중요!  -  arr 배열에서 n개의 요소를 선택하여 모든 조합 생성하는 함수
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6fcf3a56-b82f-4f35-a04d-dd700113f28d/Untitled.png)
    
    - 본 그림 설명 : 
    맨 아래에서 1 선택 시에 나오는 경우의 수    우측에서    1 + [ 23 / 24 / 34 ]  가 
    바로 위의 모든 경우의 수 [23 / 24 / 34] 와 동일하고, 
    
    해당 줄의 왼쪽에서 2를 선택 시의 2 + [3 / 4] 가  
    바로 위의 [3 / 4] 와 동일하다 
    
    바로 위는 3, 4 중 1개를 고르는 것이므로 알고리즘이 필요하지 않다.
    
    ```python
    # **이코드에 집중!!!** 
    
    def comb(arr, n):
        result = []  # 결과를 저장할 빈 리스트 초기화
    
    		# 탈출조건 : 
    		# **선택할 요소의 수가 1**인 경우 : 위의 그림에서 {3, 4} 중 하나 고르는 경우?
    		# 더이상 조합할 요소 필요 없이, **각 요소가 하나의 조합 > 각 요소를 리스트로 감싸서 반환**
        if n == 1:          
    		    **return [[i] for i in arr]**
    
        # 배열의 각 요소에 대해 반복
        for i in range(len(arr)):
            elem = arr[i]  # **현재 요소를 선택 ( 위 이미지에서 밑에서부터 1, 2 )** 
            
            # 현재요소(1) 이후 나머지 요소들 ( **위 이미지에서 2, 3, 4** ) 로 n-1(2)개 조합 재귀적 생성
            for rest in comb(arr[i + 1:], n - 1):
                result.append([elem] + rest)  
                # 현재 선택한 요소(1) + 재귀 호출을 통해 얻은 조합(23, 24, 34)을 합침
                # 23, 24, 34 중 23, 24 같은 경우에는 elem을 2로 하여 3, 4중 1개를 고르는 ... 
    
        return result
        
    print(comb([1, 2, 3, 4], 3))  # [1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4] 출력
    
    """
    CF > 
    for rest in comb(arr[:i] + arr[i+1:], n - 1):  # 순열 - 나를 제외하고(중복 제거) 
    
    for rest in comb(arr, n - 1):  # 중복순열 - 나를 제외하지 말고(중복 O) 
    
    for rest in comb(arr[i:], n - 1):  # 중복조합 - 나를 포함해서 ... ( + 1 빠짐 ) 
    
    """
    ```
    
    ---
    
    ## 중복 순열 :   중복 ( 1,1 ) 가능한 순열 / 조합 !
    
    # 여기서의 중복이라는 뜻 ? 값의 동일 X , 동일 IDX 값
    
    - 순서를 고려하여 여러 번 선택할 수 있게 나열하는 모든 가능한 방법
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5987474f-b560-4e34-9ebd-135213f8abf6/Untitled.png)
    
    ## 중복 조합 :
    
    - 순서 고려하지 않고 여러 번 선택할 수 있게 나열하는 모든 가능한 방법
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a8f97777-685d-4614-bd62-824a2c719516/Untitled.png)
    
    ---
    
    ## Itertools 활용 :
    
    - 앞에 나온 4가지 모듈 사용해서 구현하는 방법 : 쓸수 있을수도 없을수도
    
    ```python
    import itertools
    arr = [1, 2, 3]
    
    print(tuple(itertools.permutations(arr)))  # 순열# ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
    
    print(tuple(itertools.combinations(arr, 2)))  # 조합# ((1, 2), (1, 3), (2, 3))
    
    print(tuple(itertools.product(arr, repeat=2)))  # 중복순열# ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
    
    print(tuple(itertools.combinations_with_replacement(arr, 2)))  # 중복조합# ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
    ```
    
    ---
    
    ## 부분집합 :
    
    - 집합에 포함된 원소들 선택하는 것
    - **N개의 원소 포함한 집합에서 공집합 포함한 부분집합의 개수 ? 2^n 개**
    - 각 원소를 부분집합에 **포함하거나 포함하지 않는 2가지 경우를 모든 원소에 적용한 것과 같다**
    
    - 많은 중요 알고리즘들이 원소들의 그룹에서 최적의 부분 집합 찾는데 사용됨
    ex> **Knapsack ( 배낭 짐싸기 ) - DP, 그리디로도 가능**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f31bb75e-1dc7-4d71-a894-4968c9ebc19d/Untitled.png)
    
    ### 부분집합 - 반복문 & 재귀 :
    
    - {1, 2, 3} 집합의 모든 부분집합 생성 :
    
    ```python
    selected = [0] * 3 # 원소 선택 했냐 안했냐 확인하는 리스트
    
    for i in range(2): # 0 / 1
    	selected[0] = i
    	for j in range(2): # 0 / 1 
    		selected[1] = j
    		for m in range(2): # 0 / 1 
    			selected[2] = m
    			
    			subset = []
    			
    			for n in range(3):
    				if selected[n] == 1:# 선택을 했으면
    					subset.append(n+1) # 해당 원소 idx + 1 해서 subset 에 넣기
    			print(subset)
    ```
    
    ```python
    # 주어진 input_list에서 모든 부분 집합을 생성하는 함수
    # 넘어렵네 ;;; 16:00 
    # https://edu.ssafy.com/edu/board/free/detail.do?brdItmSeq=85135&listMenu=&lctCd=0207&lctrRepId=RE20240715130206&lctrBrdSeq=7734&_csrf=e3e8bd54-77ec-4e7c-b2d7-c8e43367de5e
    
    def create_subset(depth, included): # depth : 볼 idx 
    
        if depth == len(input_list):  # 재귀 호출 깊이가 input_list의 길이와 같아지면
            cnt_subset = [input_list[i] for i in range(len(input_list)) if included[i]]  # 포함된 요소들로 부분 집합 생성
            subsets.append(cnt_subset)  # 생성된 부분 집합을 subsets 리스트에 추가
            return
    
        # 현재 요소를 포함하지 않는 경우
        included[depth] = False
        create_subset(depth + 1, included)  # 다음 깊이로 재귀 호출
    
        # 현재 요소를 포함하는 경우
        included[depth] = True
        create_subset(depth + 1, included)  # 다음 깊이로 재귀 호출
    
    input_list = [1, 2, 3]  # 부분 집합을 생성할 입력 리스트
    subsets = []  # 모든 부분 집합을 저장할 리스트
    init_included = [False] * len(input_list)  # 각 요소의 포함 여부를 저장할 리스트 초기화
    create_subset(0, init_included)  # 부분 집합 생성 함수 호출
    print(subsets)  # 생성된 모든 부분 집합 출력
    
    ```
    
    ### 부분집합 - 바이너리 카운팅 :
    
    - 부분집합을 생성하기 위한 가장 자연스러운 방법
    - 원소 수에 해당하는 N개의 비트열을 이용 > N번 비트값이 1이면, N번 원소가 포함되었음을 의미
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/28f692b7-ff29-42a8-92f5-8330984e0dac/Untitled.png)
        
    
    - cf> 비트연산
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/397ab6e3-73a5-40a5-8d93-2d59d81e1aae/Untitled.png)
    
                                              a<<b : a라는 값을 b비트만큼 왼쪽으로 shift 
    
    ** 2의 보수 : 
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/25f3d636-3489-4fdd-99b5-05cce7205dc8/Untitled.png)
    
    ### 바이너리 카운팅을 통한 부분집합 생성 :
    
    ```python
    # 바이너리 카운팅을 통한 부분집합 생성
    # arr 배열에서 모든 부분 집합을 생성하는 코드 
    
    arr = [1, 2, 3]
    n = len(arr)  # 배열의 길이
    subset_cnt = 2 ** n  # **생성 가능한 부분 집합의 총 개수**
    
    subsets = []
    for i in range(subset_cnt):  # 0 ~ 7
    
        subset = []
        for j in range(n):  # **비트 자리 다 확인 : 자리수만큼** 1 밀며 i의 j번째 비트가 1인지 확인
            if i & (1 << j):
                subset.append(arr[j])
    
        subsets.append(subset)
    
    print(subsets)
    
    ```
    
    ---
    
    ## 예시)
    
    ### 베이비진 게임
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a2e9fceb-6044-4bde-959e-4a2e1371d7be/Untitled.png)
    
    - CASE 1) 모든 숫자 나열 ( 순열 ) : 각각 확인 
    앞뒤 33 잘라서 RUN / TRIPLET 확인
    
    ```python
    # 수정하고 올린건지 Q / 카운팅 코드 틀린거 맞는지 Q 
    
    from itertools import permutations
    
    # 카드가 순서대로 놓였는 지 확인 
    def is_run(cards):
        return cards[0] + 1 == cards[1] and cards[1] + 1 == cards[2]
    
    # 카드가 모두 같은 지 확인 
    def is_triplet(cards):
        return cards[0] == cards[1] == cards[2]
    
    def is_babygin_permutations(cards):
        if len(cards) != 6:        return False
        
        # 모든 가능한 6장 카드의 순열을 생성
        for perm in permutations(cards):
            first_group = perm[:3]
            second_group = perm[3:]
            
            # 두 그룹이 모두 run이거나 triplet인 경우 True 반환
            if (is_run(first_group) or is_triplet(first_group)) and
            (is_run(second_group) or is_triplet(second_group)):
                return True
        
        return False
    
    numbers_1 = [6, 6, 6, 7, 7, 7]  
    numbers_2 = [2, 3, 5, 7, 7, 7] 
    
    print(f"{numbers_1} 은 baby-gin이 {is_babygin_permutations(numbers_1)}")
    print(f"{numbers_2} 은 baby-gin이 {is_babygin_permutations(numbers_2)}")
    ```
    
    ```python
    코드 맛 간거 삭제함 
    ```
    
    ---
    
    ### 여행 계획 문제 :
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c2613487-b605-4ae1-866a-f9cba20a60cd/Untitled.png)
    
    - 6개 중 3개 도시 선택시 숙박비 지원 > 가장 이득?  >  **조합 : 6C3**
    - 여행경비 넘지 않으며 **최대 만족도 선택** ? 70만원 / 경로 고려 X 
    **부분 집합 : 6C1 + 6C2 + 6C3 + 6C4 + 6C5 + 6C6   >    Knapscak**
    

---

# 자료구조 - 스택 & 큐

## 스택

- **선형 구조** : 데이터 요소들 사이에 순서가 존재
                   1대 1 관계 : 각 요소가 앞 / 뒤 연결 
                    **리스트 / 스택 / 큐 / 연결리스트**

- cf> **비선형** ? : 데이터 순서 X 
                       1:N / N:M 관계 
                       **트리 / 그래프**

### 스택의 구조 & 작동원리

- 파이썬은 top pointer 사용 X

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3e7b2d81-430d-4419-8ab0-cf79fdbc2895/Untitled.png)

- 스택 구현

```python
class Stack:
    def __init__(self):    # 스택을 초기화하는 메서드
        self.stack = []  # **스택을 빈 리스트로 초기화**

    def push(self, item):    # 스택에 아이템을 추가하는 메서드
        self.stack.append(item)  # 스택의 맨 끝에 아이템 추가

    def pop(self):    # 스택에서 아이템을 제거하고 반환하는 메서드
        if not self**.is_empty()**: return self.stack.pop() # 스택이 비어있 X > 맨 끝 제거+반환
        else:            print("스택이 비었습니다.")  # 스택이 비어 있는 경우 경고 메시지 출력

    def peek(self):    # 스택의 맨 위 아이템을 반환하는 메서드
        if not self.is_empty(): return self.stack[-1]  # 스택이 비어있 X > **맨 끝 아이템 반환 [-1]**
        else:            print("스택이 비었습니다.")  # 스택이 비어 있는 경우 경고 메시지 출력

    def is_empty(self):    # **스택이 비어 있는지 확인하는 메서드 len == 0**
        return len(self.stack) == 0  # 스택의 길이가 0인지 확인
    
stack = Stack()

stack.push(1)
stack.push(2)
stack.push(3)

print(stack.pop())
print(stack.pop())
print(stack.peek())
print(stack.is_empty())
print(stack.pop())
print(stack.pop())
```

### 스택 응용

```python
# 괄호 검사 

def check_match(expression):
	stack = []
	matching_dict = {')':'(', '}' : '{', ']' : '['} 	# 괄호 짝으로 dict 생성
	
	for ch in expression: 	# 입력받은 문자열 순회 
		if ch in matching_dict.values(): stack.append() # 열린 괄호 만날 때 스택에 넣어  
		
			
		"""
		닫힌 괄호 만날 때 : 스택 비어있을 때에 pop하면 error 나므로, 
		스택이 비어있거나 (if not stack) 스택의 맨 위에있는 원소 (pop 할 원소) 가 짝이 안맞는다면
		"""
		elif ch in matchind_dict.keys():
			if not stack or stack[-1] != matching_dict[ch] : 
				return False

		# 닫힌 괄호인데 짝도 맞아? pop해
		else:  stack.pop()

	return not stack 
	# 스택에 값 남아있으면 stack << True라는 뜻 
	# not 을 통해 스택이 비어있으면 True / 값이 남아있으면 False 반환
```

```python
/* Function Call : 가장 마지막에 호출된 함수가 가장 먼저 실행 완료하고 복귀 
함수 호출 발생하면 호출한 함수 수행에 필요한 지역,매개변수 / 수행 후 복귀주소 등의 정보
스택 프레임에 저장하여 시스템 스택에 삽입 

함수 실행 끝나면 시스템 스택의 top ( 스택 프레임) 삭제(pop) 
> 프레임에 저장되어 있던 복귀주소 확인하고 복귀 
> 함수 호출, 복귀에 따라 이 과정을 반복 
> 전체 프로그램 수행 종료되면 시스템 스택은 공백 스택 */
```

```python
"""
계산기 : 
1. 중위 표기법 수식을 스택 이용해 후위 표기법으로 
2. 후위 표기법 수식 스택을 이용해 계산 
( 왼 > 오 시 연산자 만나면 앞의 피연산자 2개꺼내 계산)

변환법 : 밑의 img 참고
1. 수식의 각 연산자에 대해 우선순위 따라 괄호를 사용해 다시 표현
2. 각 연산자를 그에 대응하는 오른쪽 괄호의 뒤로 이동시킨다
3. 괄호 제거

계산 : 밑의 img 참고
1. 피연산자 만나면 push 
2. 연산자 만나면 필요한 만큼의 피연산자 ( 2개 ) pop해서 연산 > 결과를 push
3. 수식 끝나면 pop해서 출력 
*** 먼저 빼낸 피연산자가 뒤로 간다는거 생각
"""
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/eba8b096-e13e-4bae-911d-27ea8041c8b2/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1453edb1-ac6a-40a5-8982-9db3eb1de5b9/Untitled.png)

```python
# 수업에서 리뷰 안함 < ????????????

# 중위 표기식을 후위 표기식으로 변환하는 함수
def infix_to_postfix(expression):
    # 연산자의 우선순위를 정의
    op_dict = {'+': 1, '-': 1, '*': 2, '/': 2, '(': 0}
    stack = []  # 연산자를 저장할 스택
    postfix = []  # 후위 표기식을 저장할 리스트

    for ch in expression:
        if ch.isnumeric():  # 숫자인 경우
            postfix.append(ch)  # 후위 표기식에 추가
        elif ch == '(':  # 여는 괄호인 경우
            stack.append(ch)  # 스택에 추가
        elif ch == ')':  # 닫는 괄호인 경우
            top_token = stack.pop()  # 스택에서 연산자를 꺼냄
            while top_token != '(':  # 여는 괄호를 만날 때까지
                postfix.append(top_token)  # 후위 표기식에 추가
                top_token = stack.pop()
        else:  # 연산자인 경우
            # 스택에 들어 있는 연산자가 지금 검사하는 연산자보다 우선순위가 더 높은 경우
            # 높은 친구들을 모두 거내서 후위 표기식에 추가하고, 검사하는 연산자를 스택에 추가
            while stack and op_dict[stack[-1]] >= op_dict[ch]:
                postfix.append(stack.pop())
            stack.append(ch)

    while stack:  # 스택에 남아 있는 모든 연산자를 후위 표기식에 추가
        postfix.append(stack.pop())
    
    return ' '.join(postfix)  # 리스트를 문자열로 변환하여 반환

# 후위 표기식 계산 함수 
def run_calculator(expr):
    stack = []  # 피연산자를 저장할 스택
    tokens = expr.split()  # 후위 표기식을 공백으로 구분하여 토큰으로 분리

    for token in tokens:
        if token.isnumeric():  # 숫자인 경우
            stack.append(int(token))  # 스택에 추가
        else:  # 연산자인 경우
            op2 = stack.pop()  # 스택에서 두 번째 피연산자를 꺼냄
            op1 = stack.pop()  # 스택에서 첫 번째 피연산자를 꺼냄
            if token == '+':
                result = op1 + op2
            elif token == '-':
                result = op1 - op2
            elif token == '*':
                result = op1 * op2
            elif token == '/':
                result = op1 / op2
            stack.append(result)  # 계산 결과를 스택에 추가

    return stack.pop()  # 최종 결과를 반환

# 예시
infix_expression = "3+(2*5)-8/4"
postfix_expression = infix_to_postfix(infix_expression)
print(f"후위 표기식: {postfix_expression}")

result = run_calculator(postfix_expression)
print(result)
```

---

## 큐 : 앞에서 빼는 문제

- **뒤에서는 삽입만 / 앞에서는 삭제만**
- 파이썬에서는 front, rear 생각 X

### 큐의 구조 & 작동원리

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e1f8aa35-3e48-4ccb-8dac-836dcf0628a5/Untitled.png)

- 연산 과정 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e2c54aae-eceb-406b-bc7a-6c97e84f7899/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2c15e6d3-7d99-476f-b036-2d5a98190833/Untitled.png)

- 전형적 단점 : 작업이 끝났을 때에 앞의 공간을 버리게 됨
                      >>>> 이 단점 메꾸고자 원형큐, 이중/연결 리스트 사용

### 큐 응용

- 버퍼 : 
1. 데이터 한 곳에서 다른 곳으로 전송하는동안 일시적으로 해당 데이터 보관하는 메모리 영역
2. 입출력 / 네트워크 관련 기능에서 이용
3. 순서대로 입,출력, 전달되어야 하므로 FIFO 인 큐 사용됨
- 마이쮸 :
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a777eee2-e2ff-47e6-8050-d1f28ee6a753/Untitled.png)
    

- 문제 RR

```python
from collections import deque 
# **데크 : 앞에서 데이터 꺼내도 시간 복잡도 O(1)**  VS        list pop >> O(n) 

total_candy = 20  # 총 마이쮸 개수
queue = deque()  # 사람들을 저장할 큐
queue.append((1, 1))  # 첫 번째 사람과 받을 마이쮸 개수를 큐에 추가

last_person = None  # 마지막으로 마이쮸를 받은 사람을 저장할 변수

person_cnt = 1
while total_candy > 0:  # 마이쮸가 남아 있는 동안 반복
    person, count = queue.popleft()  # 큐에서 사람과 받을 마이쮸 개수를 꺼냄
    
    if total_candy - count <= 0:  # 남은 마이쮸가 현재 사람이 받을 마이쮸 개수보다 적거나 같은 경우
        last_person = person  # 마지막으로 마이쮸를 받은 사람으로 현재 사람을 설정
        break  

    total_candy -= count  # 현재 사람이 받을 마이쮸 개수를 총 마이쮸 개수에서 뺌
    person_cnt += 1
    queue.append((person, count + 1))  # 현재 사람은 다음 차례에 받을 마이쮸 개수를 1 증가시켜 큐에 다시 추가
    queue.append((person_cnt, 1))  # 다음 사람을 큐에 추가, 받을 마이쮸 개수는 1로 설정

print(f"마지막 마이쮸는 {last_person}번")  # 마지막으로 마이쮸를 받은 사람을 출력

```

- 원형큐 ~ 이중 리스트 : 파이썬에서는 살짝 중요도 down

## 원형 큐

- 잘못된 포화상태 인식 : 
선형큐 이용해 삽입&삭제 계속할 경우, 리스트의 앞부분에 활용공간 있어도 
rear = n-1 (포화상태) 로 인식하여 삽입 불가하게 됨.
- 해결? : 
1. 매 연산이 일어날 때마다 저장된 원소들을 배열 앞부분으로 > 시간복잡도 너무 손해 

2. 1차원 배열 사용하되, 논리적으로 배열의 처음과 끝을 연결되어 있다고 생각해 
   원형 형태의 큐를 이룬다고 가정하고 사용 
***** idx에 리스트 크기만큼 나누기 *****

### 원형 큐의 구조 & 작동원리

- 연산과정 ㄹㄹㄹ

```python
class CircularQueue: # ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ
    # 원형 큐를 초기화하는 메서드
    def __init__(self, size):
        self.size = size  # 큐의 크기
        self.queue = [None] * size  # 큐를 지정된 크기의 None 리스트로 초기화
        self.front = -1  # 큐의 앞쪽 인덱스 초기화
        self.rear = -1  # 큐의 뒤쪽 인덱스 초기화

    # 큐에 아이템을 추가하는 메서드
    def enqueue(self, item):
        if self.is_full():  # 큐가 가득 찬 경우
            print("큐가 가득 찼습니다.")  # 경고 메시지 출력
        else:
            if self.front == -1:  # 큐가 비어 있는 경우
                self.front = 0  # front를 0으로 설정
            self.rear = (self.rear + 1) % self.size  # rear 인덱스를 순환하여 증가
            self.queue[self.rear] = item  # rear 위치에 아이템 추가

    # 큐에서 아이템을 제거하고 반환하는 메서드
    def dequeue(self):
        if self.front == -1:  # 큐가 비어 있는 경우
            print("큐가 비었습니다.")  # 경고 메시지 출력
            return None
        else:
            dequeued_item = self.queue[self.front]  # front 위치의 아이템 제거
            if self.front == self.rear:  # 큐에 하나의 아이템만 있는 경우
                self.front = -1  # 큐를 비움
                self.rear = -1
            else:
                self.front = (self.front + 1) % self.size  # front 인덱스를 순환하여 증가
            return dequeued_item

    # 큐의 맨 앞 아이템을 반환하는 메서드
    def peek(self):
        if self.front == -1:  # 큐가 비어 있는 경우
            print("큐가 비었습니다.")  # 경고 메시지 출력
            return None
        else:
            return self.queue[self.front]  # front 위치의 아이템 반환

    # **큐가 비어 있는지 확인하는 메서드**
    def is_empty(self):
        return self.front == -1  # 큐가 비어 있는지 확인

    # 큐가 가득 찼는지 확인하는 메서드
    def is_full(self):
        return (self.rear + 1) % self.size == self.front  
        # **rear 인덱스가 front 인덱스의 바로 앞**에 있는지 확인

    # 큐의 현재 상태를 문자열로 반환하는 메서드 (디버깅을 위해 추가)
    def __str__(self):
        if self.is_empty():
            return "원형 큐가 비어 있습니다"
        elif self.rear >= self.front:
            return f"원형 큐: {self.queue[self.front:self.rear + 1]}"
        else:
            return f"원형 큐: {self.queue[self.front:] + self.queue[:self.rear + 1]}"

# CircularQueue 클래스 테스트
cq = CircularQueue(5)
cq.enqueue(1)
cq.enqueue(2)
cq.enqueue(3)
cq.enqueue(4)
print(cq)  # 원형 큐: [1, 2, 3, 4]

dequeued_item = cq.dequeue()
print(dequeued_item)  # 1
print(cq)  # 원형 큐: [2, 3, 4]

```

- Front = -1 : 비어있음
rear → front : 꽉 찼다 
front == root : 원소가 하나 남음

…

---

## 연결 리스트

- 리스트의 문제점? : 
자료의 삽입/삭제 연산 중 연속적인 메모리 배열을 위해 원소들을 이동시키는 작업 필요
원소 개수 많고, 삽입/삭제 연산 빈번하게 일어날수록 작업에 소요되는 시간 증대

### 연결 리스트 개요

- 자료의 논리적인 순서 / 메모리 상의 물리적 순서 일치 X / 개별적으로 위치하는 
각 원소 연결해 하나의 전체적인 자료구조를 이룬다. 

링크를 통해 원소에 접근 > 리스트에서처럼 물리적 순서를 맞추기 위한 작업 필요 X 

자료구조 크기 동적으로 조정 가능 > 메모리 효율적인 사용 가능

- 기본 구조 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8169c75e-05d6-4964-9661-f8187bcbbebf/Untitled.png)

### 단순 연결 리스트

- 노드가 하나의 링크 필드에 의해 다음 노드와 연결되는 구조
헤드가 가장 앞의 노드 가리킴 / 링크 필드가 연속적으로 다음 노드 가리킴
링크 필드가 Null인 노드 : Linked List의 가장 마지막 노드
- 장점 : 삽입, 삭제 효율적 O(1) < 맨 앞/뒤의 삽입, 삭제만 / 필요한 만큼만 메모리 사용
단점 : 특정 요소 접근하려면 순차적 탐색 O(N) / 역방향 탐색 불가
- 삽입 연산 : 
1. 첫번째 노드 삽입 : 
새로운 노드 생성 후 데이터 필드에 데이터 저장 / 링크 필드에 Null 저장 
헤드에 본 노드의 주소값 부여 

2. 첫번째 노드로 새 노드 삽입 : 
새로운 노드 생성 후 데이터 필드에 데이터 저장 / 링크 필드에 **헤드의 주소값** 넣기
헤드에 본 노드의 주소값 부여 

3. 마지막 노드로 삽입 : 
새로운 노드 생성 후 데이터 필드에 데이터 저장 / 링크 필드에 Null 저장 
이전 노드의 주소값에 Null이 들어 있을텐데, 해당 공간에 본 노드의 주소값 넣어주기 

4. 가운데 삽입 
새로운 노드 생성 후 데이터 필드에 데이터 저장 / 링크 필드에 **다음 노드 주소값** 저장 
이전 노드의 링크값을 본 노드의 주소값으로 넣어주기
- 삭제 연산 :  
1. 중앙 / 맨 앞 :
노드 하나 삭제 후 이전 노드 or HEAD의 링크 필드에 다음 노드의 주소값 넣어줌
2. 마지막 : 이전 노드의 링크 필드를 Null로

```python
class Node: # ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ
    def __init__(self, data):
        self.data = data  # 노드의 데이터
        self.next = None  # 다음 노드를 가리키는 포인터

class SinglyLinkedList:
    def __init__(self):
        self.head = None  # 링크드 리스트의 헤드 초기화

    # 특정 위치에 노드를 삽입하는 메서드
    def insert(self, data, position):
        new_node = Node(data)  # 삽입할 새로운 노드 생성
        if position == 0:  # 위치가 0인 경우
            new_node.next = self.head  # 새로운 노드의 다음을 헤드로 설정
            self.head = new_node  # 헤드를 새로운 노드로 변경
        else:
            current = self.head
            for _ in range(position - 1):  # 삽입 위치 이전까지 이동
                if current is None:  # 범위를 벗어난 경우
                    print("범위를 벗어난 삽입입니다.")
                    return
                current = current.next
            new_node.next = current.next  # 새로운 노드의 다음을 현재 노드의 다음으로 설정
            current.next = new_node  # 현재 노드의 다음을 새로운 노드로 설정

    # 리스트의 끝에 노드를 추가하는 메서드
    def append(self, data):
        new_node = Node(data)  # 추가할 새로운 노드 생성
        if self.is_empty():  # 리스트가 비어있는 경우
            self.head = new_node  # 헤드를 새로운 노드로 설정
        else:
            current = self.head
            while current.next:  # 마지막 노드까지 이동
                current = current.next
            current.next = new_node  # 마지막 노드의 다음을 새로운 노드로 설정

    # 리스트가 비어있는지 확인하는 메서드
    def is_empty(self):
        return self.head is None  # 헤드가 None인지 확인

    # 특정 위치의 노드를 삭제하는 메서드
    def delete(self, position):
        if self.is_empty():  # 리스트가 비어있는 경우
            print("싱글 링크드 리스트가 비었습니다.")
            return
        
        if position == 0:  # 첫 번째 노드를 삭제하는 경우
            deleted_data = self.head.data  # 삭제할 데이터 저장
            self.head = self.head.next  # 헤드를 다음 노드로 변경
        else:
            current = self.head
            for _ in range(position - 1):  # 삭제할 노드의 이전까지 이동
                if current is None or current.next is None:  # 범위를 벗어난 경우
                    print("범위를 벗어났습니다.")
                    return
                current = current.next
            deleted_node = current.next  # 삭제할 노드 저장
            deleted_data = deleted_node.data  # 삭제할 데이터 저장
            current.next = current.next.next  # 이전 노드의 다음을 삭제할 노드의 다음으로 변경
        return deleted_data  # 삭제한 데이터 반환

    # 특정 데이터를 가진 노드의 위치를 찾는 메서드
    def search(self, data):
        current = self.head
        position = 0
        while current:  # 리스트를 순회하며 데이터 찾기
            if current.data == data:
                return position  # 데이터가 있는 위치 반환
            current = current.next
            position += 1
        return -1  # 데이터를 찾지 못한 경우 -1 반환

    # 리스트를 문자열로 변환하는 메서드
    def __str__(self):
        result = []
        current = self.head
        while current:  # 리스트를 순회하며 데이터를 결과 리스트에 추가
            result.append(current.data)
            current = current.next
        return str(result)  # 결과 리스트를 문자열로 변환하여 반환

sll = SinglyLinkedList()
sll.append(1)
sll.append(2)
sll.append(3)
print(sll)  # [1, 2, 3]

deleted_item = sll.delete(1)
print(f"Deleted item: {deleted_item}")  # 2
print(sll)  # [1, 2, 3]
```

### 이중 연결 리스트

- 단순 연결 리스트의 역방향 탐색이 힘들다는 단점을 개선하기 위해 등장
- 이전 리스트를 가리키는 링크 필드를 추가

```python
class Node: # ㄱㄱㄱ 
    def __init__(self, data):
        self.data = data  # 노드의 데이터
        self.prev = None  # 이전 노드를 가리키는 포인터
        self.next = None  # 다음 노드를 가리키는 포인터

class DoublyLinkedList:
    def __init__(self):
        self.head = None  # 리스트의 첫 번째 노드를 가리키는 포인터
        self.tail = None  # 리스트의 마지막 노드를 가리키는 포인터

    def append(self, data):
        new_node = Node(data)  # 새로운 노드 생성
        if self.is_empty():
            self.head = new_node  # 리스트가 비어있으면 head와 tail 모두 새로운 노드를 가리킴
            self.tail = new_node
        else:
            self.tail.next = new_node  # 현재 tail의 next가 새로운 노드를 가리키도록 설정
            new_node.prev = self.tail  # 새로운 노드의 prev가 현재 tail을 가리키도록 설정
            self.tail = new_node  # tail이 새로운 노드를 가리키도록 업데이트

    def insert(self, data, position):
        new_node = Node(data)  # 새로운 노드 생성
        if position == 0:
            if self.is_empty():
                self.head = new_node  # 리스트가 비어있으면 head와 tail 모두 새로운 노드를 가리킴
                self.tail = new_node
            else:
                new_node.next = self.head  # 새로운 노드의 next가 현재 head를 가리키도록 설정
                self.head.prev = new_node  # 현재 head의 prev가 새로운 노드를 가리키도록 설정
                self.head = new_node  # head가 새로운 노드를 가리키도록 업데이트
        else:
            current = self.head
            for _ in range(position - 1):
                if current is None:
                    print("범위를 벗어났습니다.")
                    return
                current = current.next
            new_node.next = current.next  # 새로운 노드의 next가 current의 next를 가리키도록 설정
            new_node.prev = current  # 새로운 노드의 prev가 current를 가리키도록 설정
            if current.next:
                current.next.prev = new_node  # current의 next의 prev가 새로운 노드를 가리키도록 설정
            current.next = new_node  # current의 next가 새로운 노드를 가리키도록 설정
            if new_node.next is None:
                self.tail = new_node  # 새로운 노드가 마지막 노드라면 tail 업데이트

    def is_empty(self):
        return self.head is None  # 리스트가 비어 있는지 확인

    def delete(self, position):
        if self.is_empty():
            print("더블 링크드 리스트가 비었습니다.")
            return
        
        if position == 0:
            deleted_data = self.head.data
            self.head = self.head.next  # head를 다음 노드로 업데이트
            if self.head:
                self.head.prev = None  # 새로운 head의 prev를 None으로 설정
            else:
                self.tail = None  # 리스트가 비어있으면 tail도 None으로 설정
        else:
            current = self.head
            for _ in range(position - 1):
                if current is None or current.next is None:
                    print("범위를 벗어났습니다.")
                    return
                current = current.next
            deleted_node = current.next
            deleted_data = deleted_node.data
            current.next = deleted_node.next  # current의 next가 deleted_node의 next를 가리키도록 설정
            if deleted_node.next:
                deleted_node.next.prev = current  # deleted_node의 next의 prev가 current를 가리키도록 설정
            if current.next is None:
                self.tail = current  # 삭제 후 current가 마지막 노드라면 tail 업데이트
        return deleted_data

    def search(self, data):
        current = self.head
        position = 0
        while current:
            if current.data == data:
                return position  # 데이터를 찾으면 위치 반환
            current = current.next
            position += 1
        return -1  # 데이터를 찾지 못하면 -1 반환

    def __str__(self):
        result = []
        current = self.head
        while current:
            result.append(current.data)  # 리스트의 모든 노드를 순회하며 데이터를 결과 리스트에 추가
            current = current.next
        return str(result)  # 결과 리스트를 문자열로 변환하여 반환

dll = DoublyLinkedList()
dll.append(1)
dll.append(2)
dll.append(3)
print(dll)  # [1, 2, 3]

deleted_item = dll.delete(1)
print(f"Deleted item: {deleted_item}")  # 2
print(dll)  # [1, 2, 3]
```

---

# 트리

- **비선형 구조**  /  원소들 간에 1:N / 계층 관계를 가지는 자료구조 / 폴더             <> N:M = 그래프
  ㄴ **데이터 순차적 X, 트리 & 그래프** 

<> 선형구조 : 데이터 사이에 **순서 존재**, 1:1관계 연결 [ 스택 큐 리스트 연결리스트 … ]

- 상위 원소 → 하위 원소로 내려가면서 확장되는 나무모양의 구조

### 트리 정의 :

- 한 개 이상의 노드로 이루어진 유한 집합 & 다음 조건을 만족해야 한다 : 
- 노드 중 최상위 노드 : 루트 ( 노드는 트리의 원소 ) 
- 나머지 노드들은 n ( 0 이상 ) 개의 분리 집합 T1 … TN 으로 분리될 수 있다 
- 이들 T1 … TN 은 하나의 트리가 되며(재귀적 정의),  루트의 **sub tree** 라 한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/47981e2a-1fc1-47ec-b3e8-6f2eb6560db4/Untitled.png)

## 트리 용어

- 노드 : 트리의 원소  A,B,…  /  간선 : 노드와 노드 / 자식과 부모 연결하는 선
루트 노드 : A 
**리프 노드 : 맨 끝 노드 ( == 단말 노드 )** 
**형제** 노드 : **부모가 같은 자식** 노드들
**조상** 노드 : 간선을 따라 **루트까지 있는 모든 노드들
자손** 노드 : 본인 제외 서브 트리에 있는 **하위 레벨의 노드들**

- 서브 트리 : 부모 노드와 연결된 **간선 끊었을 때 생성되는 트리**

- 차수 : == 자수(자식 수)

**노드의 차수 - 노드에 연결된 자식 노드 수** / B의 차수 2, C의 차수 1

**트리의 차수** : **트리에 있는 노드에서 가장 큰 차수** ( D가 3개 있으니까 3 ) 
단말 노드 : 치수가 0인 노드, 즉 **자식 노드가 없는 노드**

- 레벨 : **루트까지의 거리** / 루트 레벨은 0, 자식 레벨은 부모 레벨 + 1
  VS
높이 : **리프 노드까지의 간선 수** / 리프 노드의 높이는 0, 트리의 높이는 최대 레벨

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dc5c55bf-2e9a-410f-aa2c-ccf28a0e422e/Untitled.png)

## 이진 트리 ( 자식 없어도, 하나여도 )

- 차수가 2인 트리  /  각 노드가 자식 노드를 **최대 2개**까지만
- **모든 노드들이 최대 2개의 서브트리** 갖는 특별한 형태의 트리
- 노드 번호 : 루트 1번으로 하여 BFS 식으로

### 특성 :

- 높이 i (레벨 i) 에서의 노드의 최대 개수는 2개
- **높이가 h**인 이진 트리가 가질 수 있는 **노드의 최소 개수 h+1개       << 편향 이진 트리 
                                                                        최대 개수 2^(n+1) - 1 개   << 포화 이진 트리**

### 포화 이진 트리 :

- 모든 레벨에 노드가 포화 상태로 차 있는 이진 트리
- 높이가 h일 때, 최대 노드 개수를 가진 이진 트리 ( 높이가 3일 때 노드 15개 )
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6fec61b3-881a-41fb-ad91-94e727ab2ed6/Untitled.png)
    

### 완전 이진 트리 :

- 높이가 h이고 노드 수가 n일 때, h+1 ≤ n < 2^(h+1) - 1
포화 이진 트리의 노드 번호 1 ~ n 번 까지 **빈 자리가 없는 이진 트리**
- **번호를 더 넣으면 포화 이진 트리가 되어야 함**
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/26c127c3-e0cf-4ece-8c9f-6bf353232051/Untitled.png)
    

### 편향 이진 트리 : 시간복잡도 - (O(N))

- 높이 h에 대한 최소 개수의 노드를 가지며, 한쪽 방향의 자식 노드만을 가진 이진 트리
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a0a6d66b-5cf2-4fa9-8285-9fe8de38e2a4/Untitled.png)
    

### 리스트 이용한 이진 트리의 표현

- 이진 트리에 각 노드 번호를 다음과 같이 부여  /  루트 : 1번
- 레벨 n에 있는 노드에 대해 BFS 방식으로 번호 부여
- **노드 번호를 리스트의 idx**로 사용
- 높이 h 이진 트리 **배열의 크기** :  레벨 i의 최대 노드 수 : 2^i  **리스트의 크기 : 2 ^(h + 1)**
- **맨 왼쪽의 노드들의 번호 : 2^(레벨)  ==**      레벨 n의 노드 번호 시작 번호 : 2^n
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/621dad89-650e-4454-a59f-82b783afaa23/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/974a5ffe-54c0-4cba-aa4b-374463ddf4aa/Untitled.png)
    

- **노드 번호 i일 때 : 부모 노드 번호  :   i // 2 
                            왼쪽 자식 노드   :   2 * i
                            오른 자식 노드   : 2 * i + 1**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/87b2eafb-9aa1-48aa-bf81-238159834e1e/Untitled.png)

편향 이진 트리일때 너무 안좋음! 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/29292653-a8f1-43f3-9279-445f1bd83811/Untitled.png)

```python
# 리스트를 이용한 이진 트리 구현 
"""
**0번째 idx 안비움** > 위에서 배운 것과 다르게 짬 
**왼쪽 자식 노드 : 2 * i + 1  /  우측 자식 노드 : 2 * i + 2  /  부모 노드 : (i - 1) // 2**
set_node idx 넣어 줄 때, 중간 노드들 비어있다면 None 으로 채워주고 해당 칸에 값 넣기 
"""
class BinaryTree:
    def __init__(self):
        # 이진 트리를 저장할 리스트
        self.tree = ['A', 'B', 'C', 'D', 'E',
                     'F', 'G', 'H', 'I', 'J', 'K', 'L']
    
    def insert(self, value):
        # 리스트 끝에 값을 추가하여 트리에 삽입
        self.tree.append(value)

    def get_node(self, index):
        # 주어진 인덱스의 노드를 반환, 인덱스가 유효하지 않으면 None 반환
        if index < len(self.tree):
            return self.tree[index]
        return None

    def set_node(self, index, value):
        # 주어진 인덱스에 노드 값을 설정
        # 만약 인덱스가 현재 리스트의 범위를 벗어나면 리스트를 확장
        while len(self.tree) <= index:
            self.tree.append(None)
        self.tree[index] = value
    
    def get_left_child(self, index):
        # 주어진 인덱스의 왼쪽 자식 인덱스 계산
        left_index = 2 * index + 1
        # 왼쪽 자식이 리스트 범위 내에 있으면 해당 값을 반환
        if left_index < len(self.tree):
            return self.tree[left_index]
        return None
    
    def get_right_child(self, index):
        # 주어진 인덱스의 오른쪽 자식 인덱스 계산
        right_index = 2 * index + 2
        # 오른쪽 자식이 리스트 범위 내에 있으면 해당 값을 반환
        if right_index < len(self.tree):
            return self.tree[right_index]
        return None
    
    def get_parent(self, index):
        # 루트 노드는 부모가 없으므로 None 반환
        if index == 0:
            return None
        # 주어진 인덱스의 부모 인덱스 계산
        parent_index = (index - 1) // 2
        return self.tree[parent_index]
    
    def __str__(self):
        return str(self.tree)

# 이진 트리 생성
bt = BinaryTree()
bt.insert('M')

# 이진 트리 출력 
print("Binary Tree:", bt)
# 루트 노드 반환
print('루트노드:', bt.get_node(0))
# 'B', 'C' 의 부모 노드 반환 
print('B의 부모노드:', bt.get_parent(1))
print('C의 부모노드:', bt.get_parent(2))
# 루트 노드의 왼쪽 자식 B 출력
print('루트노드 왼쪽자식:', bt.get_left_child(0))  
# 루트 노드의 오른쪽 자식 C 출력
print('루트노드 오른쪽자식:', bt.get_right_child(0)) 
```

- 단점 : 
편향 이진 트리 > 사용X 원소에 대한 메모리 공간 낭비
트리 중간에 새 노드 삽입, 기존의 노드 삭제 > 리스트의 크기 변경 어려워 비효율적
- 이 단점 해결 위해서 연결 리스트로 이진 트리 구현

### 연결 리스트 이용한 이진 트리 표현

- 이진 트리의 모든 노드 : 최대 2개 자식 노드 >일정 구조의 이중 연결 리스트 노드를  사용해 구현

 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a5c9a588-e885-4c00-b85e-8bceb6517a17/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/99afbd09-9312-4458-9628-1146a36022a8/Untitled.png)

```python
# 소스 
class TreeNode:
    def __init__(self, value):
        self.value = value  # 노드의 값을 저장
        self.left = None    # 왼쪽 자식 노드를 가리키는 포인터
        self.right = None   # 오른쪽 자식 노드를 가리키는 포인터

class BinaryTree:
    def __init__(self):
        self.root = None  # 루트 노드 초기화
    
    def insert(self, value):
        if self.root is None:
            self.root = TreeNode(value)  # 트리가 비어 있으면 루트 노드로 설정
        else:
            self._insert_recursive(self.root, value)  # 루트 노드부터 재귀적으로 삽입
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)  # 왼쪽 자식이 없으면 왼쪽에 삽입
            else:
                self._insert_recursive(node.left, value)  # 왼쪽 자식이 있으면 재귀적으로 왼쪽에 삽입
        else:
            if node.right is None:
                node.right = TreeNode(value)  # 오른쪽 자식이 없으면 오른쪽에 삽입
            else:
                self._insert_recursive(node.right, value)  # 오른쪽 자식이 있으면 재귀적으로 오른쪽에 삽입
    
    def search(self, value):
        return self._search_recursive(self.root, value)  # 루트 노드부터 재귀적으로 검색
    
    def _search_recursive(self, node, value):
        if node is None or node.value == value:
            return node  # 노드가 없거나 값을 찾으면 해당 노드를 반환
        if value < node.value:
            return self._search_recursive(node.left, value)  # 왼쪽 자식에서 재귀적으로 검색
        else:
            return self._search_recursive(node.right, value)  # 오른쪽 자식에서 재귀적으로 검색
    
    def inorder_traversal(self):
        nodes = []
        self._inorder_recursive(self.root, nodes)  # 중위 순회 시작
        return nodes
    
    def _inorder_recursive(self, node, nodes):
        if node:
            self._inorder_recursive(node.left, nodes)  # 왼쪽 자식을 중위 순회
            nodes.append(node.value)  # 현재 노드의 값을 추가
            self._inorder_recursive(node.right, nodes)  # 오른쪽 자식을 중위 순회
    
    def __str__(self):
        return str(self.inorder_traversal())

# 이진 트리 생성 및 값 삽입
bt = BinaryTree()
bt.insert('A')
bt.insert('B')
bt.insert('C')
bt.insert('D')
bt.insert('E')
bt.insert('F')
bt.insert('G')
bt.insert('H')
bt.insert('I')
bt.insert('J')
bt.insert('K')
bt.insert('L')

# 이진 트리 출력
print("Binary Tree (Inorder Traversal):", bt)

# 노드 검색
print("Search for node with value 'E':", bt.search('E')) 
print("Search for node with value 'E':", bt.search('E').value)  
print("Search for node with value 'Z':", bt.search('Z')) 

```

---

## 이진 트리 순회 :

- 순회 : 트리의 노드들 체계적으로 방문
- 모든 노드 탐방하는 문제 풀 때에 비선형이라 순회하는 방법을 따로 익혀야 함
- 3가지의 기본적인 순회 방법 : 부모를 언제 방문하냐에 따라 나뉨

**1. 전위순회 : 부모 > 자식 좌 > 자식 우**

- 디렉토리 탐색할 때에 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7522bf25-9e76-482d-9f77-1aa3e0786243/Untitled.png)

요소 넣는 부분을 날먹; 

```python
# 구현 소스, 리프 노드에서 None 반환해 return 
class TreeNode:
    def __init__(self, key):
        self.left = None  # 왼쪽 자식 노드
        self.right = None  # 오른쪽 자식 노드
        self.val = key  # 노드의 값

def preorder_traversal(root):
    if root:
        print(root.val)  # 현재 노드 방문
        preorder_traversal(root.left)  # 왼쪽 서브트리 방문
        preorder_traversal(root.right)  # 오른쪽 서브트리 방문
def inorder_traversal(root):
    if root:
        inorder_traversal(root.left)  # 왼쪽 서브트리 방문
        print(root.val)  # 현재 노드 방문
        inorder_traversal(root.right)  # 오른쪽 서브트리 방문
def postorder_traversal(root):
    if root:
        postorder_traversal(root.left)  # 왼쪽 서브트리 방문
        postorder_traversal(root.right)  # 오른쪽 서브트리 방문
        print(root.val)  # 현재 노드 방문

# 트리 생성
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)

preorder_traversal(root)  # 1 2 4 5 3
```

2. 중위순회 : 왼쪽 자식 > 부모 > 자식 우 

- 정렬된 순서로 출력

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/231e2276-2c77-4f2e-9a08-059fa11fbbac/Untitled.png)

3. 후위순회 : 자식 좌 > 자식 우 > 부모

- 디렉토리 총 크기 구할 때에 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b78d8991-6229-4a65-bb4d-45a73fa1c799/Untitled.png)

### 수식 (이진) 트리 : 문제 DOWN

- 수식 표현하는 이진 트리
- 연산자는 루트 노드이거나 가지 노드(중간에 있는 노드) / 피연산자는 모두 리프 노드

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/25e7182f-25c9-4593-bd2b-195c37b7fc44/Untitled.png)

- 수식 트리 순회 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9e4bf4c3-5ed8-4d50-b334-28a450cef2cd/Untitled.png)

---

# 그래프

## 그래프 유형과 표현

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6e9da9ff-de72-45be-acc2-7f0d5a361b4e/Untitled.png)

- 아이템 ( 사물 또는 추상적 개념 ) 들과 이들 사이의 연결 관계 표현
- 정점 : 그래프의 구성요소 / 하나의 연결점  &  간선 : 두 정점 연결하는 선
- **차수 : 정점에 연결된 간선의 수**
- 경로 : 어떤 정점 A에서 B로 끝나는 순회, 두 정점 사이를 잇는 간선을 순서대로 나열한 것
- 싸이클 : 경로의 시작/끝 정점이 같음. ( 유향 그래프 )
- 인접 : 두개의 정점에 간선이 존재하면, 서로 인접해 있다고 한다.
- 그래프는 정점들의 집합, 이들을 연결하는 간선의 집합으로 구성된 자료 구조
V : 정점의 개수, E : 간선의 개수  /  **최대 E : V * (V-1) / 2**
- 선형이나 트리로 표현하기 어려운 N:M 관계를 가지는 원소들 표현하기에 용이

- 유형 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3130d380-8e3a-44b2-b387-77d70af32e70/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3cadc9e7-72d8-4fe7-9902-6ba111cf7228/Untitled.png)

- 표현 : 간선 정보 저장 방식  / 메모리 / 성능을 고려해 결정
1. 인접 행렬 : **두 정점 연결하는 간선의 유무를 행렬**로 표현 :

V*V 정방 행렬 (행의 수 == 열의 수)  /  행번호, 열번호는 V 따라서
두 정점이 인접되어 있으면 1 / 아니면 0 

무향 그래프 > i 번째 행의 합 = i번째 열의 합 = Vi의 차수 
유향 그래프 > 행 i의 합 = Vi의 진출 차수 / 열 i의 합 = Vi의 진입 차수 rrrr  

장점 : 
두 정점 사이에 간선 있는지 확인하는 연산이 O(1)로 빠름  /  구현이 단순
정적 그래프(V, E 개수 변함 X) 에 적합

단점 : 
메모리 많이 사용 ( 공간 복잡도 (O(V^2) ) 
간선 수를 확인하거나 인접한 정점을 나열하는 연산이 느림 ( O(N) )

사용하기 좋은 상황 : 
밀집 그래스 / 두 정점 사이에 간선 있는지 빠르게 확인해야 할 때에 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a998ab59-0de0-4581-a3d8-321adb93edde/Untitled.png)

- 인접 리스트 : **익숙해져야**
각 정점마다 다른 정점으로 나가는 간선의 정보 연결 리스트로 저장

장점 : 
필요한 공간만 사용 > O(V+E)  <  연결 리스트라 
인접한 정점 나열하는 연산이 빠름

단점 : 
두 정점 사이에 간선 있는지 확인하는 연산이 느림 > O(V)
LL로 구현이 복잡

사용하기 좋은 상황 : 
희소 그래프(Sparse Graph)에 적합  <  정점 많지만 간선 적은 
그래프가 동적으로 변하는 경우(정점, 간선이 자주 추가/삭제)
인접한 정점 자주 순회해야 하는 경우

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3f2d26d5-2700-4608-aa96-f72989bcb375/Untitled.png)

- 간선 리스트 : 
두 정점에 대한 간선 그 자체를 객체로 표현 > 리스트에 저장 
간선 표현하는 두 정점의 정보를 나타냄 ( 시작, 끝 정점 ) 

장점 : 
필요한 간선만 저장 > 공간 복잡도 O(E)   <   최소신장트리
간선을 직접 다루는 연산에 효율적

단점 : 
두 정점 사이에 간선 있는지 확인하는 연산이 느림 > O(E)
특정 정점에 인접한 정점을 찾는 연산이 느림 > O(E) 

사용하기 좋은 상황 : 
간선 중심 연산을 자주 수행해야 하는 경우에 ( MST )
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/18726d0d-24d8-4fc8-9379-8ece3ab624ef/Untitled.png)
    

```python
### 그래프 문제 **INPUT LIST**

# 인접 행렬 
adj_matrix = [
    [0, 1, 1, 0, 0, 1, 1],  # 0번 정점
    [1, 0, 0, 1, 0, 0, 0],  # 1번 정점
    [1, 0, 0, 1, 0, 0, 0],  # 2번 정점
    [0, 1, 1, 0, 1, 1, 1],  # 3번 정점
    [0, 0, 0, 1, 0, 0, 1],  # 4번 정점
    [1, 0, 0, 1, 0, 0, 0],  # 5번 정점
    [1, 0, 0, 1, 1, 0, 0]   # 6번 정점
]

# 인접 리스트
adj_list = {
    0: [1, 2, 5, 6],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2, 4, 5, 6],
    4: [3, 6],
    5: [0, 3],
    6: [0, 3, 4]
}

# 간선 리스트
edges = [
    (0, 1),
    (0, 2),
    (0, 5),
    (0, 6),
    (1, 3),
    (2, 3),
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 6)
]

```

## 참고

### BST :

- 개요
- 빠른 탐색의 필요성 : 
대량의 데이터를 다루게 되며 효율적인 저장 / 검색 / 수정이 중요해짐

선형 데이터 구조의 한계 : 
   배열 - 데이터 삽입,삭제 비효율적 > O(N)
   연결 리스트 - 검색이 비효율적 >  O(N)
   데이터 양 많을수록 성능 저하됨
- BST 정의 : 데이터 저장/검색/삽입/삭제 효율적으로 처리하기 위한 자료구조

- BST 특징 : 각 노드가 최대 2개 자식 
                 데이터를 정렬된 형태 - 순서속성 로 저장 > 탐색, 삽입, 삭제 효율적 수행

- BST 구조 : 이진 트리의 특성을 가지지만, 추가 속성을 가짐 ( **순서속성** )
- 왼쪽 자식 노드의 키값 < 부모 노드의 키값 < 우측 자식 노드의 키값

- BST 장점 : 
 - 배열이나 LL과 달리, 삽입/삭제 후에도 데이터가 정렬된 상태 유지
 - 데이터가 균형있게 분포되어 있을 떄에, 평균적 탐색,삽입,삭제 > O(logN)
 - 동적으로 크기 조정 가능 > 크기가 고정된 배열에 비해 유연성이 높음

- BST 단점 :  >  AVL / 레드블랙트리로 해결
 - 트리가 한쪽으로 치우치는 최악의 경우 > O(N)
 - 각 노드는 2개의 자식 포인터 지정해야 > 큰 데이터 집합의 경우 메모리 오버헤드 발생 가능

- 균형 잡힌 트리의 높이 : O(logN) 
불균형 > O(N)

- 주요연산 : 탐색
*** 노드보다 크면 우측 / 작으면 왼쪽으로 이동 
없는 키값이면 None 찾았을떄 끝

- 주요연산 : 삽입
 - 순서속성 유지 위해 **새로운 노드를 ‘리프 노드’에 삽입** 
 - 처음에는 탐색과 동일하게 내려감 > 자리 찾으면 들어가기
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9bcba04e-9daf-42a5-bf25-5de029240dfb/Untitled.png)
    

- 주요연산 : 삭제   >   삭제하려는 노드의 위치/자식 노드 유무에 따라 3가지 경우로 나누어 처리
1. 삭제할 노드가 리프 : 
- 삭제할 노드 찾는거는 탐색과 동일하게 진행
- 리프니까 단순하게 제거

1. 삭제할 노드가 1개의 자식 노드 : 
- 삭제할 노드의 자식 노드를 부모 노드에 연결
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/769e84ff-7e9f-41fc-a72b-a75634f2b898/Untitled.png)
    

1. 삭제할 노드가 2개의 자식 노드 : 
- 중위 후속자 OR 중위 전임자 찾기 
중위 후속자 ? : 삭제할 노드의 오른쪽 서브 트리에서 가장 작은 값 ( 일반적 ) 
                               ㄴ 가장 삭제하려는 값과 가까운 
*** 중위 후속자는 무조건 리프 노드이거나, 하나의 자식만을 가짐** 

중위 전임자 ? : 삭제할 노드의 **왼쪽 서브 트리에서 가장 큰 값** 

결론 : BST 구조를 유지하기 위해서, 삭제할 노드와 가장 가까운 값을 찾는 것

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5f6941cc-8d62-4f83-8664-5098e2b3513d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4c85fc65-7fdd-4a6b-bf9a-34a3c3b7380c/Untitled.png)

- 중위 후속자(의 값)을 삭제할 노드에 복사 
자식 노드가 1개이므로 삭제 과정 진행 ( 18을 17(15자리에 온) 에 연결 )
- 이후 ???????

- BST의 균형과 불균형
1. BST의 구조는 삽입되는 데이터의 순서에 따라 결정 > 특정 패턴으로 삽입되면 불균형 발생
2. 불균형한 BST의 문제 ? 
- 검색,삽입,삭제 연산의 시간복잡도 O(N)
- 트리의 높이가 증가 > 많은 메모리 공간 필요
- 트리의 높이 증가 > 깊이 있는 재귀호출 > 스택 오버플로우 문제 발생 가능 
3. 해결 ? 
- 자가 균형 트리 ( Self-Balancing Tree ) : AVL / 레드-블랙 트리 

```python
class Node:
    def __init__(self, key):
        self.key = key  # 노드의 값
        self.left = None  # 왼쪽 자식 노드
        self.right = None  # 오른쪽 자식 노드

class BST:
    def __init__(self):
        self.root = None  # 루트 노드를 초기화

    def insert(self, key):
        if self.root is None:
            self.root = Node(key)  # 트리가 비어 있으면 루트 노드로 설정
        else:
            self._insert(self.root, key)  # 재귀적으로 삽입

    def _insert(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = Node(key)  # 왼쪽 자식이 없으면 왼쪽에 삽입
            else:
                self._insert(node.left, key)  # 왼쪽 자식이 있으면 재귀적으로 삽입
        else:
            if node.right is None:
                node.right = Node(key)  # 오른쪽 자식이 없으면 오른쪽에 삽입
            else:
                self._insert(node.right, key)  # 오른쪽 자식이 있으면 재귀적으로 삽입

    def delete(self, key):
        self.root = self._delete(self.root, key)  # 루트 노드부터 재귀적으로 삭제

    def _delete(self, node, key):
        if node is None:
            return node  # 노드가 없으면 그대로 반환

        if key < node.key:
            node.left = self._delete(node.left, key)  # 왼쪽 서브트리에서 삭제
        elif key > node.key:
            node.right = self._delete(node.right, key)  # 오른쪽 서브트리에서 삭제
        else:
            if node.left is None:
                return node.right  # 왼쪽 자식이 없으면 오른쪽 자식을 반환
            elif node.right is None:
                return node.left  # 오른쪽 자식이 없으면 왼쪽 자식을 반환

            temp = self._minValueNode(node.right)  # 오른쪽 서브트리의 최소값 노드를 찾음
            node.key = temp.key  # 현재 노드의 키를 최소값 노드의 키로 대체
            node.right = self._delete(node.right, temp.key)  # 최소값 노드를 삭제

        return node

    def _minValueNode(self, node):
        current = node
        while current.left is not None:
            current = current.left  # 왼쪽 자식이 없을 때까지 이동하여 최소값 노드를 찾음
        return current

    def search(self, key):
        return self._search(self.root, key)  # 루트 노드부터 재귀적으로 검색

    def _search(self, node, key):
        if node is None or node.key == key:
            return node  # 노드가 없거나 값을 찾으면 해당 노드를 반환
        if key < node.key:
            return self._search(node.left, key)  # 왼쪽 서브트리에서 검색
        return self._search(node.right, key)  # 오른쪽 서브트리에서 검색

    def inorder(self):
        self._inorder(self.root)  # 중위 순회 시작
        print()

    def _inorder(self, node):
        if node:
            self._inorder(node.left)  
            print(node.key, end=' ') 
            self._inorder(node.right) 

# BST 생성
bst = BST()
bst.insert(10)
bst.insert(5)
bst.insert(15)

bst.inorder()

```

---

## 비선형 자료구조

- 탐색 : 비선형구조인 트리 / 그래프의 각 노드(정점)을 **중복되지 않**게 전부 방문하는 것
비전형구조는 선형(선후연결  1:1  리스트 / 스택 / 큐)에서와 같이???선후 연결관계를 알 수 없다. > 특별한 방법 필요  :  DFS / BFS

---

# DFS : 모의 A형 거의 나온다 // BFS나 DFS 중 1

## DFS : 트리

- 구현 / 이해 쉽다  <  사이클 X   =   중복 노드 탐색 걱정 X
부모 - 자식 (1:N) 구조
- 루트 노드에서 출발해 한 방향으로 갈 수 있는 경로가 있는 곳까지 깊이 탐색
갈곳 없으면 가장 마지막에 만난 갈림길 노드로 되돌아와 
다른 방향의 노드로 탐색을 반복하여 모든 노드를 방문하는 순회 방법
- 가장 마지막에 만난 갈림길의 노드로 되돌아 가서 다시 DFS 를 반복해야 하므로 
**재귀적으로 구현**하거나 **후입선출 구조의 스택** 사용해 구현

### DFS SUDO

- V 방문이 PRINT 찍는 부분
- 자식노드 없으면 리턴

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/517439f1-31fc-43e6-b192-b5b3402b272b/Untitled.png)

```python
tree = {'A' : ['B', 'C', 'D'],
        'B' : ['E', 'F'],
        'D' : ['G', 'H', 'I']}

def dfs(tree, node):

    print(node) # 방문 표시 

    if node not in tree: # **키값 아니면** == 자식노드 없으면 탈출
        return

    # 자식노드 있으면 해당 키값의 value 의 자식들 보면서 dfs 돌려주기
    for child in tree[node]:
        dfs(tree, child)

dfs(tree, 'A')
```

## DFS : 그래프 <  일반적으로 그래프 유형 문제가 나옴

- 시작 정점에서 출발해 한 방향으로 갈 수 있는 경로가 있는 곳까지 깊이 탐색하다가
갈곳 없으면 가장 마지막에 만났던 갈림길 간선 있는 정점으로 되돌아와 
다른 방향의 정점으로 탐색을 계속 반복하여 결국 모든 정점을 방문하는 순회방법 
   ㄴ  트리와 동일한 설명 but
- 사이클 있을수도   (서로 N:M으로 얽혀 있어서 중복탐색 걱정 해야한다 )
  ㄴ 중복 노드 탐색 걱정해야 한다
  ㄴ **방문 노드 체크**해야 한다 

모든 노드가 인접하지 않을 수 있다 
  **ㄴ 모든 노드에 대해 dfs** 해줘야 한다 
****
- idea : 
visited 1차원 배열 사용 ( V만큼 )
호출 스택 : 방문한 V 순서대로 PUSH 

방문 >  visited True로 + 연결되어 있는 노드들의 DFS() 진행

방문시 visited = True인 인접 정점에는 DFS 돌리지 않음 
방문시 인접 노드 없을 때에 스택 POP 진행

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/93ac5cf5-0af9-4882-ac69-c1b2b1fb6bc0/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/65ee848b-7cc3-4f4e-a571-5eae79ffe5bc/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/181e5752-6599-4bcb-b6be-6402005bb530/Untitled.png)

```python
# DFS 인접행렬 코드

# current : 탐색 정점, adj_matrix : 인접 행렬 ( 각 노드의 )
def dfs(current, adj_matrix, visited):
		visited[current] = True # 방문 표시
		
		for i in range(len(adj_matrix)):
				if adj_matrix[current][i] and not visited[i]: # 인접해 있는데 방문 안했으면
						dfs(i, adj_matrix, visited) # dfs 돌려
						
N = 5
adj_matrix = [
		[0,1,1,0,0],
		[1,0,0,1,1],
		[1,0,0,0,1],
		[0,1,0,0,1],
		[0,1,1,1,0]
]
visited = [False] * N
dfs(0, adj_matrix, visited)		
```

```python
# 연습 문제  <<< **set 쓰는거 되게 좋은 아이디어인듯** 
# graph : 그래프(인접 리스트) / start : 탐색을 시작할 정점
# visited : 방문한 정점을 저장 집합 / result : 탐색 경로를 저장하는 리스트

def dfs(graph, start, visited, result):

    **visited.add**(start)    # 현재 정점을 방문 표시 < set이라서 add 
    result.append(start)  # 현재 정점을 탐색 경로에 추가

    for neighbor in graph[start]: # 현재 정점에 연결되어 있는 노드들 (value) 를 돌며
        if neighbor not in visited: # 방문 안했었으면
            dfs(graph, neighbor, visited, result) # 해당 노드에 대해서 dfs 

# 그래프 인접 리스트
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'G'],
    'D': ['B', 'F'],
    'E': ['B', 'F'],
    'F': ['D', 'E', 'G'],
    'G': ['C', 'F']
}

start_vertex = 'A'
# **set 중복이 없고, 특정 원소가 set에 포함되어 있는 지 확인하는 과정이 시간복잡도 O(1)**
# 리스트는 포함되어 있는 원소 유무를 확인하는데 시간복잡도 O(N)
visited = set()
result = []  # 탐색 경로를 저장할 리스트

dfs(graph, start_vertex, visited, result)

```

---

# DFS 문제풀이

- 장훈이의 높은 선반
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d88beae4-f134-4b18-9070-d481acf2af83/Untitled.png)
    

처음 아이디어 : 완전탐색 ( 부분집합 ) 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e4a1b8ae-afd6-4717-992e-3e37a38f4eb1/Untitled.png)

```python
import sys
from itertools import combinations
sys.stdin = open("1486_input.txt")
"""
itertools 
1. 내부 구현이 C, 파이썬 코드보다 빠릅니다.
2. 제너레이터를 활용 ( 모든 조합을 한 번에 생성 X , 필요할 때마다 생성을 해요 ) 
(대량의 객체를 만들지 않아서 오버헤드도 없고 빠릅니다. 
(작은 단위로 데이터를 처리해서, CPU 캐시를 효율적으로 사용합니다) 
3. 내장함수여 인터프리터에 의해서 최적화, 알고리즘 자체도 극한으로 최적화 
"""

"""
모든 조합의 경우의 수를 구한다 => subset(부분집합), 시간복잡도 : O(2^n) 
문제에서 N의 최대값이 최대 몇이였다? 20이였다. => 2의 20승 => 2의 10승 = 1024 , 대략 백만 
못할 건 아닌거같아요. 백만
"""

"""
DFS 로 하니까 속도가 빨라졌네 ? 
"""

"""
모든 조합을 구하는 것 => 부분집합을 구하는 것
부분집합을 가장 간단하게 구현하는 방법 => 비트마스크 활용 
"""
def get_combinations_recur(arr, n):
    result = []  # 결과를 저장할 빈 리스트 초기화

    if n == 1:  # 선택할 요소가 1인 경우, 각 요소 자체가 하나의 조합이 된다.
        return [[i] for i in arr]

    for i in range(len(arr)):
        elem = arr[i]  # i 번재 요소를 고정시켜 놓고
        # 현재 고정시켜 놓은 요소 다음부터 재귀로 넘긴다.
        # 한 개를 이미 골랐으니까, 다음 재귀로 넘길때는 한 개를 빼서 넘긴다.
        for rest in get_combinations_recur(arr[i+1:], n-1):
            result.append([elem] + rest)

    return result  # 최종 조합 결과 반환

def get_combinations_dfs(arr, n):
    result = []  # 조합 목록을 저장하는 변수
    # start: 시작지점
    # comb: 여태까지 선택된 조합 목록
    def get_comb(start, comb):
        if len(comb) == n:  # 여태까지 선택한 조합 목록이 우리가 원하는 조합의 개수에 도달하면
            result.append(comb[:])

        # 파라미터로 건네진 start(시작점)부터 총 개수까지 순회
        for i in range(start, len(arr)):
            comb.append(arr[i])  # 선택하고
            get_comb(i + 1, comb)  # 선택한 거 다음부터 다시 선택을 해야하므로, 시작지점을 +1 해서 넘긴다.
            comb.pop()  # 해당 선택지는 위에서 끝났고, 다른 선택지를 다시 골라야 하기 때문에 최근에 선택한 것을 추출

    # 부분 집합 했을 때 기억나시죠 ?
    # 하나의 원소를 선택했다고 치고, 그 다음 위치의 인덱스부터 다시 하나를 선택하는거에요.
    # 현재 선택해야 하는 원소의 위치
    # 여태까지 고른 조합의 목록
    get_comb(0, [])
    return result

import time

start_time = time.time()
T = int(input())
for test_case in range(1, T + 1):
    # N: 사람 수, B: 목표 높이
    N, B = map(int, input().split())
    # 각 사람의 키를 입력 받아 리스트로 저장
    arr = list(map(int, input().split()))

    def get_min_tower_height(N, B, heights):
        min_height = float('inf')  # 우리는 조합을 구현하고, 최소값을 찾으면 최소값으로 갱신할 거다.

        # height(N) 명에 대해서 1명을 선택하는 것부터, N명을 모두 선택하는 것까지
        # 모든 조합을 구한다음에, 그 조합들의 합을 구하고, 거기서 B를 넘는 것중에 가장 작은 것을 구할거다.
        for r in range(1, N+1):  # 1명부터 N명을 선택하는 조합을 구하기 위해 순회
            for comb in combinations(heights, r):  # heights(직원들의 키 리스트)에서 r명을 선택한 조합
                total_height = sum(comb)  # 직원들 조합의 모든 키를 합한 값

                # 우리가 점원들의 합이 탑의 높이와 같아지면, 더이상 조합을 구할 필요가 없다
                # 가지치기, 백트래킹
                # if total_height == B:
                #     return B
                # 문제의 조건: B보다 크거나 같으면 최소값 갱신
                if total_height >= B:
                    min_height = min(min_height, total_height)
        return min_height

    res = get_min_tower_height(N, B, arr)

    print(f"#{test_case} {res - B}")

end_time = time.time()
print("실행시간: ", end_time - start_time)

```

- 비트마스크로

```python
import sys
import time
sys.stdin = open('1486_input.txt')

"""
비트 연산이 조합_재귀 / 조합_dfs 보다 빠른 이유 
1. 비트 연산은 cpu 직접 처리되기 때문
2. 정수 하나로 부분 집합을 표현 , 메모리 접근 최소화 

하지만 itertools 를 이길 수는 없다. ! 
"""
s_time = time.time()
T = int(input())
for tc in range(1, T+1):
    N, B = map(int, input().split())
    arr = list(map(int, input().split()))
    res = float('inf')

    def get_subsets_bitmask():
        # 아래에 부분집합을 비트마스크로 구현하는 코드 작성
        global res
        subset_cnt = 2 ** N

        # 모든 경우의 수
        for i in range(1, subset_cnt):  # i 에 해당하는 숫자들을 비트로 바꿔서 비트연산을
            h_sum = 0
            for j in range(N):
                if i & (1 << j):
                    h_sum += arr[j]
            if h_sum >= B:
                res = min(res, h_sum)
        return

    get_subsets_bitmask()
    print(f'#{tc} {res - B}')

e_time = time.time()
print(e_time - s_time)

```

- DFS로 해결하는게 좋다.

```python
import sys
import time
sys.stdin = open("1486_input.txt")

s_time = time.time()

T = int(input())
for test_case in range(1, T + 1):
    # N: 사람 수, B: 목표 높이
    N, B = map(int, input().split())
    arr = list(map(int, input().split()))
    res = 10000 * N

    # idx : 현태 탐색 중인 직원의 인덱스
    # h_sum : 현재까지 선택한 직원들의 키의 합
    def dfs(idx, h_sum):
        global res

        # 가지치기, 백트래킹 ( 이미 여기서는 더 진행해도 쓸모가 없기 때문에 스탑! )
        if h_sum >= res:
            return

        # 현재 키의 합이 목표 높이 이상이면, 최소값 갱신
        if idx == N:
            if h_sum >= B:
                res = min (res, h_sum)
            return

        # 이 문제는 결국 부분집합
        # 부분집합에서 해당 원소를 선택하냐 ? 하지 않느냐 ? => 응용해보자
        # 현재 idx가 가리키는 직원의 키를 포함하는 경우
        dfs(idx + 1, h_sum + arr[idx])

        # 현재 idx가 가리키는 직원의 키를 포함하지 않는 경우
        dfs(idx + 1, h_sum)

    # 조합을 DFS로 구현했을 때와 비슷하게
    # 현재 선택해야 하는 점원의 위치를 가리키는 파라미터
    # 여태까지 선택한 조합 목록 파라미터  => 결과가 필요하다 ( 점원들의 키의 합 )
    # 여태까지 선택된 직원들의 키의 합
    dfs(0, 0)

    # 목표 높이 B를 빼서 실제로 초과된 부분만 출력
    print(f"#{test_case} {res - B}")

e_time = time.time()

print(e_time - s_time)

```

---

- 동철이의 일 분배
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8dbe20f8-2be0-4e5f-a184-57028cad7e6a/Untitled.png)
    

```python
import sys
sys.stdin = open('1865_input.txt', "r")

# select_idx: 현재 선택중인 idx
# per: 현재까지 선택된 직원들의 능률의 곱
def search(select_idx, per):
    global result

    # 이미 작아진 per는 뭘 곱하던 result보다 커질 수 없음
    # 뒤에 작업들은 무의미함. => return으로 그냥 끝냄
    # 가지치기, 백트래킹
    if per <= result:
        return

    if select_idx == N:  # 직원들이 전부 일을 분배 받은 경우
        result = max(result, per)
        return

    # 일을 분배하기 시작할거에요
    for i in range(N):
        if visited[i]: continue  # 이 일이 어떤 직원에게 할당을 받은 경우 skip
        # 이제부터는 아직 할당 안받은 친구들
        visited[i] = True
        # 다음 직원이 선택하도록 재귀호출 하는겁니다.
        # 현재 직원의 일 능률 리스트에서 이번에 선택한 일의 능률을 가져와서 곱하고 넘깁니다.
        search(select_idx + 1, per * arr[select_idx][i])
        # 이 일을 선택하지 않았던 걸로 만들고, 다시 다른 일을 선택해서 dfs를 반복 ( 갈림길로 돌아와서 다른 방향으로 가는거)
        visited[i] = False

T = int(input())
for tc in range(1, T + 1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]

    # 효율을 볼 때는 결국 소수점으로 계산을 해야한다.
    # 20 20 => 20% * 20% => 4% => 0.2 * 0.2 = 0.04
    # 입력받은 모든 수를 소수점으로 변환해야 한다.
    for i in range(N):
        for j in range(N):
            arr[i][j] = arr[i][j] / 100

    # 순열, 순열이랑 비슷하게 구현을 할건데,
    # 일이 할당을 직원에게 받았는 지 확인하는 방문 체크 변수를 만들어야 한다.
    visited = [0] * N # 직원 X, 일이 할당을 받얐냐.
    result = 0

    # 함수 작성 필요
    # dfs 에서 일반적으로는 파라미터로 넘기는 값이 있어요.
    # 결과값이랑 재귀적으로 파고들었을 때 종료시점을 결정하는 변수를 파라미터로 넘깁니다.
    # 1. 직원을 선택하는 인덱스
    # 2. 현재까지 선택한 직원들의 효율의 곱을 나타내는 파라미터
    search(0, 1)

    print(f'#{tc}')
```

---

# BFS : 너비 우선 탐색

- 루트 노드의 자식 노드 모두 차례대로 방문 > 방문했던 자식 노드들 기준으로 하여 
다시 해당 노드의 자식 노드들 차례대로 방문 > 인접한 노드들에 대해 탐색 > 차례로 다시 BFS >> 선입선출 형태의 자료구조인 큐 활용

## BFS : 트리

- 사이클 X
- SUDO CODE

```python
def bfs():
	# 큐 생성
	# 루트 v 큐에 삽입
	while len(q) != 0:
		# 큐의 첫 원소 t 반환
		# 첫 원소 t 방문
		for (t와 연결된 모든 간선에 대해):
			# t의 자식노드 u 반환
			# u를 큐에 삽입
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/31c73b56-6006-4ee7-bfd4-4e1398c5af4c/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a458d284-d001-41de-a2a7-5ecd516c6845/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7513a13b-82d5-4099-9cce-fa335d26485d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ea375d35-108c-49b3-85e6-cca09fa141dc/Untitled.png)

```python
from collections import deque

def bfs_tree(tree, root_node):
    queue = deque([root_node]) # 큐 생성 & 루트 노드 큐에 삽입
    result = []                # 탐색 경로 저장할 리스트 

    while queue:               # 큐 비어있지 않다면 반복
        node = queue.popleft() # 원소 하나 빼서
        result.append(node)    # 방문 체크 

        if node not in tree: continue # 자식노드 없으면 다음 큐 원소 확인
        for child in tree[node]:      # 자식노드 있으면 큐에 인 
            queue.append(child)
    
    return result

tree = {
    'A':['B', 'C', 'D'],
    'B':['E', 'F'],
    'D':['G', 'H', 'I'],
}

root_node = 'A'
result = bfs_tree(tree, root_node)
print(' '.join(result))
```

## BFS : 그래프

- 사이클이 존재 / N:M    < 방문체크 필요 !
- 연결이 안된 노드가 있을 수 있다 < 모든 노드에 대해 BFS
- 큐에 넣으면서 Visited False 면 True로 바꿔줘
- 큐가 비면 탐색 끝!

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6e28bb7d-30c1-4c3a-9822-4acae65d04cf/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/48eeddba-7b41-4035-bd08-236b72469119/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8ddbfed1-82d4-45ec-ad43-0e83e743dce7/Untitled.png)

```python
from collections import deque

# 포인트 : idx 를 찍는 이유 : graph 가 dict 형태라 ? 

def bfs(graph, start):
    nodes = list(graph.keys())    # 노드가 키 형태로 저장되어 있으므로, 받아와서 리스트로
    visited = [False] * len(nodes)# visited 배열 선언 
    queue = deque([start])        # 큐에 루트 넣고
    result = []

    start_index = nodes.index(start) # 노드의 idx 에서 'A'가 어딨는지 확인 후 IDX 저장 
    visited[start_index] = True      # VISITED 와 NODE 가 동일한 IDX 사용하므로 방문 찍어줌

    while queue:
        vertex = queue.popleft()     # 노드 하나 빼서
        result.append(vertex)        # 방문 찍어주고 

        for neighbor in graph[vertex]: # 그래프 dict에서 자식들 돌면서
            neighbor_index = nodes.index(neighbor) # node 배열에서 자식 idx 받아오고

            if not visited[neighbor_index]:    # visited 안찍었으면 방문 안했다는거니까
                queue.append(neighbor)         # 큐에 넣어주고
                visited[neighbor_index] = True # 방문 찍어주고 

    return result

graph = {
    'A':['B', 'C'],
    'B':['A', 'D', 'E'],
    'C':['A'],
    'D':['B', 'F'],
    'E':['B', 'F'],
    'F':['D', 'E', 'G'],
    'G':['F'],
}

start_node = 'A'
result = bfs(graph, start_node)
print("그래프 탐색 경로 : ", result)
```

### DFS VS BFS ?

- DFS : 목표 노드가 깊은 단계 있을 때에 속도 업 <> 스택 오버플로우 가능
처음 발견한 해가 최적해가 아닐 수 있다
**경로**의 특성을 알아야 할 때에 사용
- BFS : 단계적
노드 간 거리 찾기 쉬움 (**최적해**)

---

## BFS : 그래프

### 도로 이동시간 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/62a73b52-1272-4937-9c8b-2e166a5050f4/Untitled.png)

1. 문제 볼때 우선적으로 고려할 부분 : 출발 자리 빼고 생각 Y / N 

2. visited : 
장점 : 모든 좌표의 최소 이동거리 알 수 있다 
단점 : 공간 2배 사용 

 공간을 복사해서 -1 초기화 후 방문 여부 표시 → 각 좌표마다 그 좌표에 도달하는 이동 거리 기입 
 각 좌표 찍을 때마다 탐색 방향 → 하 우 상 좌 

 움직이는 조건 → [  visited 가 -1  ] + [ 이동할 곳의 value 가 0이 아닌 곳 = 도로인 곳 ]
 
 순서 : 
  0,0 출발 → visited 배열 지금 자리 0 으로 바꾸고 4방향 탐색 → 아래만 가능 → 1,0 값 1로 
  1,0 방향 탐색 → 위/아래 VALUE가 0 아닌데, 위는 visited 가 -1 이 아님 → 아래 이동 → 2,0 값 2 
  …
  3,0 방향 탐색 → 오른쪽 value 1 && visited -1 → 3,1 값 4
  … 
  **0,4 << 아래도 1 & -1 / 우측도 1 & -1 >> 둘다 큐에 in** 
  …

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2c0e8c4c-b206-4659-9f51-7be5769706d7/Untitled.png)

```python
 # bfs 1 - 큐에 다음에 방문할 좌표 넣는 식
 
 
from collections import deque
import sys
sys.stdin = open('test.txt')

# 1. 전체 공간을 복사해서 각 공간의 좌표마다 
#    시작지점에서 얼마나 이동했는지를 저장하는 방식으로 구현
# 장점 : 모든 목적지의 최단 거리를 알 수 있다 ( 조건 : 시작지점 ( 0,0 ) 에서 시작한다 )
# 단점 : 메모리 2배 차지 ( 어짜피 방문처리 해야 해서 쩔수 )

def bfs(road, n, m):
    # 4방향 탐색  >  하, 우, 상, 좌
    # dxy를 for 로 돌며 현재 좌표에 dx, dy로 더해주면 상하좌우 이동 가능 
    dxy = [[1,0], [0,1], [-1,0], [0,-1]]
    
    queue = deque([(0,0)]) # 그래프 관련 bfs 문제 : 좌표를 queue 에 넣는 것 주의 ! 
    
    distance = [[-1] * m for _ in range(n)] 
    # visited 배열 / 노드가 한번 방문한 적 있는지 확인 + 해당 좌표까지의 이동 거리

    distance[0][0] = 0 # 문제에서 처음 부분을 0,0 으로 하기로 했음  (  1인 문제들도 있을 수 있다  )

    while queue:
        x, y = queue.popleft()

        #### 중요!!! 
        for dx, dy in dxy:
            nx, ny = x+dx, y+dy # 이동하는 좌표 
            if 0 <= nx < n and 0 <= ny < m and distance[nx][ny] == -1 and road[nx][ny] == 1:
                queue.append((nx, ny)) # 이동한 좌표가 조건 맞으면 큐에 넣어야  
                distance[nx][ny] = distance[x][y] + 1  # shit

                if nx == n-1 and ny == m-1:  # 0 idx 사용하므로!!! 
                    return (distance[nx][ny])
    
    # 와일문을 나왔다? : 목적지에 도달하지 못했다.
    return -1

n, m = map(int, input().split()) # 세로 / 가로 
road = [list(map(int, input())) for _ in range(n)] 

result = bfs(road, n, m)
print(result)
```

```python
# bfs 2 

from collections import deque
import sys
sys.stdin = open('test.txt')

def bfs(road, n, m):

    # 이런 방식으로도 가능 
    dx = [1,0,-1,0]
    dy = [0,1,0,-1]

    # 방문 여부만 확인할 것 ( 여태까지 이동거리 저장 X )
    visited = [[False] * m for _ in range(n)]

    # 큐에 좌표 넣을 때에 부가 데이터 넣어달라고 하는 문제들 풀기 위해
    # 방문할 좌표 + 여태까지 이동한 횟수 큐에 넣기 
    queue = deque([(0,0,0)])

    while queue:
        x,y,dist = queue.popleft()

        for d in range(4):
            nx = x + dx[d]
            ny= y + dy[d]
            # 아까 좌표랑 다르게 이동거리도 구해 놓기 
            next_dist = dist + 1 

            if 0 <= nx < n and 0 <= ny < m and visited[nx][ny] == False and road[nx][ny] == 1:
                queue.append((nx, ny, next_dist))
                visited[nx][ny] = True
                if nx == n-1 and ny == m-1:
                    return next_dist

    return -1 
  

n, m = map(int, input().split()) # 세로 / 가로 
road = [list(map(int, input())) for _ in range(n)] 

result = bfs(road, n, m)
print(result)
###################################################################################
import sys
from collections import deque
sys.stdin = open('practice_input.txt')

# 1. 전체 공간을 복사해서 각 공간의 좌표마다
# 시작지점에서 얼마나 이동했는 지를 저장하는 방식으로 구현
# 장점: 모든 목적지의 최단 거리를 알 수 있다. ( 조건, 시작지점 (0,0) 에서 시작한다는 조건)
# 단점: 메모리를 2배로 차지한다. ( 어차피 방문처리를 해야해서 어쩔 수없는 메모리입니다)
def get_road_move_time(road, n, m):
    # 4방향 탐색을 해야한다.
    # 하, 우, 상, 좌
    # dxy를 for loop로 돌면서 현재 좌표에 dx, dy를 더해주면, 상하좌우로 이동할 수 있다.
    dxy = [[1,0], [0,1], [-1,0], [0, -1]]

    # BFS는 큐로 구현 => deque
    # 문제에서는 [1,1] => [n,m]
    # 입력값을 [0,0] => [n-1, m-1]
    queue = deque([(0, 0)])

    # 복사하고, 각 좌표까지 걸리는 최단 이동거리를 저장해야된다.
    # m = 3 , [-1] * 3 => [-1, -1, -1]
    # 노드가 한 번 방문한 적이 있는 지 확인하는 용도로도 사용이 된다.
    # 해당 좌표까지의 이동 거리
    # -1로 꼭 초기화하지 않아도 된다. 자유롭게 초기화하자
    distance = [[-1] * m for _ in range(n)]
    distance[0][0] = 0  # 처음 시작 부분은 0 으로 하기로 문제가 정했다.

    while queue:  # 큐가 빌때까지 원소를 꺼내고 방문하는 행위를 반복한다.
        x, y = queue.popleft()

        for dx, dy in dxy:  # dxy = [[1,0], [0,1], [-1,0], [0, -1]]
            nx, ny = x + dx, y + dy  # 현재 위치에서 각 방향으로 이동

            # [nx, ny]가 갈 수 있는 곳인지를 체크해야한다.
            # 1. 도로 범위안에 포함될 것
            # 2. 방문한 적이 없을 것 (내가 갈 곳이 (-1) 이여야 한다)
            # 3. 갈 수 있는 곳일 것 (길이여야 한다. (1) 이여야 한다)
            if 0 <= nx < n and 0 <= ny < m and distance[nx][ny] == -1 and road[nx][ny] == 1:
                # 위 조건을 다 만족했으므로, 이동할 수 있는 좌표
                queue.append((nx, ny))
                # 현재 위치까지 오는데 걸린 이동 횟수 + 1 값을
                # 다음에 이동해야 할 좌표에 입력을 한다.
                distance[nx][ny] = distance[x][y] + 1

                if nx == n-1 and ny == m-1:
                    return distance[nx][ny]

    # while 문을 빠져 나왔다는 소리는, 결국 목적지에 도착하지 못했다!
    # 목적지를 찾지 못하면 "어떤 걸" 출력하세요 !! (문제에서 제시가 됨)
    # 제시된 걸 반환하면 된다.
    return -1  # -1도 제가 임의로 -1이라고 한거고요..

n, m = map(int, input().split())  # 도로의 크기 n * m 입력 받기
road = [list(map(int, input())) for _ in range(n)]  # 도로 정보 입력

# BFS를 이용해서 이동시간 구하기
result = get_road_move_time(road, n, m)
print(result)
```

```python
# 얼리 리턴 : 미리 모든 조건들을 다 쳐내준다 > 디버깅 쉽다 

import sys
from collections import deque
sys.stdin = open('practice_input.txt')

# 개인적으로 제가 좋아하는 방식으로 코드 작성
# 로직이 메인이 아닌, 코드 작성 스타일(early return 활용)
def get_road_move_time(road, n, m):
    dxy = [(-1, 0), (1,0), (0, -1), (0, 1)]
    queue = deque([0,0])
    distance = [[-1] * m for _ in range(n)]

    while queue:
        x, y = queue.popleft()

        for dx, dy in dxy:
            nx, ny = x + dx, y + dy

            # early return
            # 조건 처리를 바로바로 하는겁니다.
            # 안쪽으로 깊게 코드가 파고드는 걸 막을 수 있고요

            # 도로 범위를 벗어난 경우
            if 0 > nx or nx >= n or 0 > ny or ny >= m:
                continue

            # 방문한 적이 있는 경우
            if distance[nx][ny] != -1:
                continue

            # 도로가 아닌 경우
            if road[nx][ny] == 0:
                continue

            queue.append((nx, ny))
            distance[nx][ny] = distance[nx][ny] + 1

            if nx == n-1 and ny == m-1:
                return distance[nx][ny]

    return -1

n, m = map(int, input().split())  # 도로의 크기 n * m 입력 받기
road = [list(map(int, input())) for _ in range(n)]  # 도로 정보 입력

result = get_road_move_time(road, n, m)  # BFS를 이용해서 이동시간 구하기
print(result)
```

### 미로같은 문제 : 웬만하면 BFS  // DFS는 모든 경로를 탐색해야 하므로 답이 없

**미로를 DFS로 ? : 목적지까지 도달하는 경로에서 특정 조건 구해야 하는 경우** 

- dfs는 visited를 True로 찍었다가 재귀 돌린 이후에 False 돌려야 다음 dfs 가 갈 수 있음 .. 어렵

```python
# 그렇지만 DFS로 푼다면 ? 
# DFS => 모든 경로를 탐색해야하기 때문에, 미로같은 문제는 BFS 로
# 모든 길이 열려있다 > DFS > O( 2 ^ (N*M) )

**# 미로를 DFS 풀어야할 때?  목적지까지 도달하는 "경로" 에서 특정 조건 구해야 하는 경우**

def dfs(x, y, count):
    global min_count

    if x == N-1 and y == M-1:
        min_count = min(min_count, count)
        return

    dxy = [(1,0), (0,1), (-1,0), (0,-1)]
    for dx, dy in dxy:
        nx, ny = x +dx, y + dy

        if nx < 0 or nx >= N or ny < 0 or ny >= M: continue

        if visited[nx][ny]: continue

        if not road[nx][ny]: continue

        visited[nx][ny] = True
        dfs(nx, ny, count + 1)
        visited[nx][ny] = False

# 입력 받기
N, M = map(int, input().split())
road = [list(map(int, input())) for _ in range(N)]

# 방문 여부를 확인하기 위한 변수
visited = [[False] * M for _ in range(N)]
# 최소 이동 횟수를 저장하기 위한 변수
# min_count = float('inf')
min_count = N*M
visited[0][0] = True

# dfs 파라미터 뭐로 줘야해요 ?
# 1. 종료 조건이 될 수 있는 변수 ( 이동 좌표 => 목적지에 도착하면 DFS 중단 )
# 2. 결국 얻으려는 누적값 ( 이동 거리 )
dfs(0, 0, 0)

```

---

### 섬 찾기

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/70154729-bff3-41d2-8cc9-5f6b176201d6/Untitled.png)

```python
# 모든 노드에서 bfs 실행하는 예시 
# idea : 탐색 완료하면 0으로 

from collections import deque
import sys
sys.stdin = open('test.txt')

def find_island(island, x, y):
    # 상하좌우 + 대각선.. 
    dxy = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1, -1), (1,1), (-1,1)]

    queue = deque([(x,y)])
    island[x][y] = 0 # 바다로 바꿈 / 방문처리 

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in dxy:
            nx, ny = dx+cx, dy+cy

            if nx < 0 or nx >= n or ny < 0 or ny >= m: continue# n,m 바뀐거 아닌가 ? 

            if island[nx][ny] == 0: continue# 바다면 무시

            queue.append((nx, ny))
            island[nx][ny] = 0

    return -1 
            

n, m = map(int, input().split()) # 세로 / 가로 
arr = [list(map(int, input().split())) for _ in range(n)] 
island_cnt = 0

# 이런 묶음 구하는 문제? 탐색하는 범위를 미리 구분해준다 ? 
for i in range(n):
    for j in range(m):
        if arr[i][j] == 1: # 땅이면 bfs 실행
            find_island(arr,i,j)
            island_cnt += 1 

print(island_cnt)
```

```python
# dfs 코드 

import sys

sys.stdin = open('island_input.txt')

# 어떤 걸로 풀어도 상관이 없다
def dfs(x, y):
    dxy = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    island[x][y] = 0

    for dx, dy in dxy:
        nx, ny = x + dx, y + dy

        # 범위를 벗어난 경우
        if nx < 0 or nx >= n or ny < 0 or ny >= m:
            continue
        # 방문 못하는 곳 ( 바다)
        if not island[nx][ny]:
            continue

        dfs(nx, ny)

n, m = map(int, input().split())  # 섬의 크기 입력
island = [list(map(int, input())) for _ in range(n)]  # 섬의 상태 입력 받기
island_cnt = 0  # 섬의 개수

for i in range(n):
    for j in range(m):
        if island[i][j] == 1:  # 방문하려는 곳이 땅이여야 한다.
            dfs(i, j)
            island_cnt += 1

print(island_cnt)  # 섬의 개수 출력
```

---

# Heap : 우선순위 큐 쓰기 위한 자구

- **완전 이진 트리**의 노드 중, **키값이 가장 큰 노드나 가장 작은 노드**를 찾기 위해 만든 자료구조

- **최대 힙** : 
- 키 값이 가장 큰 노드 찾기 위한 **완전 이진 트리** : 마지막 레벨 제외하고 모든 레벨 채워짐
- **부모 노드의 키 값 > 자식 노드의 키 값**
- **루트 노드 : 키 값이 가장 큰**  노드

- 최소 힙 : 최대힙의 반대 ( 부모 < 자식 / 루트가 가장 작다 )

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f6efa614-3c9a-40ca-a74f-0c47170168f8/Untitled.png)

- 힙이 아닌 이진 트리 ?

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c32173b2-80ed-4e03-ac44-f35f4448fb86/Untitled.png)

1. 완전 이진 트리가 아님                                                          2. 최소도 최대도 아님 

## 힙 연산 : 삽입 logN

- 마지막 리프노드에 삽입 O(1)
- 부모와 비교하면서 올라간다 **O(logN) : 트리의 높이 : logN / 부모만큼만 비교 ( * 1 )**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9381af2d-1958-4dd8-af40-ada979474e82/Untitled.png)

- 완전 이진 트리 : 트리의 가장 마지막 위치에 삽입됨 ( 우선 )
      ㄴ 이걸 위해서 리스트 확장(append) : O(1) 
      ㄴ 최대 힙인데 자리에 틀림 없으니 그대로

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9d93c80f-972c-459b-845d-731156f6d4a9/Untitled.png)

- 최대 힙인데 삽입 시 부모보다 내가 크다 ? > 자리 바꿈
- 23의 부모와 비교 > 크니까 바꿈
- 20의 부모 ?  >  20이 루트므로 끝

---

## 힙 연산 : 삭제  : logN

- **힙에서는 루트 노드의 원소만을 삭제할 수 있다**
- 루트 노드의 원소 삭제하여 **반환**한다 > 힙의 종류에 따라 최대값 또는 최소값 구할 수 있다

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e6498320-441c-43df-9fcc-dc710958449a/Untitled.png)

1. 루트의 원소값 삭제 
2. 제일 마지막 노드 삭제 
3. 해당 노드 루트로 옮기기
4. 자기 자리 찾아가기 ( 자식 중 큰 / 작은 친구와 변경 )

## heap Q 코드 구현 :

```python
class MaxHeap:
    def __init__(self):  # 힙을 저장할 빈 리스트 조기화
        self.heap = []

    def heappush(self, item):  # [데이터 삽입] 힙에 새로운 요소 리스트에 추가
        self.heap.append(item)  # 완전 이진 트리의 맨 뒤에 삽입

        self._siftup(len(self.heap) - 1)
        # 힙 속성 유지하도록 sift-up 연산 수행 ( 데이터를 부모와 비교하며 올림 )
        # 넘겨주는 요소 : 마지막 원소의 idx

    # 삽입 후 힙 속성 유지 위해 사용되는 보조 메서드
    def _siftup(self, idx):
        parent = (idx - 1) // 2  # 부모 노드의 idx 계산 << 완전 이진트리 특징
        while idx > 0 and self.heap[idx] > self.heap[parent]:
            # 루트가 아니고, 내가 부모보다 크면 ? swap
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            # 내가 부모의 idx 물려받고,
            idx = parent
            # 또 바꿀지 말지 보기 위해 부모노드 확인
            parent = (idx - 1) // 2

    def heappop(self):  # 힙에서 최대 요소 제거하고 반환
        if len(self.heap) == 0:raise IndexError("힙이 비었습니다.")  # 힙이 비어있는 경우 예외 발생
        if len(self.heap) == 1:return self.heap.pop()             # 힙에 요소가 하나만 있으면 그 요소 반환

        root = self.heap[0]  # 최대, 최소가 루트 노드에 있다 > 해당 요소 밑에서 return
        self.heap[0] = self.heap.pop()  # 삭제 시 마지막에 있던 노드 루트 노드로
        self._siftdown(0)  # 쭉 내려가면서 힙 속성 맞게 유기
        return root

    # 자식들이랑 비교하면서 내려가는
    def _siftdown(self, idx):
        n = len(self.heap)
        largest = idx  # 본인
        left = 2 * idx + 1  # 완전 이진 트리 특징 이용해서 왼오 idx
        right = 2 * idx + 2

        # 왼쪽, 오른쪽 자식이 나보다 크다 > update
        if left < n and self.heap[left] > self.heap[largest]:largest = left
        if right < n and self.heap[right] > self.heap[largest]:largest = right

        # 자식중 누군가가 나보다 크다 ? 교환 > 자식 노드에 대해 sift-down 연산 수행
        if largest != idx:
            self.heap[idx], self.heap[largest] = self.heap[largest], self.heap[idx]
            self._siftdown(largest)

    # 기존 리스트를 힙 구조로 : 시간복잡도 N 
    def heapify(self, array):
        self.heap = array[:]  # 리스트의 복사본을 힙으로 활용
        n = len(array)

        # 리스트의 중간부터 시작해 모든 노드에 대해 sift-down 연산 수행???????????
        for i in range(n // 2 - 1, -1, -1):self._siftdown(i)

    def __str__(self):
        return str(self.heap)  # 힙의 문자열 표현 반환
```

## heap Q 모듈 : 모의 가능!!!!!!!!!!!

- **최소 힙**을 구현한 라이브러리
- 힙 함수 활용 : 
- heapq.headpush(heap, item) : item 을 heap 에 추가
- heapq.headpop(heap)            : heap 에서 루트 노드 pop 하고 리턴
- Heapq.heapify(x) : 리스트 x를 heap 으로 변환 ( O(N) )

```python
import heapq

numbers = [10, 1, 5, 3, 8, 7, 4]

heapq.heapify(numbers) # 리스트를 최소 힙으로 변환
heapq.heappush(numbers, -1) # 힙에 새로운 요소 추가

smallest = heapq.heappop(numbers) # 힙에서 가장 작은 요소 제거하고 반환 
```

- **최대힙 ? : 모든 값을 음수로 바꿔서 최대 힙처럼 사용**

```python
import heapq

numbers = [10, 1, 5, 3, 8, 7, 4]

# 음수로 변환해서 최대 힙으로 구현 
max_heap = []
for number in numbers:
    heapq.heappush(max_heap, -number) # 음수로 변환한 값 힙에 추가
print(max_heap)

largest = -heapq.heappop(max_heap) # 가장 큰 값 반환 
```

## 우선순위 큐 : 힙 통해 구현

- 기존 큐와 다르게 들어온 순서에 상관없이 우선순위 부여 가능
 > FIFO 순서 아니라 우선순위 높은 순서대로 나감
- 운영체제 스케쥴링 / 프로세스 우선순위 할당 / 그래프의 최단거리 / 문자열 압축 ( 허프만 코딩 )
- **노드 하나의 추가, 삭제의 시간 복잡도 : logN / 최대, 최소값 구하기 : O(1)**

```python
# 소스 가져오기 
# heapq의 우선순위 큐 활용 참고 사이트
# https://docs.python.org/ko/3/library/heapq.html#priority-queue-implementation-notes

import heapq

# 빈 우선순위 큐 생성
priority_queue = []

# 우선순위 큐에 요소 추가 (우선순위, 작업)
heapq.heappush(priority_queue, (3, "3 priority task"))  # 우선순위 3인 작업 추가
heapq.heappush(priority_queue, (1, "1 priority task"))  # 우선순위 1인 작업 추가
heapq.heappush(priority_queue, (2, "2 priority task"))  # 우선순위 2인 작업 추가

# 현재 우선순위 큐의 상태 출력
print(priority_queue)  
# [(1, '1 priority task'), (3, '3 priority task'), (2, '2 priority task')]
# 우선순위가 낮은 숫자일수록 더 높은 우선순위를 가짐

# 우선순위 큐에서 요소를 하나씩 꺼내어 출력
while priority_queue:
    task = heapq.heappop(priority_queue)  # 우선순위가 가장 높은 요소를 꺼냄
    print(task)  # 꺼낸 요소 출력

```

### 진용이네 주차타워 :

- https://swexpertacademy.com/main/code/problem/problemDetail.do?contestProbId=AW9j74FacD0DFAUY
- 차량을 입구에서 **기다리게** 한다.                                          == **큐를 써라**
- 차례 오면 번호가 **가장 작은 주차 공간**에 주차하도록 한다. == **우선순위 큐**

```python
# n = 2  # 주차공간
# m = 4  # 차량 대수
# 차량 무게 [ 100, 500, 1000, 2000 ]
# 출입 순서 : [3,1,2,4,-1,-3,-2,-4]
# 주차 공간 : 비용 [5, 2] n == 2라서 2개 // idx 작으면 먼저 배차 > 우선순위 큐 > 주차 시 PQ 에서 제거
# 차량 대기소 deque

"""
1. 주차공간 O : 가장 번호 낮은곳에 주차
2. 주차공간 X : 빈 주차공간 생길 때까지 대기
                                                               { 차 번호 : PQ IDX }
3 : 3번차 주차 : PQ 0에 들어감 / 비용 1000 * 5 = 5000 { 3 : 0 }
1 : 1번차 주차 : PQ 1에 들어감 / 비용 100 * 2 = 200   { 1 : 1 } 
2 : 2번차 주차 : PQ 꽉참 > DQ에 넣음 / 
4 : 4번차 주차 : PQ 꽉참 > DQ에 넣음 / 
-1 : 1번차 출차 : PQ 1에서 나옴 / DQ popleft 2 > PQ 1에 들어감 / 비용 500 * 2 = 1000 { 2 : 1 }
-3 : 3번차 출차 : PQ 0에서 나옴 / DQ popleft 4 > PQ 0에 들어감 / 비용 2000 * 5 = 10000 { 4 : 0 } 
"""
import heapq
from collections import deque

def start_parking(n, costs, weights, entry_order):
    total_income = 0

    waiting_queue = deque()
    free_spaces = list(range(n)) # 주차공간 접근 위한 idx list 생성
    heapq.heapify(free_spaces) #

    # 차가 놓여있는 공간 알기 위한 변수가 필요. 출차할 시에 차가 어떤 주차공간에 주차되어있는지를 알아야 하니까
    car_positions = {}

    for entry_car in entry_order:
        if entry_car > 0 : # 주차
            if len(free_spaces) > 0: # 주차공간 남아 있으면
                # 차 번호가 1 ~ 5 형태로 오고 있는데, 무게 접근 시에는 idx 로 해야 함
                car_number = entry_car - 1
                space_index = heapq.heappop(free_spaces)
                car_positions[car_number] = space_index
                total_income += costs[space_index] * weights[space_index]
            else: # 주차 공간 없으면
                waiting_queue.append(car_number)
        else :  # 출차
            car_number = entry_car * (-1) - 1  # 차의 idx를 원하는데, -3 과 같은 식으로 출차 시에는 오므로 -1 *
            space_index = car_positions[car_number]
            # 주차 공간이 비었으므로 heap 에 push
            heapq.heappush(free_spaces, space_index)

            if waiting_queue:
                waiting_car = waiting_queue.popleft()
                space_index = heapq.heappop(free_spaces)
                car_positions[waiting_car] = space_index
                total_income += costs[space_index] * weights[space_index]

    return total_income

T = int(input().strip())
for tc in range( 1, T + 1 ):
    n, m = list(map(int, input().split()))
    costs = [int(input()) for _ in range(n)]
    weights = [int(input()) for _ in range(m)]
    entry_order = [int(input()) for _ in range(2 * m )] # 주차 입. 출차 순서 목록

    result = start_parking(n, costs, weights, entry_order)

    print(f'#{tc} {result}')
```

---

# 백트래킹 : DFS의 가지치기

- 여러가지 선택지 ( 옵션 ) 존재하는 상황에서 한가지를 선택한다
- 선택 이루어지면 선택지들의 새로운 집합 생성된다
- 이런 선택 계속 하면, 목표 상태에 도달한다

- 당첨 리프 노드 찾기 : 이거는 DFS 
- 루트에서 갈 수 있는 노드 선택
- 꽝 노드까지 도달하면, 최근의 선택으로 되돌아와 다시 시작
- 더 이상의 선택지가 없다면, 이전의 선택지로 돌아가 다른 선택
- 루트까지 돌아갔을 경우 더 이상의 선택지 없다면, 찾는 답 X
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3ae6de4e-029b-4cab-9b77-eab9016f4756/Untitled.png)
    

- 백트래킹 VS DFS  : 
- 어떤 노드에서 출발하는 경로가 해결책으로 이어질 것 같지 않으면, 더 이상 그 경로를 따라가지 않음으로써 시도의 횟수 줄임 ( Punning 가지치기 ) 
- DFS는 모든 경로 추적하는데 비해 백트래킹은 불필요한 경로 조기에 차단 
- 백트래킹 알고리즘 적용하면, 일반적으로 경우의 수가 줄어들지만 이 역시 최악의 경우에는 여전히 지수함수 시간을 요하므로 처리 불가능

### 백트래킹 절차 :

- DFS > 노드 유망성 점검 > 유망하지 않으면 부모로 돌아가 다른 노드로의 검색 계속

## N-Queen

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a8347dbc-dd85-4f42-9b89-146d24d9be36/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dc07513b-b64b-4386-bb6f-bd6a65c319c2/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d0bd058a-ad59-4878-992b-0285ad4e2fe0/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/94d52fdb-613c-409a-aa2b-f12479cc2b57/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0388861a-54dc-453e-90f0-3378646bcb17/Untitled.png)

```python

# 현재 위치에 퀸을 놓을 수 잇는 검사하는 함수
def is_valid_pos(board, row, col):
    # 현재 열에 다른 퀸이 있는 지 검사
    for i in range(row):
        if board[i][col] == 1:
            return False

    # 대각선 검사를 해야한다.
    """
    row = col = 2 
    range(2, -1, -1 ) => [2, 1, 0]
    => for i in zip([2, 1, 0], [2, 1, 0]) 
    zip 함수 뭐죠 ? 리스트 2개를 병렬로 묶는다. 
    [2,2], [1,1], [0, 0]
    """
    # 왼쪽 윗 대각선 검사
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False
    # 오른쪽 윗 대각선 검사
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

# row: 현재 퀸을 놓을 행
# board: 퀸들의 위치를 나타내는 n*n 보드
def n_queens(row, board):
    if row == n:
        solutions.append([r[:] for r in board])  # deepcopy 를 써도됨
        return

    for col in range(n):  # 현재 row 행에서 각 열에 대해 퀸을 놓을 수 있는 검사를 한다.
        if is_valid_pos(board, row, col):
            # 만약에 놓을 수 있따면 ?
            board[row][col] = 1  # 현재 위치에 퀸을 놓음
            n_queens(row + 1, board)  # 다음 행으로 이동하여 재귀호출
            board[row][col] = 0  # 없었떤 일로 만든다.

n = 4  # 4개의 퀸을 놓자 !
board = [[0] * n for _ in range(n)]  # 4*4 2차원 배열 생성
solutions = []  # 모든 솔루션을 저장할 리스트

# dfs 다!
# 호출을 중단시킬 파라미터 => 각 행에 퀸을 놓을 수 있는 지 확인하고, 마지막 행까지 간다면 전부 놓은 것
# 우리가 원하는 누적값 : 마지막 행까지 놓였을 때의 체스판 저장 상태를 원한다.
n_queens(0, board)

for solution in solutions:
    print(solution)

```

## 연습문제 - 부분 집합의 합

```python
"""
# start: 탐색 시작 지점 인덱스
# current_subset: 현재까지 선택된 부분집합
# current_sum: 현재까지 선택된 원소 합
# result: 조건을 만족하는 부분집합을 저장할 리스트
"""
def find_subsets(start, current_subset, current_sum):
    # 현재 부분집합의 합이 target_sum과 일치하면 result에 추가
    if current_sum == target_sum:
        result.append(current_subset[:])
        return
    
    # 현재 부분집합의 합이 target_sum을 초과하면 백트래킹
    # 백트래킹을 하지 않았을 때는 지수 시간복잡도를 갖는다.
    if current_sum > target_sum:
        return

    # start부터 nums의 끝까지 탐색
    for i in range(start, len(nums)):
        num = nums[i]
        # 현재 수를 선택한 경우
        current_subset.append(num)
        find_subsets(i + 1, current_subset, current_sum + num)
        # 현재 수를 제외하고, 다른 수를 선택한 경우
        current_subset.pop()

nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
target_sum = 10
result = []
# DFS 에서 필요한 파라미터
# 1. 재귀를 중단시킬 파라미터 (current_sum => 원소의 합이 10이되면 멈춰야함)
# 2. 누적해서 가져가야할 파라미터 ( current_subset => 생성된 부분집합을 누적으로 가져가서 결과에 저장해야 함)
# + 현재 선택할 원소를 가리키는 idx 파라미터
find_subsets(start=0, current_subset=[], current_sum=0)

print(result)
```

---

# 그리디 알고리즘 :

## 개요

- 최적해를 반드시 구한다는 보장이 없다

800원을 500 / 400 / 100 / … 원으로 거슬러 줄 때에, 
큰거부터 뺴서 구한다고 하면 500 + 100 * 3 : 4개의 동전
BUT 최적해? 400 * 2 : 2개의 동전 
>> 탐욕적 접근이 최적해가 아니다 
>> 안되는 이유? **동전들 사이에 배수관계가 아니기 떄문에 ( 400원 <> 500원 )**
- 일반적으로, 최적해 구하는 데에 사용하는 근시안적 방법 / 머리 속에 떠오르는 생각 검증X 구현
- 여러 경우 중 하나 선택할 때마다 그 순간에 최적이라고 생각되는 것을 선택해 나가는 방식
- 한번 선택된 것은 번복 X

- 그런데 왜씀? : 완탐(DFS, BFS, 순열, 조합 … )이 무조건 해를 찾을 수 있지만 시간이 많이 걸린다
                        그럴 때에 그리디로 해결

---

## 활용 1 : Knapsack < 부분집합 ( 모든 경우 체크 )  / 그리디

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d2e7e2ba-827d-4a8c-8903-910347cbf481/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4c841895-6d51-4337-97c3-25a80567aa4e/Untitled.png)

- 문제 정의 : 무게 합 < 배낭 / 값의 합 : 최대 / S의 부분집합

### Knapsack 문제 유형 :

1. 0-1 knapsack : 배낭에 물건 통째로 담아야 하는 문제  / **[ 넣던가 안넣던가 ]**
                        물건 **쪼갤 수 없는** 경우

1. 완탐 : 부분집합
- 완전 검색으로 물건들의 집합 S에 대한 모든 **부분집합**을 구한다
- 부분집합의 총 무게가 W를 초과하는 집합들은 버리고, 
   나머지 집합에서 총 값이 가장 큰 집합을 선택할 수 있다. 
- 물건의 개수가 증가하면 시간 복잡도가 지수적으로 증가한다 : 크기 N인 부분집합의 수 : **2^n**

2. 그리디
- 0-1 knapsack에 대한 탐욕적 방법 : 

값이 비싼 물건부터 채운다  >  1번째 경우면 최적해 X

무게가 가벼운 물건부터 채운다 > 2번째 경우면 최적해 X 

무게당 값 높은 순서로 물건 채운다 > 1, 3순서대로 받아오는데 / 2, 3이 더 많이 줌 > 최적해 X 

>> 결론 : 탐욕으로 불가 / DP로 풀어야 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/34db7558-c33f-4fa8-bd6c-25ba7b0f04b2/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/181c7e0f-58d1-4773-8d41-3825e389e7fb/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4085846e-c453-43de-93c0-ce15b521c275/Untitled.png)

1. Factional Knapsack : 부분적으로 물건을 담는 것이 허용되는 문제
                                 물건 **쪼갤 수 있는** 경우

- 이럴 때는 위의 그리디에서 3번째 방식으로 진행할 수 있음 / 쪼갤수 있으니까 위에서부터

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/93ede524-ac3d-4d36-83a2-c178be10d9b1/Untitled.png)

---

## 활용 2 : 활동 선택 문제

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/45d1c8fa-7512-41b2-a627-445452aff5af/Untitled.png)

- IDEA : 먼저 시작하는 순 / 회의가 짧은 순 / 먼저 끝나는 순 ? 

ANS : 먼저 끝나는 순 > 많은 회의가 열려야 하는걸 원하기 때문에

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c4477b87-89ee-4070-9470-a9a48dfaaee0/Untitled.png)

- 탐욕 기법 적용한 반복 알고리즘 : 
- **종료 시간 빠른 순서**대로 활동들을 **정렬**한다
- 첫 번째 활동을 선택한다 
- 선택한 활동의 종료 시간보다 빠른 시작 시간을 가지는 활동은 무시하며, 
   같거나 늦은 시작시간을 갖는 활동을 선택한다
- 선택된 활동의 종료시간을 기준으로, 뒤에 남은 활동들에 대해 앞의 과정을 반복한다

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e25e6bc7-6a97-4517-b21f-fc14d20c7e09/Untitled.png)

---

## 탐욕 알고리즘의 필수 요소 : 이론적인 내용

- 탐욕적 선택 속성 : 탐욕적 선택은 최적해로 갈 수 있음을 보여라 > 탐욕적 선택은 항상 안전하다
- 최적 부분 구조 : 최적화 문제를 정형화하라 : 하나의 선택하면, 풀어야할 하나 하위 문제 남는다
- **원 문제의 최적해 = 탐욕적 선택 + 하위 문제의 최적해** 임을 증명하라

## 대표적인 탐욕 기법 알고리즘 : 그래프 이론

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/809fb808-2f2c-4ccd-beef-ddf8383c4cce/Untitled.png)

- + 허프만 코딩 ( 문자열 압축 )
- 한달 뒤에 ,,

---

# 문제풀이 < 구현

## 원자 소멸 시뮬레이션

```python
"""

1. 각 좌표에 모인 원자들을 저장 ( 주어진 좌표 ) {(1,1) : [(1,10), (2,20)] 느낌으로 중복키 들어오면 val 에 append
2. 모인 원자 수가 2 이상이면 (dict 키값의 val len >= 2 ) 에너지 방출
3. 단일 원자 : 방향에 따라 이동 { 임시공간 : 이동 후 위치에 따른 원자 정보 저장
4. 임시공간을 원본에 복사

Q1 : 위치한 곳이 정수가 아니라 0.5 ? > 이동을 0.5씩  /  좌표계를 2배로 늘린다  -  1씩 이동 >> Q2에서 적은 LIMIT 도 늘려야
Q2 : 좌표 범위에 제한이 없음 > 원자 범위 -1000 ~ 1000 - 이거 벗어나면 충돌 XXX > 이동 후 범위를 넘었으면 임시공간 저장 X

"""
import sys
sys.stdin = open("sample_input.txt", 'r')

# 0.5씩 이동하는 ver

# dir : 0123 상하좌우
## 이문제같은 경우에는 xy 를 dfs bfs 느낌으로 행렬같이 안쓰고, 좌표로 보자
dxy = [(0,0.5), (0,-0.5), (-0.5,0), (0.5,0)]
 
    N = int(input())
    boom_sum = 0
    atom_dict = {}

    # 입력값 [ x, y, dir, K ] //
    for _ in range(N):
        x, y, direction, energy = list(map(int, input().split()))

        if (x, y) not in atom_dict:
            atom_dict[(x, y)] = []
        atom_dict[(x, y)].append((direction, energy))

        ### (x, y) 키값 dict에 들어가 있지 않으면, []로 초기화 / 들어가 있으면 append
        ### atom_dict.setdefault((x,y), []).append((direction, energy)) # 위의 세줄을 한줄로!

    while atom_dict:
        tmp_dict = {}

        # dict items를 키값 cx, cy / val atoms 로 받는 법...
        for (cx, cy), atoms in atom_dict.items():
            if len(atoms) >= 2:
                boom_sum += sum(k for d, k in atoms) # 와우
                continue
            # 단일 객체 : 임시 dict 에 in
            d, k = atoms[0]  # 여기는 원자가 1개인 경우 : 1번쨰 원소 가져오면 다 가져온 것.
            # atoms 가 for 통해서 돌아가고 있으니까 처리를 다 해주네 ㄷㄷㄷ..

            dx, dy = dxy[d]  # 움직여야 하는 방향 받아옴
            nx, ny = cx + dx, cy + dy

            # 범위 벗어난 경우에는 임시 dict에 추가 X
            if nx <= -1000 or nx >= 1000 or ny <= -1000 or ny >= 1000: continue

            # 문제없는 값들 이동 좌표에 따라 임시공간에 저장하자
            tmp_dict.setdefault((nx, ny), []).append(d, k)

        # 충돌한 원자 임시공간 저장 x / 단일 객체면 범위 거르고, 이동한거 저장해주고
        # 기존 dict에 옮기며 작업 반복 
        atom_dict = tmp_dict

    print(f'#{tc} {boom_sum}')
```

---

## 요리사

```python

"""
2명의 손님에게 음식 제공   >    n개의 식재료  >  식재로 n/2 하여  A, B 2개의 요리
A, B의 맛의 차이가 최소 되도록 재료 배분
식재료 I, J 요리 : 시너지 Sij
각 음식의 맛 : 식재료들로부터 발생하는 시너지 Sij 의 합
맛의 차이중 최솟값 출력

>>>
Sij != Sji
A 가 ij 가져가면 / B는 자동적으로 나머지 2개 재료 k, l 가져가야 함
Sij + Sji - Skl + Slk 가 최소가 되게

4개일 때에
A를 고르는 조합 : N C N/2
B

6개 이상 :
A 가 N C N/2 해서 나온 123 의 시너지 ? << 3 C 2 해서 12 13 23 의 시너지의 합   << 이게 어렵다
"""
import sys
sys.stdin = open("sample_input.txt", 'r')

import itertools
T = int(input())

for tc in range(1,T+1):
    N = int(input())
    synergy = [list(map(int , input().split())) for _ in range(N)]
    half = N // 2

    # 식재료를 인덱스로 표현할 것 / 식재료의 개수만큼 순차적으로 idx 만들 것
    num_list = [i for i in range(N)]

    # A, B 요리는 식재료를 절반씩 가져간다. > 경우의 수 ? n C n/2
    food_comb_list = itertools.combinations(num_list, half)
    res = float('INF') # 맛 차이 저장변수 / 최솟값으로 갱신 예정

    for a_food_list in food_comb_list:
        b_food_list = []

        # a 요리에 속하지 않은 레시피를 b 목록에 추가
        for num in num_list:
            if num not in a_food_list:
                b_food_list.append(num)
        # comp 이런 식으로도 가능하다 ... ...
        # b_food_list = [num for num in num_list if num not in a_food_list]

        # 이런 식으로도 할 수 있다
        # def get_synergy_sum(food_list, synergy):
        #     synergy_pairs = itertools.combinations(food_list, 2)
        #     synergy_sum = 0
        #     for synergy in synergy_pairs:  # 뭐가 들어오든 이미 2개로 짤라놨음 ij . ji 다르다는거 생각
        #         i, j = synergy
        #         synergy_sum += synergy[i][j] + synergy[j][i]
        #     return synergy_sum
        # a_synergy_sum = get_synergy_sum(a_food_list, synergy)
        # b_synergy_sum = get_synergy_sum(b_food_list, synergy)

        # 시너지 구할 떄에, 2개보다 큰 경우에는 또 조합을 찾아서 더해야 함
        a_synergy_list = itertools.combinations(a_food_list, 2)
        a_synergy_sum = 0
        for a_synergy in a_synergy_list: # 뭐가 들어오든 이미 2개로 짤라놨음 ij . ji 다르다는거 생각
            i, j = a_synergy
            a_synergy_sum += synergy[i][j] + synergy[j][i]

        b_synergy_list = itertools.combinations(b_food_list, 2)
        b_synergy_sum = 0
        for b_synergy in b_synergy_list:  # 뭐가 들어오든 이미 2개로 짤라놨음 ij . ji 다르다는거 생각
            i, j = b_synergy
            b_synergy_sum += synergy[i][j] + synergy[j][i]

        res = min(res, abs(a_synergy_sum - b_synergy_sum))

    print(f'#{tc} {res}')
```

---

>>> 과목평가 

# 서로소 집합

- 그래프 알고리즘에서 **사이클 검출 도움
사이클 있으면?** → 무한루프 가능 ( 로직 추가해야되므로 복잡도 + ) 
                              최단경로 찾는데 / 위상정렬 하는데에  어려움

## 개요

- 서로소 집합 : 자료구조 ( Disjoint-set ) 
서로소 및 상호배타 집합들은 서로 중복 포함된 원소 없는 집합들
두 집합 간에 공통 원소가 하나도 없을 때에, 이를 서로소 집합이라 함
집합에 속한 하나의 특정 멤버를 통해 각 집합들을 구분 / 이를 대표자라 함
- 목표 : 
1. 여러개의 겹치지 않는 집합 관리 → 서로 다른 노드가 같은 그룹에 있는지 확인 가능 
2. 아래 그림과 같이 사이클 검출 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e26e43f0-f387-406a-8434-b6da81925ad6/image.png)

## 서로소 집합 표현

1, 연결 리스트 : 잘 사용 x 
같은 집합의 원소들은 하나의 연결 리스트로 관리
연결 리스트의 맨 앞의 원소를 대표자로
각 원소는 집합의 대표자를 가리키는 링크 갖는다 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/45c872cb-3df9-41e8-a81d-1b56bb517dfa/image.png)

1. **트리 : 메인**으로 사용 
같은 집합의 원소들은 하나의 트리로 표현
**자식 노드가 부모 노드 가리키며 루트 노드가 대표자**가 된다 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/832ff5d6-a115-49b6-a79c-c3c62738033c/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/37cca4a9-a601-4714-946c-c32f4e7d9a30/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a2780bf7-df4c-44d3-828c-a9ba6b653b59/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9850a082-d277-4402-a4f8-68691639691a/image.png)

- **본인 idx == 부모 idx → 그 그룹의 대표자**다
- a, f를 비교할 때에 대표자가 다르므로 같은 그룹이 아니고, 
그렇기 때문에 연결해도 사이클이 생기지 않는다 ㄷㄷㄷ

- Find) 대표자 찾아가는 연산의 빅오 : O(N)
- Union) 대표자가 대표자에 붙는 연산이므로, 대표자를 찾는 작업 O(N) + @ 

뒤에서 최적화 진행하면 둘다 O(1) 될 것

## 서로소 집합 연산

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7ae30d94-ce13-450b-a6f4-4d044d550731/image.png)

1. Make-Set(x) : 스스로가 집합 ( 대표자 ) 이 된다 
2. Find-Set(x) : 해당 그룹의 **대표자를 반환**한다 
3. Union(x, y) : 두 집합을 통합한다 → **2개중 대표자 : 개발자 마음** 

```python
p = [0] * (N + 1)  # 0번째 idx 비워둠

def make_set(x):
	p[x] = x  # 스스로 대표자가 된다 
	
def find_set(x):           # x를 포함하는 집합 찾는 연산 
                           # x : 찾으려는 값 / p[x]: 부모 인덱스의 값 
	if x == p[x]:  return x  # 대표자다? 반환해줘
	return find_set(p[x])    # 대표자 아니면 재귀로 올라가 
		
def union(x, y):           # x, y 포함하는 두 집합 통합하는 연산
                           # union rule : 작은 놈을 대표자로 설정하겠다 
	px = find_set(x)
	py = find_set(y)
	
	if px < py:		**p[y] = px**  # y의 부모로 x 설정하겠다 
	else:         p[x] = py
```

- 하지만 이렇게 하면, 위에서 얘기한 대로 찾는 데에 O(N)이 소요됨

## 서로소 집합의 최적화

1. **Path Compression : Find-set 최적화** 
Find-set 과정에서 만나는 모든 노드들이 직접 root 가리키도록 포인터 변경해준다 
     ㄴ    특정 노드 → 루트까지의 경로 찾아가며 부모노드 갱신 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/98edccab-b63c-48d0-85c2-00660cbb7251/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/853c2c2e-a6bf-48d0-8004-1252e400b6d5/image.png)

- POINT : **재귀 빠져 나오면서 부모를 대표로 갱신**
- 처음에는 O(N)이지만, 이후는 O(1)
- 서로소 집합은 루트노드만 알면 되는 집합이고, 다른건 의미가 없어서 
이런식으로 구현해도 전혀 문제될게 없음

```python
def find_set(x):
	if x != p[x]: p[x] = find_set(p[x])  # 루트노드 ( 대표자 ) 아니면 올라가 
	return p[x]  # 루트 만나면 루트값으로 갱신... ????????
```

1. **Rank 이용한 Union 최적화**
각 노드는 자신을 루트로 하는 서브트리의 높이를 rank로 지정 
두 집합을 합칠 때, rank가 낮은 집합 → rank가 높은 집합  붙이기 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0db44596-ef4c-40b8-b683-3ccf8e472542/image.png)

- POINT : 각 노드별 랭크 저장할 리스트 있어야 함
- 왼쪽의 2개의 노드 합칠 때에 2 > 1이므로 E가 A의 자식으로 들어가는 모습

- CF> 랭크가 같은 2개? 1개를 올리고 붙이기

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e3e532bc-4595-4678-8baa-576fa2537625/image.png)

```python
p = [0] * (N + 1)
rank = [0] * (N + 1)

def make_set(x):
	p[x] = x
	
def union(x, y)
	px = find_set(x)                  # x, y의 대표자 가져와서 px, py에 저장 
	py = find_set(y)
	
	if px != py:                      # 대표자 다를 시에만 union ( 사이클 방지 ) 
		if rank[px] > rank[py]:p[py] = px         # px가 부모
		elif rank[px] < rank[py]:p[px] = py
		else:                           # 같다면 px 부모노드로 -> rank 하나 증가시켜줌 
			p[py] = px
			rank[px] += 1 
```

- 랭크를 저떄만 올리나 ?

### 서로소 집합 최종 코드 :

```python
N = 10 
p = [0] * (N + 1)  # 부모 노드 리스트 초기화
rank = [0] * (N + 1)  # 랭크 리스트 초기화

def make_set(x):
    p[x] = x  # 각 노드가 자기 자신을 부모로 가지도록 초기화

"""
# 최적화 전 find_set
def find_set(x):
    if x == p[x]:  # 노드 x가 자기 자신을 부모 노드로 가지는 경우 그대로 반환 
        return x
    return find_set(p[x])  # 부모 노드를 재귀적으로 찾고 반환
"""

# 경로 압축을 적용한 find_set
def find_set(x):
    if x != p[x]:  # 노드 x가 자기 자신을 부모로 가지지 않는 경우
        p[x] = find_set(p[x])  # 부모 노드를 재귀적으로 찾고 경로 압축 수행
    return p[x]

"""
# 최적화 전 union 함수
def union(x, y):
    px = find_set(x)  # 노드 x의 대표자(부모) 찾기
    py = find_set(y)  # 노드 y의 대표자(부모) 찾기

    if px < py:  # 값이 더 작은 부모가 큰 부모에게 union
        p[y] = px
    else:
        p[x] = py
"""

# 랭크를 이용한 union 함수
def union(x, y):
    px = find_set(x)  # 노드 x의 대표자(부모) 찾기
    py = find_set(y)  # 노드 y의 대표자(부모) 찾기

    if px != py:  # 두 노드가 서로 다른 집합에 속해 있는 경우에만 합침
        if rank[px] > rank[py]:  # 랭크가 높은 트리에 붙임
            p[py] = px
        elif rank[px] < rank[py]:
            p[px] = py
        else:
            p[py] = px  # 랭크가 같은 경우 하나를 다른 하나의 부모로 설정
            rank[px] += 1  # 랭크 증가

# 초기화 예시
for i in range(1, N + 1):
    make_set(i)

# union 연산 예시
union(1, 2)
union(2, 3)

print(find_set(1))  # 출력: 1
print(find_set(2))  # 출력: 1
```

---

# 최소 신장 트리 : MST

## 개요

- 그래프에서 **최소 비용 문제** : 역량T 2번 
모든 정점을 연결하는 간선들의 가중치 합이 최소가 되는 트리
두 정점 사이의 최소 비용의 경로 찾기
- 신장 트리 : n개의 정점으로 이뤄진 무향 그래프에서 
                  **n개의 정점과 n-1개의 간선**으로 이뤄진 트리

- MST ( Minimum Spanning Tree ) : 
무향 가중치 그래프에서 신장 트리를 구성하는 **간선들의 가중치의 합이 최소**인 신장 트리

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0a547291-a2d3-477b-bf1c-a36767eb0482/image.png)

## **KRUSKAL 알고리즘 → 간선** 선택 / 그리디  

간선 가중치 오름차순 정렬
작은 것부터 선택 ( 사이클 생기면 = 대표자 같으면 X )

- 희소 그래프 ( 간선 적고, 정점 많은 ) 에서 유리

- 작은 간선을 하나씩 선택해서 MST 찾는 알고리즘

1. 최초, 모든 **간선**을 가중치에 따라 **오름차순 정렬**
2. 가중치가 **가장 낮은 간선**부터 **선택** 

선택한 간선의 두 정점에 대해 2가지 진행

1. 두  **대표자 다르다 → 간선을 최소비용 집합에 추가** 
2. 같다 → 사이클이 생성되므로 무시 → **사이클이 생성되면 신장트리 아님** 

1. **N-1개의 간선 선택될 때 까지 2번과정 반복**  → 종료 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/14c96384-a1c1-4630-9f79-0db40e3e5975/image.png)

- 오름차순 하여 5-3 잇는 18 선택 / 이때 5, 3에 해당하는 집합 생김 ( Union )

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cbff7444-d457-489d-971d-6951dd82102f/image.png)

- 1, 2가 이어져 있는데 거기에 6 붙일 때에 ( 오름차순으로 ) 
6의 대표자가 6 / 1, 2의 대표자가 1이므로 다르다 → 사이클 X → 연결 ( 그룹화 )

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dd52c8a2-c161-490c-8861-3f41cacf0d17/image.png)

- 쭉 잇다가 0-1 이어야 하는데, 0의 대표자, 1의 대표자 모두 1 → 사이클 O → 다음꺼 보기

- 정점 수 N - 1 == 간선 수 → 탈출

```python
class DisjointSet:
    def __init__(self, v):
        self.p = [0] * (len(v) + 1)  # 부모 노드 배열 초기화
        self.rank = [0] * (len(v) + 1)  # 랭크 배열 초기화

    def make_set(self, x):
        self.p[x] = x  # 각 노드가 자기 자신을 부모로 가지도록 초기화
        self.rank[x] = 0  # 초기 랭크는 0

    def find_set(self, x):
        if x != self.p[x]:  # 노드 x가 자기 자신을 부모로 가지지 않는 경우
            self.p[x] = self.find_set(self.p[x])  # 부모 노드를 재귀적으로 찾고 경로 압축 수행
        return self.p[x]

    def union(self, x, y):
        px = self.find_set(x)  # 노드 x의 대표자(부모) 찾기
        py = self.find_set(y)  # 노드 y의 대표자(부모) 찾기

        if px != py:
            if self.rank[px] < self.rank[py]:
                self.p[px] = py  # x의 부모를 y의 부모로 설정
            elif self.rank[px] > self.rank[py]:
                self.p[py] = px  # y의 부모를 x의 부모로 설정
            else:
                self.p[py] = px  # y의 부모를 x의 부모로 설정
                self.rank[px] += 1  # x의 랭크를 1 증가

def mst_kruskal(vertices, edges):
    mst = []  # 최소 신장 트리 저장
    n = len(vertices)

    ds = DisjointSet(vertices)

    # 초기화
    for i in range(n + 1):
        ds.make_set(i)

    # 가중치를 기준으로 오름차순 정렬
    edges.sort(key=lambda x: x[2])

    for edge in edges:
        s, e, w = edge  # 시작정점, 도착정점, 가중치
        if ds.find_set(s) != ds.find_set(e):  # 시작정점과 도착정점이 다른 집합에 속한 경우
            ds.union(s, e)  # 두 집합을 합침
            mst.append(edge)  # 현재 간선을 MST에 추가

    return mst

# [시작정점, 도착정점, 가중치]
edges = [[1, 2, 1], [2, 3, 3], [1, 3, 2]]
vertices = [1, 2, 3]  # 정점 집합

result = mst_kruskal(vertices, edges)  # [[1, 2, 1], [1, 3, 2]]
print(result)
```

## **PRIM 알고리즘 - 정점** 선택 / 서로소 집합 X 우선순위큐 O

- 밀집 그래프 ( 정점 적고, 간선 많은 ) 에서 유리

- 하나의 정점에서 연결된 간선들 중 하나씩 선택하며 MST 만들어가는 방식

1. **임의 정점** 하나 선택해서 시작
2. **우선순위 큐** ( 큐앞에 항상 작은값 ) 사용해 **간선의 가중치 가장 작은 간선 선택**
3. 이 간선이 **연결하는 정점이 이미 방문한 정점 X** → 이 간선을 MST에 추가 
                                                                            → 그 정점 방문한 것으로 표시
              VISITED 사용 
4. **우선순위 큐 빌 때 까지 위 과정 반복** 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f283eb96-4993-479c-b1f8-bc03b9bffd35/image.png)

- PQ 쓰므로 어떤 정점을 고르든 상관 없음

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0bbaa5e7-0789-4608-835d-91c1c9ec9f54/image.png)

 

- 0-2 연결 → 2-1연결
- …??????? RRR

```python
import heapq

def prim(vertices, edges):
    mst = []  # 최소 신장 트리 저장
    
    # 인접 리스트 생성
    adj_list = {v: [] for v in vertices}
    for start_v, end_v, w in edges:
        adj_list[start_v].append((end_v, w))
        adj_list[end_v].append((start_v, w))

    visited = set()  # 방문한 노드 집합
    init_vertex = vertices[0]  # 초기 정점
    min_heap = [[w, init_vertex, e] for e, w in adj_list[init_vertex]]  
    # 초기 정점의 모든 간선을 힙에 추가
    heapq.heapify(min_heap)  # 힙으로 변환
    visited.add(init_vertex)  # 초기 정점 방문

    while min_heap:
        weight, start_v, end_v = heapq.heappop(min_heap)
        if end_v in visited: continue  # 이미 방문한 정점이면 건너뜀

        visited.add(end_v)  # 새로운 정점 방문
        mst.append((start_v, end_v, weight))  # 간선을 MST에 추가

        for adj_v, adj_w in adj_list[end_v]:  # 새로 방문한 정점의 모든 인접 간선 처리
            if adj_v in visited: continue  # 이미 방문한 정점은 건너뜀
            heapq.heappush(min_heap, [adj_w, end_v, adj_v])  # 힙에 간선 추가

    return mst

vertices = [1, 2, 3]
edges = [[1, 2, 30], [2, 3, 20], [1, 3, 10]]
mst = prim(vertices, edges)  # [(1, 3, 10), (3, 2, 20)]
print(mst)
```

---

# DP - T 가려서 공부

## 개요

- 점화식을 코드로 구현 ( 다양한 유형 )
- 방식 : 메모이제이션 / Tabulation ( bottom → up )

## 피보나치

- f(0) = 0 / f(1) = 1
- f(n+2) = f(n) + f(n+1)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cb482247-1931-4b20-9e2c-3fbbd63b9c88/image.png)
    
    - 재귀의 단점 : 중복 호출이 많다 ( 계산한거에 대해 ) → 메모이제이션

- 메모이제이션 : DP의 핵심 → O(N)
계산한 키값에 V를 저장해둠 → 들어오면 DICT에서 찾아서 있으면 RETURN 

탑→다운 방식 / 추가 메모리 공간 필요

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c7bf1971-1aaa-4f67-966c-7e5ef6c2b8f8/image.png)

- DP : 그리디와 같이 최적화 문제를 해결하는 알고리즘 ( 어렵 ) 
입력 크기가 작은 부분 문제들을 모두 해결 → 그 해들을 이용해 큰 크기의 부분 문제 해결

- DP 적용 문제 요건 :
1. 중복 부분문제 구조 ( Overlapping subproblems ) → 그리디와의 차이 
- 점화식 사용 
- 해결된 작은 문제들의 해들을 어떤 저장 공간에 저장해야 → 중복 피함 

1. 최적 부분문제 구조 ( Optimal substructure ) → 그리디와 동일한 조건 
- 어떤 문제에 대한 해 최적 → 그 해 구성하는 작은 문제들의 해 최적이어야 

- 적용 안되는거 : 최장경로

- 분할정복 vs DP : 

분할정복 : 하향식
연관 없는 부분 문제로 분할
부분문제 [ 재귀적 해결 / 그들의 해를 결합 ] < 병합, 퀵 정렬 

DP : 상향식 
부분 문제들이 연관 없으면 적용 불가 ( 의존적 관계 = 함축적 순서 존재 ) 
부분 문제들은 더 작은 부분 문제들 공유

- 3단계 DP 적용 접근 방법 :
1. 최적해 구조의 특성 파악하라 ( 문제를 부분 문제로 ) 
2. 최적해의 값을 재귀적으로 정의
3. 상향식 방법으로 최적해의 값을 계산 ( 가장 작은 부분 문제부터 해 구한 뒤 테이블에 저장 ) 
→ 점차 상향식으로 상위 부분 문제의 최적해 구하기 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/eff921c0-6cd2-4ae7-b74b-4d67c97bcceb/image.png)

- 재귀를 사용하지 않지만, 재귀적으로 ( 반복문 )
- O(N)

## 이항계수

- 이항정리  :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7a713692-8643-4e38-a514-c4b354318904/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/18fa87f6-9fd2-4d2f-9407-4287cb39a422/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/efe6762f-afe9-494d-93ce-43493ed80eba/image.png)

- 파스칼의 삼각형 : 처음, 마지막은 무조건 1 / 중간은 바로 위, 왼쪽 위 원소 더하면 끝

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/52e85ad3-ab17-4d4a-8414-8b33a04ac26c/image.png)

- 이 방식은 재귀라 중복이 발생

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3c9b0758-dd48-4d5b-a413-9601e922a0bb/image.png)

- DP : 
i : 행 
k : 찾고자 하는 위치

## 동전 거스름돈 DP → 완탐

- 동전 종류 : 1 / 4 / 6원
- 8원 거슬러주려 할 때에, 최소 몇개의 동전? ‘
- 그리디 : 611 / 최적 : 44 → 1, 4, 6이 배수관계가 아니기 떄문에 그렇다

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9828c5cd-cbc9-49b3-a6d6-9b28c38b11ba/image.png)

- 가장 우측의 2 / 1 / 0 같은 SUBTREE는 중복이 많이 됨 → 이걸 DP 테이블에 저장해 생략

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/39e9be4b-4c94-469c-b561-64e2e5cb703d/image.png)

??????????

## Knapsack

# 최장 증가 수열 : LIS - T X

## 개요 :

## BruteForce

## DP

## 이진탐색

---

# 최단 경로

- 간선의 가중치가 있는 그래프에서 두 정점 사이의 경로들 중, **가중치의 합이 최소**인 경로
- 하나의 **시작 정점 ~> 끝 정점**까지의 최단 경로 : 
- 다익스트라 : 음의 가중치 허용 X 
- 벨만포드 : 음의 가중치 허용 O
- 모든 정점들에 대한 최단 경로 : 플로이드-워셜 알고리즘

## 다익스트라 - 그리디 / 우선순위큐

- 시작 정점 ~> 다른 모든 정점으로의 최단 경로를 구하는 알고리즘
- 시작 정점에서의 **거리가 최소인 정점을 선택**해 나가며 최단 경로를 구하는 방식
- **그리디** 사용 / MST, PRIM 과 유사
- **우선순위 큐** 활용

- 동작 순서 :
1. 시작 정점 ~> 각 정점까지의 최단거리 저장할 리스트 생성 
모든 거리 무한대로 초기화 / 시작 정점의 거리 0으로 설정 
2. **시작 정점 & 거리를 PQ에 삽입**
3. 가장 짧은 거리 가진 정점 추출 / 추출한 정점과 인접한 정점 모두 확인
4. **인접한 정점과의 거리 < 기존에 저장된 거리** → 거리 **갱신하고 PQ 삽입**
5. PQ 빌때까지 3~4 반복 

- EX>

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1850a388-0ae8-4904-95be-4ef06281d096/image.png)

- A를 PQ에 넣음 → A를 추출 → 인접한 정점 접근

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/91bd25a4-3611-482a-b9fa-e97912912bb4/image.png)

- A까지 오는 거리 0 + B까지 오는 거리 3 = 3 < 무한대 → 갱신하고 PQ 삽입 
C도 동일하게 진행
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d6a3eaff-878c-4d7f-81b8-1e670da896ad/image.png)
    
- B를 추출 → B와 인접한 정점 접근 
C : 0 + 3 + 2 == 5              → 갱신, PQ 삽입 X 
D : 0 + 3 + 6 = 9 < 무한대 → 갱신, PQ삽입 O
- C 추출 → C와 인접한 정점 접근 
B : 0 + 5 + 1 > 0 + 3           → 갱신, PQ 삽입 X 
D : 0 + 5 + 4 == 0 + 3 + 6 → 갱신, PQ 삽입 X 
E : 0 + 5 + 6 < 무한대         → 갱신, PQ 삽입 O

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6e2299ff-a3f9-4ea4-8379-90be16dedbb4/image.png)

- D 추출 → D와 인접한 정점 접근 : 
E : 0 + 3 + 6 + 2 == 11     → 갱신, PQ 삽입 X 
F : 0 + 3 + 6 + 3 < 무한대 → 갱신, PQ 삽입 O

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ec738a4a-2a10-4990-9855-2df9b06f0293/image.png)

- E 추출 → E와 인접한 정점 접근 : 
A : 0 + 5 + 6 + 3 > 0  → 갱신, PQ 삽입 X 
F : 0 + 5 + 6 + 6 > 12 → 갱신, PQ 삽입 X

- F 추출 → **연결되어 있는 것 없음 ( 빈 PQ )** → 최단거리 리스트가 정답

```python
import heapq, math
def dijkstra(graph, start):
    distances = {v: math.inf for v in graph}  # 모든 정점의 거리를 무한대로 초기화

    distances[start] = 0  # 시작 정점 거리를 0으로 설정

		# PQ
    min_heap = []
    heapq.heappush(min_heap, [0, start])  # 시작 정점을 힙에 추가
    
    # 우선순위 큐에 아무것도 들어있지 않을 때까지 최단거리 갱신 반복 
    while min_heap:

        # 거리 / 정점 POP
        current_distance, current_vertex = heapq.heappop(min_heap)
        
        # 정점까지의 기존 거리 < 갱신된 거리 -> 넘어감
        if distances[current_vertex] < current_distance: continue
        
				# 갱신 해야 함 
        # 인접노드에 대해 갱신된 거리 < 기존 거리 -> 갱신 & PQ 삽입 
        for adjacent, weight in graph[current_vertex].items():
            
            distance = current_distance + weight  
            # 현재 정점까지의 거리 + 인접 정점까지의 거리
            
            if distance < distances[adjacent]:  # 새로운 거리가 더 짧으면 갱신
                distances[adjacent] = distance
                heapq.heappush(min_heap, [distance, adjacent])  # PQ에 추가

    return distances

graph = {
    'a': {'b': 3, 'c': 5},
    'b': {'c': 2},
    'c': {'b': 1, 'd': 4, 'e': 6},
    'd': {'e': 2, 'f': 3},
    'e': {'f': 6},
    'f': {}
}
start_v = 'a'
res = dijkstra(graph, start_v)
print(res)  # {'a': 0, 'b': 3, 'c': 5, 'd': 9, 'e': 11, 'f': 12}

```

## 벨만 - 포드 - DP / 음수 가중치 간선

- 시작 정점 ~> 다른 모든 정점으로의 최단 경로 구하는 알고리즘
- **음수 가중치 간선** 존재 → 음수 사이클 XXX
- **DP** 사용
- **PQ X**

- 동작 순서
1. 시작 정점 ~> 각 정점 최단거리 리스트 ( 무한대 초기화, 시작정점 0으로)
== 다익스트라
2. 모든 간선 반복해서 검사 / 각 간선 통해 더 짧은 경로 발견되면 업데이트
3. 마지막 정점 제외한 모든 정점에 대해 2번 반복 ( v-1번 탐색 ) 
4. 마지막으로 한번 더 모든 간선 검사하여 거리 갱신되면 음수 사이클 존재 의미 

- 추가 ㄹㄹ

## 플로이드 - 워셜 - DP