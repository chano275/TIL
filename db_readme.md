# DATABASE
# Database :

## 개요

- 의미 : 체계적인 데이터 모음
- Data : 저장이나 처리에 효율적인 형태로 변환된 정보
- 데이터를 저장하고 잘 관리하여 활용할 수 있는 기술이 중요해짐
- DB의 역할 : DB를 저장 ( 구조적 저장 ) 하고 조작 ( CRUD )

## Relation Database ( 관계형 DB ) :

- 의미 :  데이터 간에 관계가 있는 데이터 항목들의 모음
- 테이블 / 행 / 열의 정보를 구조화하는 방식
- 서로 관련된 데이터 포인터를 저장하고, 이에 대한 엑세스를 제공

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/903b3708-182e-4a6c-b939-4c0e018f483b/image.png)

- 관계형 DB에서의 ‘관계’? : 여러 테이블 간의 논리적 연결
- 이 관계로 인해 두 테이블을 사용해 데이터를 다양한 형식으로 조회 가능 
EX> 특정 날짜에 구매한 모든 고객 조회 / 전달에 배송일이 지연된 고객 조회 등

- 예시) 이름 / 청구지 / 주소지만 있는 고객 테이블에서, 고객 데이터간 비교 힘듦 ( 중복 인해 )
→ **각 데이터의 고유한 식별 값을 부여**하는 ( **기본키 / Primary Key** = 주민번호 ) 이용 
→ cf> 학번 / 이메일 등 unique한 속성 또 있다 → 그들은 **후보키** 
     ( 후보키중 1개를 기본키로, 남겨진 후보키들은 **대체키** ) 
→ 해당 값을 위의 img에서 왼쪽 테이블의 ( **외래키** / Foreign Key = 고객 id ) 이용 
     ( 주문 정보에 고객의 **고유한 식별 값 저장하기 ( 후보키 )** )

### 관계형 DB 관련 키워드 :

1. Table ( = Relation = 관계 ) : 데이터를 기록하는 곳  
2. Field (  = Column = Attribute ) : 고유한 데이터 형식 ( 타입 ) 이 지정됨 
3. Record ( = Row = Tuple ) : 구체적인 데이터 값이 저장됨 
4. Database ( = Schema ) : 테이블의 집합 
5. Primary Key ( = 기본키, PK ) : 각 레코드의 고유값 / 관계형 DB에서 **레코드의 식별자**로 활용 
6. Foreign Key ( = 외래키, FK ) : 테이블의 필드 중 다른 테이블의 레코드를 식별 가능한 키 
다른 테이블의 기본 키 참조 / 각 레코드에서 서로 다른 테이블간의 관계 만드는 데에 사용 

- DB 정리 : 
- 테이블은 데이터가 기록되는 곳 
- 테이블에는 행에서 고유하게 식별 가능한 기본 키라는 속성이 있으며 
외래키를 사용해 각 행에서 서로 다른 테이블간의 관계 만들 수 있음 
- 데이터는 기본 키 OR 외래키 통해 결합 ( JOIN ) 될 수 있는 여러 테이블에 걸쳐 구조화 됨

## RDBMS :

- DBMS : Database Management System : DB를 관리하는 SW 프로그램 
- 데이터 저장 및 관리 용이하게 하는 시스템 
- DB, 사용자 간의 인터페이스 역할 
- 사용자가 데이터 구성 / 업데이트 / 모니터링 / 백업 / 복구 등 할 수 있도록 도움

- RDBMS : Relational + DBMS : 관계형 DB를 관리하는 SW 프로그램
- 종류 : SQLite / MySQL / PostgreSQL / Oracle Database

# Database Modeling

- DB 시스템을 구축하기 위한 과정으로, 데이터의 구조 / 관계 / 특성을 결정하는 작업
- 효율성 / 일관성 / 무결성 보장하기 위해 진행하는 중요한 단계

- 무결성 제약조건 : DB의 데이터 일관성을 유지하고, 부적절한 데이터의 삽입 / 수정 / 삭제를 방지하여 데이터의 신뢰성을 보장하는 조건

1. 개체 무결성 ( Entity Integrity ) : 
- 기본 키의 값이 중복되지 않고, NULL값이 허용되지 않는 것을 보장하는 제약조건 
- 각 레코드는 유일한 식별자를 가져야 한다 / 기본 키는 NULL 값을 가질 수 없다 

1. 참조 무결성 ( Referential Integrity ) : 
- 외래키와 기본키간의 관계를 유지하고, 무효한 참조를 방지하는 제약조건 
- 외래키 값은 참조하는 테이블의 기본키 값을 참조하거나, NULL값을 가질 수 있다.
- 존재하지 않는 기본 키는 참조할 수 없다 

1. 도메인 무결성 ( Domain Integrity ) : 유효성 검사 
- 각 속성의 값이 정의된 도메인에 속하는 것을 보장하는 제약 조건 
- 도메인은 속성이 가질 수 있는 값의 범위 / 형식 / 제약 조건 등을 정의해야 한다 
- 속성값이 도메인에 속하지 않는 경우에는 삽입 / 수정을 제한하거나 오류를 발생시킨다 

- 도메인 무결성 위배 예시 : 나이 속성은 0~150 사이의 정수로 제약 안걸면 
~년생, 영어, 스물 다섯, 이립 등 다양한 유형으로 input 들어올 수 있어 관리 힘들어진다. 

## Proceeding with Database modeling :

- DB 모델링 진행 순서 : 
- 요구사항 수집 및 분석 → 개념적 설계 → 논리적 설계 → 물리적 설계

1. 요구사항 수집 및 분석 : 
- 어떤 종류의 데이터 정리하는지 정보 수집하고, 어떤 작업 수행해야 하는지 파악하는 단계 
- 개체 ( Entity ) : 업무에 필요하고 유용한 정보 저장하는 집합적인 것 
( 고객 / 제품 / 사원 / 부서 )
- 속성 ( Attribute ) : 관리하고자 하는 것 / 더 이상 작은 단위로 분리되지 않은 데이터 단위 
( 고객명 / 고객 전화번호 / 상품명 / 상품 가격 )  
- 관계 ( Relationship ) : 개체 사이의 논리적 연관성을 의미하는 것 ( 고객 - 상품, 부서 - 사원 ) 

1. 개념적 설계 : 
- 요구사항 기반으로 DB의 개념적 모델 설계 
- 개체와 관게를 식별 / 개체간의 관계 정의하여 ER 다이어그램 작성 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/41c62bbc-1c14-47f6-abbf-115f8db01929/image.png)

- ERD ( Entity Relationship Diagram ) 표기 방법 :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8c0e3129-43d7-4f50-b3e7-778b814472a5/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8c188a71-6f1a-44c7-814a-4f354e703938/image.png)

1. 논리적 설계 : 
- 개념적 설계 기반으로 DB의 논리적 구조 설계 
- **테이블 / 속성 / 제약 조건** 등과 같은 구체적 DB 개체를 정의 
- 정규화 수행해 데이터의 중복 최소 / 일관성 유지 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/550db198-d7d9-4041-935f-702221995a39/image.png)

1. 물리적 설계 : 
- 논리적 설계 기반으로 DB를 실제 저장 및 운영할 수 있는 형태로 변환하는 단계 
- 테이블의 인덱스 등 물리적 구조와 접근 방식을 결정 
- 보안, 백업 및 복구, 성능 최적화 등을 고려해 DB 설정 

# DB Normalization :

- 의미 : 중복 최소화 / 데이터 일관성&효율성 유지 위해 데이터를 구조화하는 과정
- 목적 : 
- 불필요한 데이터 제거하여 중복을 최소화 하기 위해 
- 각종 이상현상 ( Anomaly ) 방지하기 위해 
- DB 구조를 변경할 때 다시 구조화 해야하는 영역을 최소화
- 이상현상 ( Anomaly ) : 
- DB 잘못 설계했을 때에 발생할 수 있는 불필요한 데이터의 중복으로 인한 부작용 

- 삽입 이상 : 새로운 데이터 삽입 위해 불필요한 데이터도 함께 삽입해야 하는 문제 
- 갱신 이상 : 중복 튜플 데이터 중, 일부만 변경하여 데이터가 불일치 되는 문제 
- 삭제 이상 : 튜플 데이터 삭제하는 경우 반드시 있어야 하는 데이터까지 같이 삭제되는 문제

- DB 정규화의 종류 : 1 ~ 6차 / BCNF 있지만 3NF만 되면 정규화 되었다고 표현

## 1NF ~ 3NF / BCNF

- 1NF : 
- 각 요소의 중복되는 항목은 없어야 한다. 
- 각 속성이 원자적(Atomic)하며, 각 행이 PK 가져야 한다. 
- 전화번호 2개 넣기 위해 고객 번호 ( PK ) 중복 ?   >   **1NF X** 
- 테이블 쪼개서 사용

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1bcad97e-b192-42bc-a13c-185c8059c24f/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c73b5053-e6d0-4368-9f5a-4d375630419d/image.png)

- 2NF : 
- 1 정규화 만족 + PK가 아닌 속성이 PK에 완전 함수 종속되어야 한다.
- 즉, 테이블의 모든 칼럼이 기본키에 대해 완전히 종속되어야 함 
- 아래는 생산지가 생산자에 부분 함수적 종속이 되어 있음
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5e221430-996e-4e61-87f9-d6c7b8fa46de/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a9295d4a-5253-4fd0-b1b3-0ede8dc8b644/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c40e0f43-9009-4ae3-bd35-3f186537890c/image.png)
    
- 3NF : 
- 2 정규화 만족 + 모든 속성이 PK에 **이행적 함수 종속이 되지 않아야 한다**. 
- 이행적 함수 종속 : 
   A→B, B→C 성립할 때에, A→C 성립하는 것 / 기본키 아닌 다른 속성에 종속된 경우 의미

- EX> 학생번호 1번 찍었을 때에 학과장 정보 나오는 것은 이상하다.
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9da13bf8-91d8-443c-80bf-25c21fbcc40a/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7a0d9b1a-38b5-4775-b536-8f35fa0b856f/image.png)
    
- BCNF ( Boyce Codd Normalization Form ) : 
- 3정규화 진행한 테이블에 대해 모든 결정자가 후보 키가 되도록 테이블을 분리하는 것 
- 3정규화에 PK가 아닌 속성이 PK의 속성을 결정지을 수 없어야 한다. 

- EX>
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/425d524c-9a23-4321-8dbf-ed5049a03faf/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cfc14439-6cad-4efc-853a-9b71a7295c80/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6ee62e3a-4b34-435e-8c8e-637b7f10264c/image.png)
    

# SQL 기본 :

- SQL ( Structure Query Language ) : 
DB에 정보 저장하고 처리하기 위한 프로그래밍 언어 
테이블의 형태로 구조화된 RDB에 요청을 질의 
관계형 DB와의 대화를 위해 사용하는 프로그래밍 언어
- Syntax : 
- SQL 키워드는 대소문자 구분 X 지만, 대문자 작성 권장 
- 각 SQL문 끝에는 세미콜론 (’;’) 필요

## SQL Statements :

- 수행 목적에 따른 4가지 유형의 Statements :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/958178d0-b28e-4259-85b2-04f731be4330/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7b0b6c10-8bb4-4ad3-9c28-84c1af9ae449/image.png)

# DDL ( Data Definition Language ) : 정의

## CREATE TABLE :

```sql
-- CREATE DATABASE
CREATE DATABASE db01_live
	-- 일부 RDBMS에서 지원
  DEFAULT CHARACTER SET = 'utf8mb4';
USE db01_live;

-- DDL CREATE : 테이블 생성
CREATE TABLE examples (
  ExamId INT **PRIMARY KEY** AUTO_INCREMENT, 
  -- email 사용한다고 하면 int -> varchar // **auto_inc -> UNIQUE**  
  LastName VARCHAR(50) NOT NULL,
  FirstName VARCHAR(50) NOT NULL
);

-- Table 스키마 확인 1 / 2 << 테이블 정보 알려줌 
DESCRIBE examples; -- 밑에 EX 
SHOW CREATE TABLE examples; -- 테이블 만들때 친 명령어 보여줌 
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9be2535c-8561-49a7-98fc-7db8d64879b8/image.png)

- MySQL Dtype :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/40a75f6b-9a5e-4b85-af00-538a4565e4dd/image.png)

```sql
-- 제약 조건 정의하여 테이블 생성
CREATE TABLE users (
    userid INT PRIMARY KEY AUTO_INCREMENT,
    examid INT,
    username VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL,
    age INT NOT NULL **CHECK** (age >= 0 AND age <= 150),
    balance INT DEFAULT 0, 
    
    -- 위에서 선언한 examid가 실은 foreign key 인데, 너무 조건이 길어서 밑에서 진행 
    FOREIGN KEY (examid) REFERENCES examples(examid),
    -- examples 테이블의 examid 값을 참조할 것 
    CHECK (balance >= -5000 AND balance <= 5000),
 
    -- constraint : 제약조건에 이름 부여하는 키워드 
    CONSTRAINT unique_email UNIQUE (email),
    CONSTRAINT valid_email CHECK (email LIKE '%_@__%.__%')
    -- 이메일이 이런 형식이어야 한다고 체크 
);
DESCRIBE users;
SHOW CREATE TABLE users;

```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/587605ee-e705-4d15-83b3-70f057e7ac03/image.png)

- 제약 조건 ( Constraints ) : 
- 테이블의 필드에 적용되는 규칙 / 제한사항 
- 데이터의 무결성 유지 & DB의 일관성 보장

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b3144aff-8bf9-4bb0-aa89-226e01eee5fa/image.png)

- AUTO_INCREMENT 특징 : 삭제된 값은 무시되며 재사용 X

## ALTER TABLE :

- 테이블과 필드를 조작

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b58c0f5e-5e14-4153-88f3-41f819ef3ce3/image.png)

```sql
-- 테이블 수정

-- Add Column : 세미콜론까지가 한줄이라는거 생각
ALTER TABLE examples
ADD COLUMN Country VARCHAR(100) NOT NULL DEFAULT 'default value'
ADD COLUMN Age INTEGER NOT NULL DEFAULT 0,
ADD COLUMN Address VARCHAR(100) NOT NULL DEFAULT 'default value';

-- Rename Column / Table
ALTER TABLE examples
RENAME COLUMN Address TO PostCode  -- A TO B 있으면 컬럼명 / TO만 있으면 컬럼명 
RENAME TO new_examples;

SHOW CREATE TABLE users;

DROP TABLE users;
```

## DROP & TRUNCATE TABLE :

- DROP : 테이블 삭제
- DELETE : 데이터 삭제  →  데이터 생성 > 삭제 > 생성하면 ID 2부터 시작
- TRUNCATE : 테이블을 DROP 시켰다가 다시 만든다  →  위와 동일하게 하면 ID 1부터 시작

```sql
-- 테이블 삭제
DROP TABLE new_examples;

-- 테이블 데이터 비우기
-- 데이터를 넣는 것을 아직 진도가 나가지 않아 실습은 다음 수업에서 진행
TRUNCATE TABLE examples;

DELETE FROM examples;
```

- 과제 코드 : 
1. 외래키 선언 및 참조 방법 확인 
2. ALTER 할 시 여러줄 ALTER할 생각이면 , 으로 나누기
   기존 칼럼 수정할 때에는 MODIFY 사용

```sql
-- 1. 
CREATE DATABASE sns_db;
USE sns_db;

CREATE TABLE users(
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(255),
    email VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE posts(
    id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) -- 중요 포인트 
);
-----------------------------------------------------------------------------
-- 2. 
ALTER TABLE users
ADD COLUMN profile_picture VARCHAR(255),
MODIFY email VARCHAR(320);

ALTER TABLE posts
ADD COLUMN title VARCHAR(255),
MODIFY content LONGTEXT;
```

# DML ( Data Manipulation Language ) : 조작

## INSERT : 추가

- 테이블에 레코드 삽입

```sql
CREATE TABLE articles (
  id INT PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(100) NOT NULL,
  content VARCHAR(200) NOT NULL,
  createdAt DATE NOT NULL
);

INSERT INTO   articles (title, content, createdAt)
VALUES -- 단일도, 여러개도 가능  
  ('title1', 'content1', '1900-01-01'),
  ('title2', 'content2', '1800-01-01'),
  ('title3', 'content3', '1700-01-01');
-- VALUES   ('mytitle', 'mycontent', NOW()); DATE는 이렇게도 가능 
  
SELECT * FROM articles;
```

## UPDATE : 수정

- 테이블 레코드 수정
- SET절 : 수정할 필드 = 표현식 OR 값

```sql
-- Update 활용 : id에 따른 데이터 수정

UPDATE   articles
SET  title = 'update Title'
WHERE  id = 1;

UPDATE   articles
SET
  title = 'update Title',
  content = 'update Content'
WHERE  id = 2;

SELECT * FROM articles;
```

## DELETE : 삭제

- DELETE FROM 테이블명

```sql
-- Delete 활용 -  1번 레코드 삭제
DELETE FROM   articles
WHERE   id = 1;

-- Truncate 과 Delete 와 비교

-- 아이디가 초기화 안됨
DELETE FROM articles;
INSERT INTO   articles (title, content, createdAt)
VALUES   ('hello', 'world', '2000-01-01');
SELECT * FROM articles;

-- 아이디가 초기화 됨
TRUNCATE articles;
INSERT INTO   articles (title, content, createdAt)
VALUES   ('hello', 'world', '2000-01-01');
SELECT * FROM articles;
```

- 조회는 기능이 많아 DQL로 따로 빼서 구분하기도 함

# DQL ( Data Query Language ) : 검색

## SELECT :

- SELECT 칼럼 FROM 테이블 WHERE 조건

```sql
USE world;
-- SELECT 활용---------------------------------------------------- 
SELECT   Code, Name -- 테이블 country 에서 Code, Name 필드의 모든 데이터를 조회
FROM   country;

SELECT *            -- 테이블 country 에서 모든 필드 데이터를 조회
FROM country;

-- Name 필드 모든 데이터 조회 (**조회 시 Name 이 아닌 ‘국가’로 출력 될 수 있도록 변경**)
SELECT   Name **AS** '국가'
FROM   country;

-- Name, Population 필드의 모든 데이터 (Population 1000 으로 나눠 k 단위 값으로 출력)
SELECT  Name,  Population / 1000 AS '인구 (k)'
FROM  country;
```

# Filtering Data :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/93f1547a-ea9b-483b-82db-d862709e9da4/daf3f36e-2d2a-4c8f-8b89-6dae6d1900b6.png)

## DISTINCT:

```sql
-- Filtering Data - Distinct 활용-----------------------------------------
SELECT   DISTINCT Continent -- Continent 필드의 데이터를 중복없이 조회
FROM   country;
```

- **여러 component 앞에 DISTINCT 하나 ? 
ㄴ 3개가 동일해야 중복이라고 봄**

## WHERE

```sql
-- WHERE 활용 1 - Region필드값 ‘Eastern Asia’인 데이터의 Name, Population 조회
SELECT   Name, Population
FROM   country
WHERE Region = 'Eastern Asia';

-- IndepYear 필드 값이 Null이 아닌 데이터의 Name, Region, Population, IndepYear 조회
SELECT   Name, Region, Population, IndepYear
FROM   country
WHERE  IndepYear IS NOT NULL;

-- 테이블 country 에서 Population 필드 값이 천 만 이상이고 
-- LifeExpectancy 필드가 78 이상인 데이터의 Name, Region, LifeExpectancy 조회
SELECT  Name, Region, LifeExpectancy
FROM   country
WHERE  Population >= 10000000  AND LifeExpectancy >= 78;
```

## Database Operator :

- == X = O

```sql
-- **BETWEEN** Operator example
-- Poputation 필드 값이 백만 ~ 오백만 & GNPOld가 GNP 보다 큰 데이터의 
-- Name, Region, Population 그리고 GNPOld와 GNP 차이를 GNP diff로 작성하여 조회
SELECT  Name, Region, Population,   GNP - GNPOld AS 'GNP Diff' 
FROM   country
WHERE  Population BETWEEN 1000000 AND 5000000  AND GNP < GNPOld;

-- IN Operator example
-- Continent 값이 -- ‘North America’ / ‘Asia’ 인 데이터의 Code, Name, Continent 조회
SELECT  Code, Name, Continent
FROM  country
WHERE  Continent **IN** ('North America', 'Asia'); -- **이 TUPLE에 포함되어 있는 경우에만**
```

- LIKE :  값이 **특정 패턴에 일치하는지** 확인 ( 와일드카드와 함께 사용 )
- % : 0개 이상의 문자열과 일치하는지
- _ : 단일 문자와 일치하는지

```sql
-- Like Operator example 1
SELECT   Name, Region, Population, GNP
FROM   country
WHERE  Name LIKE 'South%';-- Name 필드 값이 ‘South’으로 시작하는
-- WHERE  Name LIKE 'South______'; 공백을 포함하여 6자리를 가지는
-- 이메일 : '%_@__%.__%'
```

- **NULL**은 =을 이용해 확인 불가 > **IS를 통해 확인**

```sql
-- IS Operator example
SELECT   Name, GNPOld, IndepYear
FROM   country
WHERE  GNPOld IS NULL  AND IndepYear IS NOT NULL;
```

- 우선순위 : 괄호 > NOT > 비교 > **AND > OR**

```sql
-- Operator 우선 순위 example
-- Ver. Wrong
SELECT   Name, IndepYear, LifeExpectancy
FROM   country
WHERE  IndepYear = 1901 OR IndepYear = 1981   AND LifeExpectancy <= 75;

-- Ver. Good
SELECT   Name, IndepYear, LifeExpectancy
FROM   country
WHERE  (IndepYear = 1901 OR IndepYear = 1981)  AND LifeExpectancy <= 75;
```

# MySQL Built-in Function

![ISFULL X             IFNULL O ](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5c7cd57c-0d2b-41b6-86a5-29794ae5c9a8/image.png)

ISFULL X             IFNULL O 

```sql
-- 문자형 함수
SELECT CONCAT('FirstName', '_', 'LastName'); -- CONCAT

-- TRIM : 왼쪽OR오른쪽의 특정 문자 삭제  
SELECT TRIM('   PHONE   '); -- STR 생략시 공백 삭제
SELECT TRIM('-' FROM '---TITLE---'); -- TITLE
SELECT TRIM(LEADING '-' FROM '---TITLE---'); -- 앞에만
SELECT TRIM(TRAILING '-' FROM '---TITLE---'); -- 뒤에만 

SELECT REPLACE('$10000', '$', '￦');-- REPLACE

-- LOCATE // A를 B에서 어디에 있는지 찾아줌 idx 1부터 9번째 자리에 있다. 
SELECT LOCATE('path', 'www.web-path-site.com/path/');
SELECT LOCATE('path', 'www.web-path-site.com/path/', 10); -- 시작 위치를 10으로 
-- POSITION : 
SELECT POSITION('path' IN 'www.web-path-site.com/path/'); -- 결과는 LOCATE와 동일 
-- 위의 2개 다 없으면 0 리턴 

-- 숫자형 함수
SELECT ABS(-12);-- ABS 함수
SELECT MOD(10, 7);-- MOD 함수 (나머지 구하기)    SELECT 10%7;   -- MOD와 동일함
SELECT POW(2, 6); -- POW 함수 (제곱근 구하기)    SELECT POWER(2, 6); 
SELECT CEIL(3.7); -- CEIL 함수  - 올림  SELECT CEIL(3.3); 
SELECT FLOOR(3.7);-- FLOOR 함수 - 내림   SELECT FLOOR(3.2);
SELECT ROUND(3.7);-- ROUND 함수 - 반올림 SELECT ROUND(3.2);

-- 날짜형 함수
SELECT CURDATE();-- 현재 날짜
SELECT CURTIME();-- 현재 시간
SELECT NOW();-- 현재 날짜 및 시간
SELECT DATE_FORMAT('2024-08-23 13:35:20', '%b-%d (%a) %r'); -- 시간 포멧 설정하기

-- NULL 관련 함수
-- IFNULL 함수 : 1번째 IN이 NULL이면, 2번째 값 반환 / 아니면 1번째 값 반환 
SELECT IFNULL(NULL, 'expr1 is NULL');
SELECT IFNULL('expr1', 'expr1 is NULL');

SELECT   Name,  IFNULL(IndepYear, 'no_data')  -- IndepYear,  
FROM   country
WHERE  Continent = 'North America'; 

-- NULLIF 함수 : 동일하면 NULL / 다르면 1번째 값 반환 
SELECT NULLIF('expr1', 'expr1');
SELECT NULLIF('expr1', 'expr2');

SELECT   Name, NULLIF(Population, 0)
FROM country;

-- COALESCE 함수 : 첫번째 인자부터 순서대로 확인하여 NULL이 아닌 값 반환 
--                 모두 NULL이면 NULL 반환 < 밑의 예시 보기 
SELECT COALESCE('expr1', 'expr2', NULL);
SELECT COALESCE(NULL, 'expr2', NULL);
SELECT COALESCE(NULL, NULL, NULL);

SELECT   Name,  COALESCE(NULLIF(GNP, 0), GNPOld, 'No data') AS gnp_data
FROM country
WHERE  Continent = 'Africa'  AND LifeExpectancy < 70;

-- *********** 동일한 기능, case문 사용 
SELECT Name, 
	CASE
		WHEN GNP > 0 THEN CONCAT(GNP, ' (GNP) ')
		WHEN GNPOld IS NOT NULL THEN CONCAT(GNPOld, ' (GNPOld) ')
		ELSE 'no_data'
	END AS gnp_data
FROM country
WHERE  Continent = 'Africa'  AND LifeExpectancy < 70;
```

# Sorting Data :

## ORDER BY

- 조회 결과의 레코드를 정렬
- SELECT select_list FROM 테이블명 ORDER BY 기준컬럼(ASC/DESC)

```sql
-- Order by example 1
SELECT   GovernmentForm
FROM   country
ORDER BY  GovernmentForm;
-- ORDER BY  GovernmentForm DESC;

-- 동일한거 보는데 다른 기준으로 
SELECT   GovernmentForm, SurfaceArea
FROM   country
ORDER BY  GovernmentForm DESC, SurfaceArea ASC;

SELECT  Name, IndepYear
FROM  country
WHERE  Continent = 'Asia' -- WHERE절 함께 가능 
ORDER BY   IndepYear;

SELECT  Name, IndepYear
FROM  country
WHERE  Continent = 'Asia'
ORDER BY   IndepYear IS NULL, IndepYear;
```

## LIMIT

```sql
-- Limit example 1 : 조회하는 최대 레코드 수 지정 
SELECT   IndepYear, Name, Population
FROM   country
ORDER BY   Population DESC
LIMIT 7; -- 7개만 나옴 

SELECT   IndepYear, Name, Population
FROM   country
ORDER BY   Population DESC
LIMIT 4, 7; -- 5 ~ 12 
-- LIMIT 7 OFFSET 4;  # 내 앞쪽에 대해 4개 무시하겠다 그 다음부터 7개 택 
```

# Grouping Data :

## Aggeregate Function :

- Aggregate Function ( 집계 함수 ) : 값에 대한 계산 수행하고 단일한 값 반환하는 함수
GROUP BY 절과 많이 사용

```sql
SELECT  Continent, COUNT(*)  # COUNT * : NULL값을 포함한 행의 수 출력 
                             # COUNT expr : NULL을 제외한 행의 수를 출력  

STDDEV(expr) : NULL값을 제외한 표준편차 
VARIANCE(expr) : NULL값을 제외한 분산 
```

## GROUP BY :

- 레코드를 그룹화해서 요약본 생성 ( 집계함수와 함께 사용 )
- <> DISTINCT : 중복제거

SELECT 칼럼

FROM 테이블

GROUP BY (그룹화 할 필드 목록) 

```sql
-- Group by 예시 1
SELECT  Continent
FROM  country
GROUP BY  Continent;

SELECT  Continent, COUNT(*)  
FROM  country
GROUP BY  Continent;

SELECT  Continent,   ROUND(AVG(GNP), 2) AS avg_gnp
FROM  country
GROUP BY  Continent;

###########################################################

-- error code
SELECT  Region,  COUNT(Region) AS count_reg
FROM  country
WHERE  count_reg BETWEEN 15 AND 20 
GROUP BY  Region
ORDER BY   count_reg DESC;

-- correct code *************
SELECT  Region,  COUNT(Region) AS count_reg
FROM  country
GROUP BY  Region
HAVING  count_reg BETWEEN 15 AND 20 -- GROUP BY의 WHERE은 HAVING 이후에 사용 
ORDER BY   count_reg DESC;
```

### SELECT문 실행순서 :

FROM > WHERE > GROUP BY > HAVING > SELECT > ORDER BY > LIMIT

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/71ac4d2c-cd8b-4514-bcf4-7bef52ee85d4/image.png)

- 오늘 과제 코드 복기

```sql
-- 데이터베이스 생성
CREATE DATABASE libraries;

-- 데이터베이스 사용
USE libraries;

-- books 테이블 생성
CREATE TABLE books (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    publisher VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL,
    published_date DATE NOT NULL,
    isbn VARCHAR(13) NOT NULL UNIQUE,
    price DECIMAL(5,2) NOT NULL,
    genre VARCHAR(50) NOT NULL
);

INSERT INTO books(title, publisher, author, published_date, isbn, price, genre)
VALUES
('The Great Gatsby', 'Scribner', 'F. Scott Fitzgerald', 
	'1925-04-10', '9780743273565', 10.99, 'Classic'),
('1984', 'Secker & Warburg', 'George Orwell', 
	'1949-06-08', '9780451524935', 8.99, 'Dystopian');

SELECT * FROM books

##############################################################
use libraries;

-- 특정 레코드의 값을 수정하려면 UPDATE
UPDATE books
SET price = 12.99, 
    genre = 'Science Fiction' 
WHERE isbn = '9780451524935';

UPDATE books
SET publisher = "Charles Scribner's Sons" 
WHERE isbn = '9780743273565';

SELECT * FROM books

```

# JOIN

- 관계 : 여러 테이블 간의 논리적 연결 
이런식으로 만들면 user의 role이 변해도 articles에는 아무런 영향이 가지 않아 좋지만, 
하석주 user가 쓴 모든 글 보기 위해서는 articles / users 테이블을 취합해야 한다. 

이때 사용해야 하는게 JOIN

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/977dd3df-80d8-454a-ade5-05a10ac4e36a/image.png)

## JOIN절 :

- 둘 이상의 테이블에서 데이터를 검색하는 방법

## INNER / LEFT / RIGHT / SELF JOIN :

- 테이블 소개

```sql
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  name VARCHAR(50) NOT NULL,
  age INTEGER,
  parent_id INTEGER,
  FOREIGN KEY (parent_id) REFERENCES users(id)
);

CREATE TABLE articles (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(50) NOT NULL,
  content VARCHAR(100) NOT NULL,
  userId INTEGER,
  FOREIGN KEY (userId) REFERENCES users(id)
);
```

- INNER JOIN : 두 테이블에서 값이 일치하는 레코드에 대해서만 결과 반환

SELECT

FROM 왼쪽

INNER JOIN 오른쪽 

ON 조인 조건 

WHERE 조건 

```sql
SELECT articles.*, users.id, users.name 
FROM articles
INNER JOIN users   
ON users.id = articles.userId;

-- 1번 회원(하석주)가 작성한 모든 게시글의 제목과 작성자명을 조회
SELECT articles.title, users.name  -- articles의 제목, 유저의 이름만 
FROM articles
INNER JOIN users   
ON **users.id = articles.userId**
WHERE users.id = 1;  -- 하석주 ID 
```

- LEFT & RIGHT JOIN : 
오른쪽 테이블의 일치하는 레코드와 함께 **왼쪽 테이블의 모든 레코드** 반환 
왼쪽 테이블의 일치하는 레코드와 함께 **오른쪽 테이블의 모든 레코드** 반환
- INNER 와의 차이 : INNER는 NULL값은 무시하고 교집합만 가져온다면, 
LEFT, RIGHT는 한쪽 레코드를 모두 가져오기 때문에 NULL값도 같이 가져옴
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/022d2ed6-8dd1-4eb9-91b0-ad55e555feb5/image.png)
    

```sql
SELECT articles.*, users.id, users.name 
FROM articles
LEFT JOIN users   
ON users.id = articles.userId;

-- 게시글을 작성한 이력이 없는 회원 정보 조회
SELECT users.name FROM users
LEFT JOIN articles   ON articles.userId = users.id
WHERE articles.userId IS NULL;
```

```sql
-- RIGHT JOIN
SELECT articles.*, users.id, users.name 
FROM articles
RIGHT JOIN users   
ON users.id = articles.userId;
```

- SELF JOIN : 동일한 테이블의 칼럼 비교해 일치하는 데이터 추가로 붙여 반환 
계층적 데이터 구조 표현 / 동일 테이블 내에서 특정 관계 찾을 때에 사용

```sql
SELECT   

parent.id AS parent_id,    # AS 생략 가능 
parent.name AS parent_name, 
child.id AS child_id,   
child.name AS child_name

FROM   users parent # ( AS 빠진 모습 ) 
INNER JOIN   users child 
ON parent.id = child.parent_id;

-- 테이블을 구분하지 않으면 구분이 어려워지기 때문에 Error 발생

-- SELECT 
--   users.id AS parent_id, 
--   users.name AS parent_name, 
--   users.id AS child_id, 
--   users.name AS child_name
-- FROM users
-- JOIN users 
-- ON users.id = users.parent_id;

-- 서로의 형제자매가 누구인지 id와 이름 조회
SELECT 
  users.id AS user_id, 
  users.name AS user_name, 
  sibling.id AS sibling_id, 
  sibling.name AS sibling_name
FROM   users
JOIN   users sibling ON users.parent_id = sibling.parent_id
WHERE  users.id != sibling.id;  # 본인 제외 ( 형제자매만 뽑기 위해 ) 

```

# SubQuery

- 하나의 SQL문 안에 포함되어 있는 또다른 SQL문
- 복잡한 데이터 이용해 검색 / 값을 비교할 때에 사용

- 사용 방법 : 괄호 감싸서 사용
- 사용 가능한 곳 :
1. SELECT절 : 일반적으로 계산된 값을 하나 이상의 칼럼으로 반환해야 할 때 사용 
2. FROM 절 : 중간 집계나 계산, 결과 재사용, 조인 조건 단순화 할 때에 사용 
3. WHERE절 : 특정 조건을 만족하는 데이터 필터링 OR 검색할 때에 사용 
- 그 외 HAVING / INSERT의 VALUES절 / UPDATE의 SET절에서 사용

- 종류 : 단일 행 / 다중 행 / 다중 열 서브쿼리

## Single-row SubQuery :

- 실행 결과가 항상 1건 이하인 서브쿼리
- 비교 연산자 ( = / < / > / … / <> ) 와 함께 사용

```sql
-- 'North America' 대륙의 평균 인구보다 인구가 많은 국가의 이름과 인구 조회
SELECT   Name, Population
FROM   country
WHERE   Population > 
	(SELECT AVG(Population)     FROM country    WHERE Continent = 'North America');

-- **단일행이 아닐 때 에러 발생**
-- SELECT Name, Population
-- FROM country
-- WHERE Population > 
-- (SELECT Population FROM country WHERE Continent = 'North America');

-- AVG가 아닌 열 자체를 들고와서 에러 발생 

-- country 테이블의 ‘Benin’ 이라는 국가의 국가 코드를 이용하여, 
-- city 테이블에 등록된 ‘Benin’의 모든 도시 정보를 조회
-- 아래 2개는 동일 결과 / BUT JOIN이 빠름 
SELECT  * 
FROM   city
WHERE  CountryCode = (SELECT Code   FROM country    WHERE Name = 'Benin');

-- with JOIN statement
SELECT c.* FROM city c
JOIN country ct
ON ct.Code = c.CountryCode
WHERE ct.Name = 'Benin';

-- country 테이블의 각 국가 마다 해당 국가 소속의 평균 도시 인구를 
-- city 테이블을 이용하여 조회
SELECT   country.Name AS CountryName,
  (SELECT AVG(city.Population) FROM city WHERE city.CountryCode = country.Code) 
  AS AvgCityPopulation
FROM country;

-- with JOIN statement
SELECT   country.Name AS CountryName,  AVG(city.Population) AS AvgCityPopulation
FROM country
LEFT JOIN city 
  ON city.CountryCode = country.Code
GROUP BY country.Code, country.Name;

-- 인구가 10,000,000명 이상인 국가들의 국가 코드, 도시 이름, 지구, 인구를 조회
SELECT co.Code, c.Name, c.District, c.Population
FROM   city c
JOIN (
      SELECT Code
      FROM country
      WHERE Population >= 10000000
    ) co
  ON c.CountryCode = co.Code;
```

## Multi-row SubQuery :

- 실행 결과가 여러 개의 결과 행을 반환할 수 있는 서브쿼리
- IN / ANY / ALL 연산자와 함께 사용

```sql
-- Asia 에 소속된 모든 도시를 조회
SELECT * 
FROM city
WHERE 
  **CountryCode IN** ( 
    SELECT Code
    FROM country
    WHERE Continent = 'Asia'
  )

-- 인구가 10,000,000명 이상인 국가들의 국가 코드, 도시 이름, 지구, 인구를 조회
SELECT co.Code, c.Name, c.District, c.Population
FROM 
  city c
  JOIN (SELECT Code
        FROM country
        WHERE Population >= 10000000) co
	ON c.CountryCode = co.Code;
```

## Multi-column SubQuery

- 실행 결과가 하나 이상의 칼럼을 반환하는 서브쿼리
- 비교 연산자 / EXISTS / IN 연산자와 함께 사용

```sql
-- 각 대륙별 독립 연도가 최근인 국가의 이름, 대륙, 독립 연도를 조회
SELECT Name, Continent, IndepYear
FROM country
WHERE (Continent, IndepYear) IN (
    SELECT Continent, MAX(IndepYear)
    FROM country
    GROUP BY Continent
);

-- Africa에서 가장 땅이 넓은 국가의 이름, 대륙, 면적을 조회
SELECT Name, Continent, SurfaceArea
FROM country
WHERE (Continent, SurfaceArea) = (
    SELECT Continent, MAX(SurfaceArea)
    FROM country
    GROUP BY Continent
    HAVING Continent = 'Africa'
);

```

# DB Index

## INDEX / Index 종류

- DB에서 데이터를 보다 빠르게 찾기 위해 사용되는 자료구조
- 장점 : 조회하는 속도 전반적으로 빠름 / 시스템 부하 적음
- 단점 : 인덱스 정보 추가로 저장 위한 공간 필요 
          삽입 수정 삭제 빈번한 테이블이면 성능 오히려 떨어짐

- 기본 인덱스 : 
일반적으로 선언했을 때의 인덱스
인덱스로 설정된 칼럽의 데이터에 NULL값 / 중복 가능

- 유니크 인덱스 : 
UNIQUE 키워드와 함께 선언했을 때의 인덱스
인덱스를 설정할 칼럼의 데이터들은 각각 고유한 값이어야 함 
중복된 데이터 존재 시 에러발생 / 칼럼 추가로 구성해 고유한 값으로 구성하여 해결

## Index 생성 / 추가 / 사용 / 삭제

1. 생성 ( 추가보다 효율적 ) 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cba103c6-0324-42a4-8697-4183d3351fd6/image.png)

- 인덱스로 설정하는 컬럼 : 자주 검색되는 / 중복되는 데이터가 적은

1. 추가

CREATE INDEX 인덱스명

ON 테이블명 (칼럼명, … ) 

ALTER TABLE 테이블명

ADD INDEX 인덱스명 ( 칼럼명 … ) 

```sql
-- DROP INDEX code_idx
-- ON city;

-- Index 생성하기
CREATE TABLE table_name (
  column1 INT PRIMARY KEY AUTO_INCREMENT,
  column2 VARCHAR(150) DEFAULT NULL,
  column3 VARCHAR(30),
  -- 생성하기 1 (INDEX 키워드 사용)
  INDEX index_name (column2),
  -- 생성하기 2 (KEY 키워드 사용)
  KEY index_name2 (column3)
);

-- index 추가하기 1
CREATE INDEX index_name
ON table_name (column1, column2, ...);

-- index 추가하기 2
ALTER TABLE table_name
ADD INDEX index_name (column1, column2, ...);
```

```sql
SHOW CREATE TABLE city;

-- CREATE TABLE `city` (
--   `ID` int NOT NULL AUTO_INCREMENT,
--   `Name` char(35) NOT NULL DEFAULT '',
--   `CountryCode` char(3) NOT NULL DEFAULT '',
--   `District` char(20) NOT NULL DEFAULT '',
--   `Population` int NOT NULL DEFAULT '0',
--   PRIMARY KEY (`ID`),
--   KEY `CountryCode` (`CountryCode`),
--   KEY `idx_city_name` (`Name`),
--   CONSTRAINT `city_ibfk_1` FOREIGN KEY (`CountryCode`) REFERENCES `country` (`Code`)
-- ) ENGINE=InnoDB AUTO_INCREMENT=4080 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci

-- index 사용
-- 일반 조회시 활용
SELECT * FROM city
WHERE Name = 'Seoul';

-- 정렬시 활용
SELECT * FROM city
ORDER BY Name;

-- JOIN 시 활용
SELECT city.Name, country.Name 
FROM city
JOIN country ON city.CountryCode = country.Code
WHERE city.Name = 'Seoul';

-- index 삭제하기 1
ALTER TABLE table_name
DROP INDEX index_name;

DROP INDEX index_name
ON table_name;
```

- 인덱스 정리 :

데이터는 주로 탐색 / 정렬 할 때 매우 효율적 / 단, 데이터 양적으면 이점 떨어짐

주로 유일한 값 칼럼 사용할수록 유리 / 과도한 인덱스 사용 → 성능 저하 

추가 공간 필요 → 비용적 측면 고려해야 

### VIEW : 불러올때마다 갱신

- DB 내에서 저장된 쿼리의 결과를 가상으로 나타내는 객체 = 가상 테이블
- 실제 데이터 X, 실행 시점에 쿼리 실행해 결과를 생성하여 보여줌

- 장점 : 
1. 편리성 : 복잡한 쿼리를 미리 정의된 뷰로 만들어 단순화
3. 재사용성 : 동일한 뷰를 여러 쿼리에서 사용 가능

CREATE VIEW 뷰이름 

AS 

SELECT문 

```sql
-- View 활용 1
-- 국가 코드와 이름을 v_simple_country라는 이름의 view로 생성
CREATE VIEW v_simple_country 
AS
SELECT Code, Name
FROM country;

SELECT * FROM v_simple_country;
-- drop view v_simple_country;

-- 각 국가별 가장 인구가 많은 도시를 조인한 뷰를 
-- v_largest_city_per_country 라는 이름의 view로 생성
CREATE VIEW v_largest_city_per_country AS
SELECT 
  c.Name AS CountryName, 
  ci.Name AS LargestCity, 
  ci.Population AS CityPopulation,
  c.Population AS CountryPopulation
FROM country c
JOIN city ci ON c.Code = ci.CountryCode
WHERE (ci.CountryCode, ci.Population) IN (
    SELECT CountryCode, MAX(Population)
    FROM city
    GROUP BY CountryCode
);

SELECT * FROM V_largest_city_per_country;

-- 생성한 View 활용
-- v_largest_city_per_country 뷰를 이용하여 Asia 국가 중 가장 인구가 
-- 작은 곳 보다 인구가 많은 모든 도시의 개수를 조회
SELECT COUNT(*) AS CountCity
FROM **v_largest_city_per_country -- 변수와 같이 이용**
WHERE CityPopulation >= (
  SELECT MIN(Population)
  FROM country
  GROUP BY Continent
  HAVING Continent = 'Asia'
);
```

### B-tree

- 가장 일반적으로 사용하는 INDEX 구조
- 이진 탐색 트리의 한계 ( 최악 O(N)을 보완한 것이 B-Tree )
- B-Tree는 하나의 레벨에 많은 자료 저장 가능해 높이가 낮음
- 높이가 낮다 = leag node까지의 거리가 짧다 = 탐색 성능이 좋다
- B-tree는 항상 **O(logN)의 성능** 가진다