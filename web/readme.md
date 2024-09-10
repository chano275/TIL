# WEB

---

# HTML & CSS

- VSCode에서 alt + b 누르면 바로 웹사이트 열림 ( 익스텐션 필요 ) 
웹
- WWW : 인터넷으로 연결된 컴퓨터들이 정보 공유하는 거대한 정보 공간
- Web Page : **HTML(구조) + CSS(스타일링) + JS(행동, 조작)**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/bed2b949-0234-47b1-acc5-5126a4af3c9f/Untitled.png)

---

## 웹 구조화 - HTML

- HyperText Markup Language : 웹 페이지의 의미, **구조를 정의**하는 단어
- HyperText : 웹 페이지를 다른 페이지로 연결하는 링크
                   참조를 통해 사용자가 한 문서에서 다른 문서로 즉시 접근할 수 있는 텍스트
                   비 선형적으로 이루어진 텍스트
- **MarkUp** Language : 태그 등을 이용해 문서 / 데이터의 **구조**를 명시하는 언어 
ex> HTML / Markdown
- ! + TAB : 기본 구조 잡아줌  [ ex> h1 + TAB ]

### HTML 구조 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a95b08c2-9dac-4d3e-bbcf-7fe2234bea01/Untitled.png)

- <!**DOCTYPE** html> : **해당 문서가 html 문서**리는 것을 나타냄
- **<html>**</html> : **전체 페이지의 콘텐츠를 포함**
- **<head>**</head> : HTML 문서에 관련된 **설명, 설정** 등 / 사용자에게 보이지 않음
- **<body>**</body> : 페이지에 **표시되는 모든 컨텐츠**

### HTML Element(요소) :

- 태그 : Opening tag + content + Closing tag 
          <p> <h1> 등… 
          <p></p> : 본문 나타내는 태그, 두 태그 중간은 내용(본문)
- 여는 태그 : / 없이  /  종료 부분 : / 붙이고   
** 닫는 태그 없는 태그도 존재 [ **img** ]

### HTML Attributes(속성) : [ class, … ]

- EX> <p class=”editor-note”>content</p>
        속성 : class=”editor-note”
- 규칙
**1. 속성은 [ 요소이름 / 속성 ] 사이 공백이 있어야 함
2. 하나 이상의 속성 있으면 속성 사이를 공백으로 구분
3. 속성 값은 열고 닫는 따옴표로 감싸야**
- 목적
1. 나타내고 싶지 않지만, 추가적인 기능, 내용을 담고 싶을 때 사용
2. CSS 에서 해당 요소를 선택하기 위한 값으로 활용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0e2e9f75-8d17-4f2a-bf27-c1c07c239a74/Untitled.png)

- 크롬 개발자 도구를 통해서 html ex 볼 수 있음.  [ F 12 ]

## HTML Text 구조 :

- HTML의 주요 목적중 하나는, 텍스트 구조와 의미 제공하는 것
- <**h1**>content</h1> : 단순히 텍스트 크게 X / **현재 문서의 최상위 제목이라는 의미 부여**

- 대표적 예시 : 
1. Heading & Paragraphs    : h1 ~ h6 / p
2. Lists                                 : o1 / ul / li            순서가 있는 / 없는 / 리스트 아이템
3. Emphasis & Importance : em / strong         강조, 굵게
- html 태그들은 중첩될 수 있다  >  <p>This is <em>emphasis</em></p>
- div 와 card 태그의 차이 X ????????
html 에 지정되어 있는 element 외의 이름으로 요소 생성시
그렇게 생성된 모든 요소는 기본적으로 div와 같은 역할.
div는 그냥 하나의 블락이라는 의미 외에는 없는데, 우리가 직접 이름 지어주면

cf> user agent stylesheet : 각각의 브라우저에서 html 문서 열었을 때에 
                                           사용자가 보기 편하도록 제공해주는 기본적인 스타일시트

---

## 웹 스타일링 - CSS

- CSS : Cascading Style Sheet : 계단식 스타일 시트 
                                               웹 페이지의 디자인 / 레이아웃 구성하는 언어
- UI / UX 중점적으로 생각해야

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fd1683c8-de94-4f46-8c30-e3ce6496a174/Untitled.png)

- HTML 구조만 볼 수 있음.

### CSS 구문 :

- 어떤 대상에게 어떤 속성을 부여할 것인지

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b17449a2-6ee3-4ce7-b786-76b58a023112/Untitled.png)

### CSS 적용 방법 :

1. 인라인 스타일 : HTML 요소 안에 style 속성 값으로 작성 > 안씀 
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2467297d-1a97-435d-8f88-bea70b8e4f68/Untitled.png)
    

1. 내부 스타일 시트 : head 태그 안의 style 태그에 작성 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9205ddc0-35ae-45c5-b7d5-fb31355e715a/Untitled.png)

1. 외부 스타일 시트 : 문서가 커서 head 안에 적용되는 style 너무 많아질 때에
**별도의 css 파일** 생성 후 **HTML ‘link’ 태그 사용해 불러오기** 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/21e485c2-7482-4c99-a21b-99549579d906/Untitled.png)

```html
// CSS 적용 방법 3가지 

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
  
  // 외부  :  경로 = 상대경로 
  <link rel="stylesheet" href="style.css">
  
  // 내부
  <style>
    h2 {
      color: red;
    }
  </style>
  
</head>

// 인라인
<body>
  <h1 style="color:blue; background-color: yellow;">Inline Style</h1>
  <h2>Internal Style</h2>
  <h3>External Style</h3>
</body>

</html>
```

### CSS 선택자 ( Selectors )

- HTML 요소 선택하여 스타일 적용할 수 있도록 하는 선택자
- HEAD 내의 STYLE 에서 선언
BODY의 요소들에 자동적으로 적용됨

```html
// CSS 선택자 종류 < 내부 CSS 적용법으로 컨트롤 
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
  
  <style>
    /* 전체( * ) 선택자       : HTML 모든 요소를 선택 */
    * {
      color: red;
    }
    
    /* 요소( tag ) 선택자    : 지정한 모든 태그 선택 */
    h2 {
      color: orange;
    }
    h3,
    h4 {
      color: blue;
    }

    /* 클래스( class ) 선택자 (’.’ (dot)) : 주어진 클래스 속성 가진 모든 요소 선택 
                   class item 선언해 놓았을 때, 나중에 불러올 시 .item 형식으로 */
    .green {
      color: green;
    }

    /* 아이디( id ) 선택자 (’#’) : 주어진 아이디 속성 가진 요소 선택
              문서에는 주어진 아이디 가진 요소가 하나만 있어야 함 */
    #purple {
      color: purple;
    }

    /* 자식 결합자 (”>”) : 첫번째 요소의 직계 자식만 선택
           ex> ul > li 는 <ul> 안에 있는 모든 <li> 를 선택 ( 한단계 아래 자식들만 )  */
    .green>span {                // green class 가지고 있는 요소중 자식 span 태그? 마지막 
      font-size: 50px;           // 마지막 친구만 폰트 크기 키워라 
    }

    /* 자손 결합자 (” “ (space)) : 첫번째 요소의 자손 요소들 선택
          ex> p span 은 <p> 안에 있는 모든 <span> 을 선택 (하위레벨관계x) */
    .green li {                  // green 클래스 가지고 있는 요소의 자손 li 태그가 
      color: brown;              // 모두 브라운으로 : 파이썬 ~ python 모두 브라운으로 
    }
  </style>
</head>

<body>
  <h1 class="green">Heading</h1> // 위에 보면 .green 으로 위치 
  <h2>선택자</h2>
  <h3>연습</h3>
  <h4>반가워요</h4>
  <p id="purple">과목 목록</p> // 위에 #으로 purple 지정 
  <ul class="green">
    <li>파이썬</li>
    <li>알고리즘</li>
    <li>웹
      <ol>
        <li>HTML</li>
        <li>CSS</li>
        <li>PYTHON</li>
      </ol>
    </li>
  </ul>
  <p class="green">Lorem, <span>ipsum</span> dolor.</p>
</body>

</html>

```

### 명시도 (Specificity) :

- 결과적으로 **요소에 적용할 CSS 선언을 결정**하기 위한 알고리즘
- 동일 요소 가리키는 2개 이상의 CSS 규칙 > 가장 **높은 명시도 가진 선택자가 승리**해 스타일 적용 
CSS 는 Cascade(계단식) > 한 요소에 동일한 가중치의 선택지 여러개 > 맨 마지막 선언 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fd30992d-a3b1-4772-9a45-37d66268fe0b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2160ec64-bd0c-425b-9ab6-4739f28cb201/Untitled.png)

- 명시도 높은 순? : 
- importance : 우리가 직접적으로 스타일 적용할 때에는 사용 x   /   디버깅할때만 사용
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fcd62205-6086-41db-ae32-ea07c420fb97/Untitled.png)
    

```html
// 명시도 TEST CODE 
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
  <style>
    h2 {
      color: darkviolet !important;
    }

    p {
      color: blue;
    }

    .orange {
      color: orange;
    }

    .green {
      color: green;
    }

    #red {
      color: red;
    }
  </style>
</head>

<body>
  <p>1</p>
  <p class="orange">2</p>
  <p class="green orange">3</p>
  <p class="orange green">4</p>
  <p id="red" class="orange">5</p>
  <h2 id="red" class="orange">6</h2>
  <p id="red" class="orange" style="color: brown;">7</p>
  <h2 id="red" class="orange" style="color: brown;">8</h2>
</body>

</html>

```

## 웹 스타일링 - CSS Box Model :

- 모든 HTML 요소를 사각형 박스로 표현하는 개념

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/da12c895-fd3c-4ae2-8291-e34fc0d5dbca/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9f43c214-53ad-490c-8a6c-e3171f99961e/Untitled.png)

- 구성 요소 : 
내용(content) : 콘텐츠 표시 영역 
안쪽 여백(**padding**) : **콘텐츠 주위**에 
테두리(border) : 콘텐츠와 패딩 감싸는 테두리 영역 
외부 간격(margin) : 이 박스와 다른 요소 사이의 공백 / 가장 바깥 영역

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e21c8db1-fd1b-44f5-877f-14477f817920/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/36ac0d55-647b-4799-9754-f58dfcb5aaeb/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/12846028-3ff4-480b-bb1a-05e870a72643/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8c91b4b4-c409-4335-a53a-bae4d3e803e1/Untitled.png)

- width & height 속성 :
요소의 너비, 높이 지정 / 이때 지정되는 요소의 너비 & 높이는 **콘텐츠 영역** 대상

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/58f66aa5-a259-4aa6-8319-27ddf8c793e0/Untitled.png)

- 화살표(인스펙터) 통해서 요소들 정보 체크 가능
- width가 컨텐츠 영역 대상이기 때문에, **200px로 지정해도 전체 box는 301px**
- **computed** 통해서 더 자세히 볼 수 있음

```html
// 아래구문 통해 width 설정 > content 부분이 아닌 전체 박스가 200으로 

* {
  box-sizing: border-box; // 원래는 content-box 였음 
}
```

cf> shorthand 속성 : 
border  << border-width / border-style / border-color : 
style : 기본적으로 solid 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/84798586-7b4c-435f-a83b-5ecca9b73fa1/Untitled.png)

margin & padding : 상하좌우 X 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fda61f8a-787e-45f1-a125-dc21b2d4248f/Untitled.png)

- 마진 상쇄 : 40픽셀이 아닌 20픽셀이다

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a597aa25-adb3-4c38-b25e-62741be9fafe/Untitled.png)

## 웹 스타일링 - BOX TYPE :

- Block & Inline : 기본적으로 정해진게 있지만, 바꿀수도 잇음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ac8258fa-3446-4d73-9e3b-36946573a8fc/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6a3dcbd0-5515-4f23-941f-68bec90ea975/Untitled.png)

- Inline 타입 특징 :
1. 새로운 행으로 나뉘지 않음
2. width height 사용 불가능
3. 수직 방향 : padding / margins / borders 적용되지만 다른 요소 밀어낼 수 없음
    수평 방향 : padding / margins / borders 적용되어 다른 요소 밀어낼 수 있음 
4. 대표적 inline 타입 태그 : a / img / span

<a> 와 같이 링크 만들어주는 태그 : 
화면 출력시 굳이 새로운 단락으로 이동할 필요 없다 >> 기본적으로 inline 으로 구성 
동일한 inline : <span> / <img> : 이미지는 뉴스기사 생각

- block 타입 특징 : 
1. 항상 새로운 행으로 나뉨
2. width height 속성을 사용해 너비, 높이 지정 가능
3. 기본적으로 width 자정하지 않으면 박스는 inline 방향으로 사용가능한 공간 모두 차지
                         ( 너비를 사용가능한 공간의 100% 로 채움 ) 
4. 대표적인 block 타입 태그 : <h1 ~ h6>, <p>, <div>

- 속성에 따른 수평 정렬 ??????????

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a8ffddb3-5f0d-49c0-9635-92541cb51201/Untitled.png)

- 요소의 위치를 지정 : margin 통한 정렬
- content 에서 정렬 : text-align 통한 정렬

- Inline-block :
1. inline . block 의 중간 지점을 제공
2. block 요소의 특징 가짐
      width height 속성 사용 가능 
      padding margin 및 border 로 인해 다른 요소 밀려남 
3. 요소가 줄 바꿈 되는 것을 원하지 않으며 너비와 높이 적용하고 싶은 경우 사용

- flex :
1. 공간 배열 & 정렬 ( 레이아웃 ) 구성하는 방법 다룰 수 있음 
2. 요소를 행과 열 형태로 배치하는 1차원 레이아웃 방식
3. flex container 관련 속성 : 
display / fiex-direction / flex-wrap / justify-content / align-items / align-content
4. flex item 관련 속성 : aligh-self / flex-grow / flex-basis / order

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ae029466-4530-443e-b65d-36dc745407cd/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0f315421-1f18-4d37-b6f0-0000821f09be/Untitled.png)

- none : 
요소를 화면에 표시하지 않고 공간조차 부여되지 않음 
문서 구조상에는 존재하는데 화면에는 나오지 않음

## 웹 스타일링 - CSS Position :

- 요소를 normal flow 에서 제거하여 다른 위치로 배치하는 것 
다른 요소 위에 올리기 / 화면의 특정 위치에 고정시키기 등

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3ada051a-5542-4288-8628-9707395af623/Untitled.png)

- html 요소 모든 좌표값 기준 : 본인 box 왼쪽 위 꼭짓점
- Position 유형 : 
1. static : 기본. 

2. relative : 상대적
나의 normal flow 상 왼쪽 상단 꼭짓점 기준으로 이동 

3. absolute : 절대적 
absolute 속성 설정 시 본인의 normal flow 사라진다. 
본래 위치? : 부모중 relative 속성이 있는지 먼저 확인 > 
1. 없다 > 기준점 부모요소가 아니라 html 문서의 최상단 좌측 으로 위치 > 픽셀 이동 
2. 있다 > 부모 요소를 기준으로 이동 ? 
absolute 속성 없애면 기존에 없어졌던 공간 다시 먹고,  normal flow 따라간다 > relative 가 밀림 
밑의 네이버 vibe 에서 버튼이 absolute 

( 기준을 무엇으로 할 것인지에 따라 fixed / sticky 갈림 ) 
4. fixed : 고정된 시점 : 사용자가 바라보고 있는 화면 기준으로 고정지점 잡음 
               (에듀싸피 알림 메세지) 
5. sticky : 고정되어 있지만, 부모 요소 영역 안에서 fixed 되어있다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6ad3ee44-3629-40f0-93f0-b5762bd8ecc6/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0a8bc043-5abf-46e2-844e-e4e07e33853d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/00c34e32-c5ec-4b9d-a567-f6205f8063c1/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/42419fee-c5cb-47ff-aadf-853e729ff420/Untitled.png)

### Position 의 역할 :

- 전체 페이지의 레이아웃 구성 X ( 이거는 박스모델 / 디스플레이 ) 
페이지 특정 항목의 위치 조정 O

### CSS 상속 :

- 기본적으로 부모 요소의 속성 자식에게 상속해 재사용성 높임

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7072699c-783c-48b3-b4bf-0ff35271ce82/Untitled.png)

### Bootstrap :

- CSS 프론트엔드 프레임워크 ( Toolkit )
- 미리 만들어진 디자인 요소 제공 > 웹사이트 개발 빠르게

### MDN :

- HTML 태그, 기능, 속성, CSS 선택자 등등 설명 GOOD..

---

# Javascript

## JS & DOM 기본 개념 :

- 웹 브라우저에서의 JS : 웹 페이지의 동적 기능 구현
- **JS 코드 실행 환경** 종류 :
1. **HTML** 내에서 **body의 script 태그 안**에 
2. vue 에서 node 설치해 런타임 환경에서 **js 확장자 파일** 사용해 실행 
3. 브라우저에서 **개발자도구 > console** 에서 사용
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/164de1b9-1059-40c0-872c-b7c1d97adc35/Untitled.png)
    

- 네이버에서 JS 날리면 데이터가 거의 없는데, 처음부터 HTML, CSS에 작성되어있는 내용 보이는게 아니라 실시간으로 필요한 데이터 서버에서 받아오는 것이라 < 이게 JS로 작성 JS를 빼게 되면 대부분의 요소가 없는 네이버가 보이게 됨.
 
아이콘들도 개인 유저마다 설정 바꿀 수 있기 때문에, 아예 보이지 않는 구성이 됨. 

DOM
- The Document Object Model : 문서 객체 모델 
웹 페이지(Document)를 구조화한 객체로 제공하여, 
프로그래밍 언어가 페이지 구조에 접근할 수 있는 방법을 제공 

문서 구조 / 스타일 / 내용 등을 변경할 수 있도록 함

- DOM API : 통해서 js에 접근 > js에서 조작한걸 웹 페이지에 반환

다른 프로그래밍 언어가 웹 페이지에 접근 및 조작할 수 있도록 
페이지 요소들을 객체 형태로 제공하여 이에 따른 메서드 또한 제공
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b494d822-3942-4358-8e1d-a4e8782de328/Untitled.png)
    

- DOM 특징 :
DOM에서 모든 요소, 속성, 텍스트는 하나의 **객체** 
모두 **document 객체의 하위 객체로 구성**됨 
window ( 브라우저 자체, 새 창 ) > document > html > head/body > title h1 a > …
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/053580e0-26a1-424a-814c-baf5801f1afe/Untitled.png)
    

- DOM tree : 위의 이미지 참조 
브라우저는 HTML 문서 해석하여 DOM tree 라는 객체 트리로 구조화 
객체 간 상속 구조 존재

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/35efa1bd-aa7f-4a1f-9051-24c8aa9ea260/Untitled.png)

- 브라우저가 웹 페이지를 불러오는 과정 :
웹 페이지  >  웹 브라우저(파싱 = ~통해 해석됨)  >  웹 브라우저 화면 나타남 
HTML + CSS + JS  >  웹 브라우저  >  화면

- DOM 핵심 :
문서의 요소들을 객체로 제공하여 다른 프로그래밍 언어에서 
접근하고 조작할 수 있는 방법을 제공하는 API

## document 객체 : html 문서 자체

- 웹 페이지 객체
- DOM Tree 의 진입점
- 페이지를 구성하는 모든 객체 요소를 포함
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b3182a77-e30a-4143-9077-d0a93f08361e/Untitled.png)
    

### document 객체 예시 :

- HTML 의 title 변경하기
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/bdea2f03-b182-42b4-9609-7c4ba5e7f274/Untitled.png)
    

---

# DOM 선택 :

- DOM 조작 위해 선택
- document 객체 하위의 h1을 조작하기 위해 어떤 식으로 접근해야 하는지 ?

### DOM 조작 시 기억할 것 : 웹 페이지 조작 = 웹 페이지를 동적으로 만들기

### 조작 순서

1. 조작하고자 하는 요소를 선택(또는 탐색 ) 
2. 선택된 요소의 콘텐츠 또는 속성을 조작 

## 선택 메서드 :

- 선택자 활용 위해서 사용해야 하는 메서드
- 선택자(selector) 형태 : 문자열
1. document.querySelector( selector ) : 제공한 선택자와 일치하는 요소 한개 선택 

제공한 css 선택자 만족하는 첫번째 element 객체 반환 
없으면 null    !! 
2. document.querySelectorAll( selector ) : 제공한 선택자와 일치하는 여러 요소(element) 선택 

제공한 css selector 만족하는 NodeList 반환 

```html
<!-- select.html -->

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>

<body>
  <h1 class="heading">DOM 선택</h1>
  <a href="https://www.google.com/">google</a>
  <p class="content">content1</p>
  <p class="content">content2</p>
  <p class="content">content3</p>
  <ul>
    <li>list1</li>
    <li>list2</li>
  </ul>
  <script>
    // print == console.log
    console.log(document.querySelector('.heading'))
    console.log(document.querySelector('.content'))
    console.log(document.querySelectorAll('.content'))
    console.log(document.querySelectorAll('ul > li'))
  </script>
</body>

</html>

```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c29799f5-8249-4957-9f04-85e41405af8f/Untitled.png)

---

# DOM 조작 : 어떤거 조작 가능한지 ?

## 속성(attribute) 조작 :

### 클래스 속성 조작 :

- ‘classList’ 라는 property 조작하겠다 
요소의 클래스 목록을 DOMTokenList(유사 배열) 형태로 반환
- property ? : html에서의 속성과 별개로, DOM API 를 통해 얻어낸 JS에서 조작 가능한 
                   객체가 가지고 있는 값, 속성, 데이터, 정보
1. classList 메서드 :
element.classList.add() : 지정한 클래스 값 추가
element.classList.remove() : 지정한 클래스 값 제거
element.classList.toggle() : 클래스가 존재한다면 제거하고 false 반환
                                                           존재하지 않으면 클래스 추가하고 true 반환

```html
  <script>
    // 속성 요소 조작
    // 클래스 속성 조작
    const h1Tag = document.querySelector('.heading')
    console.log(h1Tag.classList)

    h1Tag.classList.add('red') // 메서드 추가 
    console.log(h1Tag.classList)

    // h1Tag.classList.remove('red') // red 속성 삭제 가능 
    // console.log(h1Tag.classList)

    // h1Tag.classList.toggle('red')
    // console.log(h1Tag.classList)

    // 일반 속성 조작
    const aTag = document.querySelector('a')
    console.log(aTag.getAttribute('href'))

    aTag.setAttribute('href', 'https://www.naver.com/')
    console.log(aTag.getAttribute('href'))

    aTag.removeAttribute('href')
    console.log(aTag.getAttribute('href'))
  </script>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ce294cba-139a-494a-8515-4cef067e139c/Untitled.png)

### 일반 속성 조작 : 바로 위의 소스 / 이미지 참조

1. 일반 속성 조작 메서드 :
Element.getAttribute() : 해당 요소에 지정된 값 반환 (조회)

        name : 조작하고자 하는 속성의 이름  / 해당 속성의 값을 value 로 수정하겠다
       인자 반드시 문자열이라 조금의 한계가 있음 
Element.setAttribute(name, value) : 지정된 요소의 속성 값 설정 
                                                         속성이 이미 있으면, 기존 값 갱신 
                                                         ( 그렇지 않으면, 저장된 이름과 값으로 새 속성 추가 ) 

Element.removeAttribute() : 요소에서 지정된 이름 가진 속성 제거 

### [ 클래스 vs 일반 ] 속성 조작 메서드 ? :

- setAttribute : HTML 속성 자체를 수정

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/978709c4-5e1d-45bd-b03b-ca4cfe38c2c8/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c1af912c-89db-42ec-8a77-7ba80ecfc097/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0493bdbd-e054-4b2c-80ac-6405726ba922/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8a629dbf-174e-4c00-88dd-55d04fe40f9c/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e30254aa-1bbd-46e8-96c8-acf8dfa0ff14/Untitled.png)

- othertype : 속성 수정이 아닌 property 에 값 추가

## HTML 콘텐츠 조작

- ‘textContent’ property : 요소의 텍스트 콘텐츠를 표현 
<p>lorem</p>

```html
  <script>
    // HTML 콘텐츠 조작
    const h1Tag = document.querySelector('.heading')
    console.log(h1Tag.textContent)

    h1Tag.textContent = '내용 수정'
    console.log(h1Tag.textContent)
  </script>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d87f9106-1657-4f8f-8633-850a9345f606/Untitled.png)

## DOM 요소 조작?

- html 안에 직접적으로 작성해놓은 내용뿐 아니라, element 자체를 생성하는것도 가능
    
    DOM 요소 조작 메서드 
    1. document.createElement(tagName) : 작성한 tagName 의 HTML 요소를 생성해 반환
                                       생성한 tagName 도 가능 ( 문자열 )  
    2. Node.appendChild() : 한 Node를 특정 부모 Node 의 자식 NodeList 중 마지막 자식으로 삽입
                                           추가된 Node 객체를 반환
    3. Node.removeChild() : DOM 에서 자식 Node를 제거 / 제거된 Node 반환 
    

```html
<!-- dom-manipulation.html -->

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>

<body>
  <div>
    <p>DOM 요소 조작</p>
  </div>

  <script>
    // 생성
    const h1Tag = document.createElement('h1')
    h1Tag.textContent = '제목'
    console.log(h1Tag)

    // 추가
    const divTag = document.querySelector('div')
    divTag.appendChild(h1Tag)
    console.log(divTag)

    // 삭제
    const pTag = document.querySelector('p')
    divTag.removeChild(pTag)
  </script>
</body>

</html>

```

## style 조작

- ‘style’ property : 해당 요소의 모든 style 속성 목록을 포함하는 속성

```html
<!-- style-property.html -->

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>

<body>
  <p>Lorem, ipsum dolor.</p>

  <script>
    const pTag = document.querySelector('p')

    pTag.style.color = 'crimson'
    pTag.style.fontSize = '2rem' // css 의 font-size와 다르다 
    pTag.style.border = '1px solid black'

    console.log(pTag.style)
  </script>
</body>

</html>

```

+@ 참고 

---

# 이벤트 :

- 일상속의 이벤트 : 키보드 눌러 txt 입력 / 전화 울리는것 / 손흔들어 인사 / 버튼눌러 통화 / …
- 웹에서 의 이벤트 : 스크롤 / 버튼 클릭시 팝업 / 커서 위치에 따라 드래그, 드롭 
웹에서의 모든 동작은 이벤트 발생과 함께 한다

## event 객체 : DOM에서 이벤트 발생했을 때에 생성되는 객체

- event : 무언가 일어났다는 신호 / 사건
- 모든 DOM 요소는 이러한 event 만들어 낸다
- 이벤트 종류 : mouse / input / keyboard / touch …

## event handler : 이벤트 처리기

- DOM 요소는 event 받고, 받은 event ‘처리’ 할 수 있음
- 이벤트 발생했을 때, 실행되는 함수 
사용자의 행동에 어떻게 반응할지를 JS 코드로 표현한 것
    - .addEventListener() : 대표적 이벤트 핸들러 중 하나
    특정 이벤트를 DOM 요소가 수신할 때마다 콜백 함수를 호출 
    
    EventTarget.addEventListener(type, handler) 
      DOM요소 . method ( 수신할 이벤트, 콜백 함수 ) 
        대상에             특정 이벤트 발생 시, 지정한 이벤트 받아 할 일 등록한다. 
    
    type : 수신할 이벤트 이름 / 문자열 작성 
    handler : 발생할 이벤트 객체 수신하는 콜백 함수 
                  콜백 함수는 발생한 event object 를 유일한 매개변수로 받음
- 활용 예시

```html
  <script>
    // 1. 버튼 선택
    const btn = document.querySelector('#btn') 

    // 2. 콜백 함수
    // 버튼 눌리면 실행, 
    const detectClick = function (event) { // 함수 표현식 : 변수 함수명 = function(인자)
      console.log(event) // PointerEvent
      console.log(event.currentTarget) // <button id="btn">버튼</button>
      console.log(this) // ==python self  <button id="btn">버튼</button>
    }

    // 3. 버튼에 이벤트 핸들러를 부착
    btn.addEventListener('click', detectClick) // click 받으면 detectclick 
  </script>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/18d4c016-a99a-49c9-896c-469c61932440/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/385ae837-8d7a-41be-944c-8e6bdf3be225/Untitled.png)

- addEventListener 콜백 함수 특징 : 
1. 발생한 이벤트 나타내는 event 객체를 유일한 매개변수로
2. 반환값 없음

## 버블링 :

- 한 요소에 이벤트 발생 > 할당된 핸들러 동작 > 이어 **부모 요소의 핸들러 동작**하는 현상
- 가장 최상단의 **조상 요소(document) 만날 떄까지 이 과정 반복** > 요소 각각의 핸들러 동작
- 이벤트가 제일 깊은 곳의 요소 ~> 부모 요소 ~> 발생하는 것이 물속 거품과 닮아서

- form > div > p 형태의 중첩된 구조에 각각 이벤트 핸들러 있을 때, <p> 요소 클릭하면 ?

```html
  <script>
    const formElement = document.querySelector('#form')
    const divElement = document.querySelector('#div')
    const pElement = document.querySelector('#p')

    const clickHandler1 = function (event) {
      console.log('form이 클릭되었습니다.')
    }
    const clickHandler2 = function (event) {
      console.log('div가 클릭되었습니다.')
    }
    const clickHandler3 = function (event) {
      console.log('p가 클릭되었습니다.')
    }

    formElement.addEventListener('click', clickHandler1)
    divElement.addEventListener('click', clickHandler2)
    pElement.addEventListener('click', clickHandler3)
  </script>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/436d6246-7c01-4dcf-ab67-1cabdc851e89/Untitled.png)

- 이벤트가 정확히 어디서 발생했는지 접근할 수 있는 방법 ? :
1. event.currentTarget() : 현재 요소 = this 
                                    항상 이벤트 핸들러가 연결된 요소만을 참조하는 속성
2. event.target() : 이벤트 발생한 가장 안쪽의 요소 참조 
                         실제 이벤트 시작된 요소 
                         버블링 진행되어도 변하지 않음 

```html
  <script>
    const outerOuterElement = document.querySelector('#outerouter')
    const outerElement = document.querySelector('#outer')
    const innerElement = document.querySelector('#inner')

    const clickHandler = function (event) {
      console.log('currentTarget:', event.currentTarget.id)
      console.log('target:', event.target.id)
    }

    outerOuterElement.addEventListener('click', clickHandler)
  </script>
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c8b91381-a904-4855-bc3e-c7dbf5f688cd/Untitled.png)

currentTarget : 버블링 끝난 요소 ( 맨 밖 ) 

target : 내가 클릭한 요소 

- 버블링 필요 이유 ? : 
1. 각자 다른 동작 수행하는 버튼 여러개 있을 때에, 버튼마다 서로 다른 이벤트 핸들러 할당 ? 
              ***   > no . 모든 버튼의 공통 조상인 div 요소에 이벤트 핸들러 단 하나만 할당!
                                하위에서 event 발생하면 버블링으로 최상위 div에서 알 수 있으므로 
                                target 사용해서 어느 버튼인지도 알 수 있다

### 캡쳐링 :

- 이벤트가 하위 요소로 전파되는 단계
- 버블링과 반대
- 최상위 객체 ~> 하위 객체

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/70fb734a-190a-450e-8908-02ae6c3bc34f/Untitled.png)

- 실제 다루는 경우 거의 x … 버블링 집중

## event handler 활용

### click 이벤트 실습 :

- 버튼 클릭하면 숫자 1씩 증가
- <button id="btn">버튼</button>
<p>클릭횟수 : <span id="counter">0</span></p>

```html
  <script>
    let counterNumber = 0// 1. 초기값 할당 // let : 재할당 가능 const : 재할당 x 
    const btn = document.querySelector('#btn')// 버튼 선택 ( 위에 id 설정해놓음 ) 
    const clickHandler = function () { // 콜백 함수 (버튼 클릭 이벤트 발생 > 실행
      counterNumber += 1// 3.1 초기값 += 1

      // 3.2 p 요소를 선택 / span 객체 할당 
      const spanTag = document.querySelector('#counter') // span 에 id counter #으로
      // ** spanTag const 인데 어떻게 숫자 재할당해?
			// span 객체는 재할당 불가능. BUT 해당 요소의 property 조작 가능       
      spanTag.textContent = counterNumber // 3.3 p 요소의 컨텐츠를 1 증가한 초기값 설정
    }
    
    btn.addEventListener('click', clickHandler)// 4. 버튼에 이벤트 핸들러 부착 (클릭)
  </script>
```

### input 이벤트 실습 :

- .currentTarget() 주의사항 : 
console.log()로 event 객체 출력시, currentTarget 키값이 null을 가진다. 
currentTarget 은 이벤트 처리되는 동안에만 사용 가능하기 떄문에
대신해서 console.log(event.currentTarget)를 사용해 콘솔 확인 가능 
currentTarget 이후의 속성 값들 ? : target 참고해서 사용
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f90917e5-abb4-4dc5-889a-e863818c7a98/Untitled.png)
    

- 사용자의 입력 값 실시간으로 출력

```html
  <input type="text" id="text-input">
  <p></p>

  <script>
    // 1. input 요소 선택
    const inputTag = document.querySelector('#text-input') // input 참조 

    const pTag = document.querySelector('p')    // 2. p 요소 선택

    // 3. 콜백 함수 (input 요소에 input 이벤트가 발생할때마다 실행할 코드)
    const inputHandler = function (event) {
      // 3.1 작성하는 데이터가 어디에 누적되고 있는지 찾기
      console.log(event.currentTarget.value)

      // 3.2 p요소의 컨텐츠에 작성하는 데이터를 추가
      pTag.textContent = event.currentTarget.value
    }

    // 4. input 요소에 이벤트 핸들러 부착 (input 이벤트)
    inputTag.addEventListener('input', inputHandler)
  </script>
```

## 이벤트 기본 동작 취소

- HTML 의 각 요소가 기본적으로 가지고 있는 이벤트가 떄로는 방해가 되는 경우가 있어 
이벤트의 기본 동작을 취소할 필요가 있음 

ex> form 요소의 제출 이벤트를 취소 > 페이지 새로고침 막을 수 있음
       a 요소 클릭 시 페이지 이동을 막고, 추가 로직 수행 가능

- 드래그 시 보통은 긁어짐 / 특정 블로그나 카페 > X 
touchstart event listeners / remove 하면 복사 가능
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/13646ba0-8c90-49de-86a8-b52307849e20/Untitled.png)
    

- .preventDefault() : 해당 이벤트에 대한 기본 동작 실행하지 않도록 지정

```html
  <h1>중요한 내용</h1>

  <form id="my-form">
    <input type="text" name="username">
    <button type="submit">Submit</button>
  </form>

  <script>
    // 1
    const h1Tag = document.querySelector('h1')

    h1Tag.addEventListener('copy', function (event) {
      console.log(event)
      event.preventDefault()
      alert('복사 할 수 없습니다.')
    })

    // 2
    const formTag = document.querySelector('#my-form')

    const handleSubmit = function (event) {
      event.preventDefault()
    }

    formTag.addEventListener('submit', handleSubmit)
  </script>
```

---

# JS Dtype :

## JS 역사 :

- ECMAScript? : 스크립트 언어가 준수해야 하는 규칙, 세부사항 등을 정의 / JS의 표준 
                       이 명세를 기반으로 하여 웹 브라우저 / Node.js 환경에서 실행됨 ( JS가 )
- 우리가 배우는 JS : ES6+
- 기존 JS : 브라우서에서만 웹 페이지의 동적인 기능 구현 
이후 JS : Node.js 통해 브라우저에서 벗어나 서버, 클라이언트 사이드 등 
              다양한 프레임워크와 라이브러리들이 개발 > 웹 개발에서 필수적인 언어로

## 변수

- 변수명 작성 규칙 : 
1. 대소문자 구분
2. 예약어 사용 불가
3. 변수명 **문자 / $ / _ 로 시작**
- Naming Case : 
1. **변수 / 객체 / 함수 : camelCase <> 파이썬(SNAKE)**
2. 클래스 / 생성자 : PascalCase
3. 상수 : SNAKE_CASE
- 변수 선언 키워드 : 
1. let : 재할당 가능 / **재선언 불가**
2. const : **선언 시 초기값 필수** / 재할당, 재선언 불가

- 기본적으로 const 사용 권장 
- 재할당 필요하다면, 그때 let으로 변경해서 사용!
- 블록 스코프 : 
if, for, 함수 등의 중괄호 {} 내부를 가리킴 
블록 스코프 변수 > 블록 바깥에서 접근 불가

## Dtype

- 원시 자료형 (Primitive)
- number / string / boolean / **null / undefined** 
- 변수에 값이 직접 저장 ( 불변 & 값이 복사 ) 

cf> NaN : Not a Number ( 문자열과 숫자 더했을 때에 에러 X  →  NaN 리턴 ) 

- String :
 ‘+’ 연산자 통해 결합 가능 / 이외의 연산자는 불가 
 fstring과 비슷하게 `이런 ${str} 식으로` 표현 가능 ‘ 이 아닌 ` ( backtick )  

- null : 변수의 값이 없음을 의도적으로 표현 ( == Python None ) 
- undefined : 변수 선언 이후 직접 값 할당하지 않으면 자동으로 할당 < 사용 X 
- boolean : true / false ( **소문자**!!! )
- 참조 자료형 (Reference)
- objects ( object / array / func ) 
- 객체의 주소가 저장 ( 가변 & 주소가 할당 )

## 연산자

- 할당/논리 연산자 : 파이썬과 동일
- 증감 연산자 : 파이썬과 차이? → ++ — 가능 but 사용 ㄴㄴ
- 비교 연산자 중 일치 연산자 : == 아닌 **=== 사용**

## 제어문

- 조건문 : 
if (조건) {         }   /   else if  빼고 동일

cf> 삼항 연산자? : 
condition ? expression1 : expression2 
조건 맞으면 exp1 / 틀리면 exp2 반환

```python
const age = 20 
const message = (age >= 18 ) ? '성인' : '미성년자'
```

- 반복문 : 
1. while() {} : let 변수 통해서 횟수만큼 돌리기 

2. for : 특정한 조건이 거짓으로 판별될때까지 반복 

for ( [ 초기문 ] ; [ 조건문 ] ; [ 증감문 ] ) {}       → C와 동일 
초기문 : 뒤의 2개에서 사용할 변수 선언
조건문 : 해당 변수에 대한 조건이 true 인 동안 실행 
증감문 : ++ — 사용 

3. for…in : **객체만 사용!!!** / 객체의 열거 가능한 속성에 대해 반복
for (variable in object) { statement } =⇒ 객체의 키값 꺼내줌

4. for…of : **iterable 한 객체 ( 배열 , 문자열** 등…) 대해 반복 
for (variable in iterable) { statement } =⇒ 객체의 요소 꺼내줌 = 파이썬

for in / of 문에서의 variable : 
**const로 선언**해도 괜찮음 - 한번 사이클 돌때마다 변수 수명주기가 끝나는 식이라서

## 함수 :

### 함수 정의 :

- 참조 자료형에 속하며, 모든 함수는 Function object 
== 객체의 주소가 저장되는 자료형

- 함수 구조 : return 없으면 undefined

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d24c871d-2d61-4aac-83e5-9b1c7e2d2358/Untitled.png)

- 함수 정의 3가지 방법 : 
1. 선언식 : 기존 방식 ( 위와 동일 ) 
2. 표현식 : const funcName = 선언식 ( 에서 **함수명만 빼고** ) 
  ㄴ 어디에 사용????? ( 사용 권장 ) 
함수 이름이 없는 익명 함수 사용 가능 
선언식과 달리 표현식으로 정의한 함수 : 호이스팅 X > 함수 정의 이전에 사용 불가 
호이스팅 : ??? 

3.

### 매개변수 :

- 정의 방법 : 
1. 기본 함수 매개변수 :  function (**name = ‘asdf’**) {} 
- 전달하는 인자 없거나 undefined 전달 → 이름 붙은 매개변수를 기본값으로 초기화 

2. 나머지 매개변수 : function ( p1, p2, …restPrams ) {}
- 파이썬에서의 * 
- 임의의 수의 인자를 배열로 허용하여 가변 인자 나타냄 
- **함수 정의 시 하나**만
- **매개변수 마지막**에 위치해야

- 매개변수 & 인자 개수 불일치 시 : 
- 매개변수 개수 > 인자 개수 : **누락된 인자 undefined 로 할당 ( 오류 X** ) 
- 매개변수 개수 < 인자 개수 : 초과 입력한 인자 사용 X ( 오류 X )

### Spread syntax :  …

- 전개 구문 : 배열이나 문자열과 같이 **반복 가능한 항목 펼치는 것** ( 확장, 전개 )
- 전개 대상에 따라 역할이 다름 
- 배열이나 객체의 요소 **개별적인 값으로 분리**하거나, 
- 다른 배열이나 객체의 요소를 **현재 배열이나 객체에 추가**하는 등
- 전개 구문 활용처 : 
1. 함수와의 사용 : 함수 호출 시 인자 확장 / 나머지 매개변수 ( 압축 ) 
         ㄴ 아래 예시 참고 

2. 객체와의 사용 ( 객체 파트에서 ) 
3. 배열과의 활용 ( 배열 파트에서 )

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8754626b-4473-488d-9515-266bc46a50fe/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/50cc414c-1dae-4e4e-bbb8-7378db430802/Untitled.png)

### 화살표 함수 :

- 함수 표현식의 간결한 표현법

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3d5d8f3c-dc36-496d-b286-c1ccc8d1b9f1/Untitled.png)

- 작성 과정
1. **func 키워드 제거 후 매개변수 / 중괄호 사이에 화살표 작성** 
2. 매개변수 only 한개 ⇒ 매개변수의 ‘ ( ) ‘ 제거 가능  
3. 본문의 표현식 한줄 ⇒ ‘{ }’ , ‘return’ 제거 가능 

### 참고 :

- 세미콜론 : 선택적으로 사용 가능 ( 없으면 자동으로 넣어줌 )

- 호이스팅 : 
변수를 선언 이전에 참조할 수 있는 현상
**변수 선언 이전의 위치에서 접근 시 undefined 반환**

## 객체 :

### 구조 및 속성 :

- **키로 구분된 데이터 집합** 저장하는 자료형 **≠ dict**

- 객체 구조 : 
- {} 이용해 작성 
- k, v 쌍으로 구성된 속성 여러개 작성 가능 
- **key는 문자형만** / value 는 모든 자료형 허용 
   ㄴ 문자열 ( 공백 포함되었을 때에 )   /   문자 형 : “” 안묶고 문자로만 이뤄졌을 때에
- dict 아니기 때문에, 메서드 가질 수 있음
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2136cb55-71d2-4a80-97fb-10ec919e3f09/Untitled.png)
    

- 속성 참조 : 
- ‘ . ‘ OR ‘ [ ] ‘ 통해 객체 요소 접근 
- key 이름에 ‘ ‘ 같은 구분자 있으면, 대괄호 접근만 가능
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/56a9a8b8-1048-4ed3-b4b2-7e3a05dff5de/Untitled.png)
    

- in 연산자 : 
’greeting’ in user  <  속성이 객체에 존재하는지 여부 확인

### 객체 & 함수 :

- Method  : 객체에 정의한 함수 / 메서드는 객체를 행동할 수 있게 함 
                 사용 예시 : object.method() 

                this 키워드 사용해 객체에 대한 특정한 작업 수행 가능
- this
- this 키워드 : 함수나 메서드 **호출한** 객체 가리키는 키워드
                    함수 내에서 객체의 속성 및 메서드에 접근하기 위해 사용 

func 의 이름이 없지만, 호출한 객체인 person 의 name을 가져오는 모습
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4be06075-df98-4e4b-93fd-5bb084efeb99/Untitled.png)
    

- 단순 호출 시 this : 가리키는 대상 → 전역 객체 window
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f5f288e6-4237-4aed-a36a-cc451feb4d53/Untitled.png)
    

- 메서드 호출 시 this : 가리키는 대상 → 메서드 호출한 객체
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d6d987a0-e6b9-44ce-b490-b064677fd599/Untitled.png)
    

- 중첩된 함수에서의 this 문제점 & 이에 대한 해결책 :
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/385ab67b-f242-4ab4-a44b-1ec3dc4af438/Untitled.png)
    

- 추가 객체 문법 :
1. 단축 속성 : **k, v의 이름이 동일**한 경우, 단축 구문 사용 가능 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/89cb9c97-5dfe-4da6-ae5f-a1d58582d263/Untitled.png)

1. func 생략 가능 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8b42eac0-84a2-40b0-a3f2-cab2c22bb385/Untitled.png)

1. 계산된 속성? : 키가 대괄호로 둘러싸여 있는 속성 - 고정된 값 아닌 변수값 사용 가능 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a46383b2-2df0-4050-9d72-ce777d995b69/Untitled.png)

1. 구조 분해 할당 : 배열/객체 분해해서 객체 속성을 변수에 쉽게 할당할 수 있는 문법 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/636975b0-69ea-4ab1-8a71-7612163b8735/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3754e5b3-8630-43ae-bc35-867882128c87/Untitled.png)

1. 전개 구문 : 객체 복사 ( 객체 내부에서 객체 전개 ) / 얕은 복사에 활용 가능 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/650b233e-e976-448f-9532-2a86ced9b01d/Untitled.png)

1. 유용한 객체 메서드 : 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2c160a2c-4277-4fb6-befb-9f68a4635fc0/Untitled.png)

### JSON :

- JS Object Notation
- **k, v**로 이뤄진 자료 표기법
- JS의 Object 와 유사한 구조 가지고 있지만, **JSON은 형식이 있는 문자열**
- **JS에서 JSON 사용 위해서는, Object 자료형으로 변경해야**

- Object <> JSON 변환 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ac388ccb-ed6f-4a11-adfe-67924468935a/Untitled.png)

### NEW 연산자 :

- JS에서 객체 하나 생성한다면? : 하나의 객체 선언하여 생성
- 동일한 형태의 객체 또 생성한다면? : 또다른 객체 선언해서 생성해야 하는데, 불편
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8bce0d2d-28fb-42b9-8a58-a611eafec8b4/Untitled.png)
    

## 배열 :

- Array : 순서가 있는 Object ( data collection ) 데이터 집합 저장하는 자료구조
- 리스트와 거의 동일
- .length 통해서 요소 개수 확인 가능
- idx 접근 가능

### 배열 메서드

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/51dfc314-a7b1-4fb6-acff-b097c1e82478/Untitled.png)

### Array Helper Method :

- 배열 조작을 보다 쉽게 수행할 수 있는 특별한 메서드 모음
- 배열에 요소를 순회하며 요소에 대한 함수 ( 콜백함수 ) 호출
- 메서드 호출 시 인자로 함수 ( 콜백함수 ) 받는 것이 특징

- 콜백 함수? : 다른 함수에 인자로 전달되는 함수 
                    외부 함수 내에서 호출되어 일종의 루틴 / 특정 작업 진행
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/968d582b-6d59-412f-b4e7-88a04c062afc/Untitled.png)
    

- for of 안쓰고 array helper method 쓰면 아래와 같이 한줄에 가능

- forEach : 배열 내에 모든 요소 각각에 대해 콜백함수 호출 / **반환값 없음** 

아래 첫번째 예시 : Alice \ Bella \ Cathy 출력
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/95f0bf6d-5716-4ad4-9a53-94741dbe5039/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ffca182a-7909-450a-a2ff-2be7b5e967ad/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a06f4b29-53e8-4fde-81d4-d1d26af2c8bf/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a395dd81-eba0-40f1-b633-96c8f5696455/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c9fc4ad5-b21b-44d9-b75f-3853c4b54d8b/Untitled.png)
    

- map : 첫번째는 동일 / 함수 **호출 결과** 모아 **새로운 배열 반환**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b64081d5-582f-4422-ab69-95008b54975d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/113179a9-781f-4a41-bce6-18bd42e4c08f/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/81d53b21-f86b-416a-b538-a3cbf97645f6/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/42ba2537-a288-4ce4-80ff-7621f9668a61/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/05534e30-9df4-43f9-bfe2-602f9529b7d5/Untitled.png)

- Python map과 비교 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/de6dbde0-1269-4e05-bbe1-d24f1301eb10/Untitled.png)

- 베열 순회 종합 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/33242508-52ea-4a91-9933-5c0d63251c07/Untitled.png)

- 기타 Array Helper Methods :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e0b632cb-c4a4-4a9e-906b-8754efe489c5/Untitled.png)

### 추가 배열 문법 :

- 전개 구문 :
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0e958e71-f7f8-49ec-9ee7-cebed128bb21/Untitled.png)
    

## 참고 :

### 콜백함수 구조 사용 이유 ?

1. 함수의 재사용성 측면 : 
- 함수를 호출하는 코드에서 콜백 함수의 동작을 자유롭게 변경 가능
- ex> map 함수는 콜백함수 인자로 받아 배열의 각 요소 순회하며 콜백함수 실행
이때, 콜백함수는 각 요소 변환하는 로직 담당 
→ map 함수 호출하는 코드는 간결 & 가독성 +

1. 비동기적 처리 측면 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4bcfd9ff-9b68-4cc1-8869-4624a81dc0b6/Untitled.png)

### forEach 에서 break 하는 대안 :

…

### 배열은 객체다 :

- const 로 배열 선언해 놓으면, const 이지만 배열에 요소 추가같은건 가능
- 배열 = 1 와 같이 하면 에러 발생
- 배열이라는 틀을 const 해주고, 그 안에서 추가 삭제는 가능하게 … ?

---

# Asynchronous JS

## 비동기

- Synchronous (동기) : 프로그램의 실행 흐름이 순차적으로 진행 ( 우리가 하던 기존 코딩 ) 
                                  하나의 작업 완료된 후에 다음 작업 실행되는 방식
                                  손님 2명 있을 때에, 2번째 손님이 첫번째 손님 커피 받을때까지 주문X

- Asynchronous (비동기) : 프로그램 실행 흐름 순차적X 
                                       작업 완료되기를 기다리지 않고 다음 작업 실행 ( 진동벨 ) 
                                       = 작업 완료 여부 신경쓰지 않고 동시에 다른 작업 수행 가능 

                                      특징 : **병렬적 수행** / 당장 처리 완료할 수 없고 시간이 필요한 작업들
                                                별도로 요청을 보낸 뒤 응답이 빨리 오는 작업부터 처리
- 비동기 예시 : - 네이버 로딩 시 창 늦게뜨는 것들 ( 병렬처리 되고 있다는 것 ) 
                      - 코드 : setTimeout 통해서 3초 비동기 대기

```jsx
    const slowRequest = function (callBack) {
      console.log('1. 오래 걸리는 작업 시작 ...')
      **setTimeout**(function () {
        callBack()
      }, 3000)
    }

    const myCallBack = function () {
      console.log('2. 콜백함수 실행됨')
    }

    slowRequest(myCallBack)

    console.log('3. 다른 작업 실행')

    // 출력결과
    // 1. 오래 걸리는 작업 시작 ...
    // 3. 다른 작업 실행
    // 2. 콜백함수 실행됨
```

### JS & 비동기 :

- JS : 싱글 스레드 언어 ( 한번에 하나의 일만 수행 가능 ) 

      스레드란? : 작업 처리할 때 실제로 작업을 수행하는 주체 
                         멀티 스레드라면 업무 수행할 수 있는 주체가 여러 개라는 의미 

      그러면 어떻게 JS가 비동기 처리 가능 ?

- **JS Runtime : JS가 동작할 수 있는 환경** ( Runtime ) 
                    JS 자체는 싱글 스레드이므로 비동기 처리 할 수 있도록 도와주는 환경 필요 
                   **JS에서 비동기와 관련한 작업은 브라우저 / Node** 같은 환경에서 처리

- **브라우저 환경**에서의 JS 비동기 처리 관련 요소 ? 
 - JS 엔진의 Call Stack / Web API / Task Queue / Event Loop 

- Call Stack : 
요청이 들어올 때마다 순차적으로 처리하는 스택 
기본적인 **JS 싱글 스레드의 작업 처리** 

- **Web API** : 
JS 엔진 XXX **브라우저에서 제공하는 런타임 환경** 
시간이 소요되는 작업을 처리 ( **setTimeout / DOM Event / 비동기 요청** 등 ) 

- Task Queue ( Callback Queue ) : 
**비동기 처리된 콜백 함수가 대기**하는 큐 

- Event Loop : 
작업이 들어오길 기다렸다가 들어오면 처리하고, 처리할게 없으면 잠드는 **JS 내 무한루프**
**Call Stack / Task Queue 를 지속적으로 모니터링** 
Call Stack 비어있는지 확인 후 비어있다 ? 
   ㄴ Task Queue에서 대기중인 오래된 작업을 **Call Stack으로 Push**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a6d2af91-4f95-4007-b2cd-b2c622601aa2/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c8655c71-dc53-4b17-aa59-a767a71fa14a/Untitled.png)

- 이미지 설명 : 
- 모든 작업 :                                   Call Stack 들어간 후 처리
- 오래 걸리는 작업 :                       **Call Stack 들어오면 web API 로 보내** 별도 처리
- WEB API에서 처리 끝난 작업들 : 곧바로 스택이 아닌 Task Queue 에 
- Event Loop :                                 Call Stack 비어있는 것 체크 
                                                       비어있다? 가장 오래된 작업 Call Stack으로 보냄

- 정리 
JS는 한번에 하나의 작업을 수행하는 싱글 스레드 언어 / 동기적 처리 진행
브라우저 환경 : WEB API에서 처리된 작업 지속적으로 Task Queue 거쳐 Event Loop 에 의해               
                         Call Stack 에 들어와 순차적으로 실행 > 비동기 작업 가능한 환경 됨

### 용어정리 :

1. 블락 BLOCK : 
동기적 실행에서, A함수가 B함수를 불렀다고 가정했을 때에, 
B함수가 실행중인 시간 ( A함수가 대기하고 있는 시간 ) 
2. 논블락 NON-BLOCK : 
비동기적 실행에서, 함수 내에서 setTimeout 찍고 WEB API로 보낸 그 순간부터 
WEB API에서 타이머 끝나고 Task 큐 > 이벤트 루프 거쳐 돌아와 작업 실행할 때까지의 시간
3. 동시성 : 다수의 작업들 하나의 타임라인에서 동시에 작업이 되는 것처럼 보이게 작업하는 형태 
4. 병렬성 : 다수의 작업들을 물리적으로 동시에 진행하는 것 

>> JS는 병렬성처럼 보이지만, 동시성이다. 

## AJAX ( Asynchronous Javascript And XML )

- XMLHttpRequest 기술 활용해 복잡하고 동적인 웹 페이지 구성하는 프로그래밍 방식

- 정의 : 
- 비동기적인 웹 애플리케이션 개발 위한 기술
- **브라우저 <> 서버 간의 데이터를 비동기적으로 교환**하는 기술
- AJAX 사용 → 페이지 전체를 새로고침 하지 않고도 동적으로 데이터 불러와 화면 갱신 가능 
- AJAX 에서의 X : XML이라는 dtype 의미하지만 
                            **요즘엔 더 가벼운 용량과 JS의 일부라는 장점 때문에 [ JSON ] 많이 사용**

- 목적 : 
전체 페이지가 다시 로드되지 않고, **HTML 페이지 일부 DOM만 업데이트** 
웹 페이지 일부가 다시 로드되는 동안에도 코드가 계속 실행 → 비동기식으로 작업 가능

### XHR ( XMLHttpRequest 객체 ) :

- **서버와 상호작용할 때에 사용하는 객체**

- 특징 : 브라우저와 서버 간의 네트워크 요청 전송 가능 
사용자의 작업 방햐하지 않고 **페이지의 일부 업데이트** 가능 
요청의 상태와 응답 모니터링 가능 
이름에 XML이라는 dtype 들어가지만, XML뿐만 아니라 모든 종류의 데이터 가져올 수 있음

- XHR 구조 : 
HTTP 요청을 생성하고 전송하는 기능을 제공 
AJAX 요청을 통해 서버에서 데이터 가져와 웹 페이지에 동적으로 표시

```jsx
    const xhr = new **XMLHttpRequest**() // **XHR 객체 인스턴스 생성**
    xhr.open('GET', 'https://jsonplaceholder.typicode.com/posts')
    // open( 요청방식, url ) 
    xhr.send() 
    // 요청 전송 > 비동기 

    xhr.**onload** = function () {  // **요청이 완료되었을 때 호출**
    
      // 응답 상태 코드가 200이라면
      if (xhr.status == 200) {
        console.log(xhr.responseText)  // 응답 받은 결과 출력
      } else {
        console.error('Request failed')  // 200 이외 상태에 대한 예외 처리
      }
    }
```

- 기존 기술과의 차이 - 기존 방식 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/789e76db-7526-40f2-8fb3-63da753a567c/Untitled.png)

- form 태그 : 완성된 PAGE 보여줌 → 계속 전체 페이지를 새로고침 → 낭비 심함

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/551aa251-abdf-4893-a44e-34b7c71e0f5a/Untitled.png)

- xhr 객체 → 사용자가 form 태그에 입력한 데이터만 JS에 뽑아와 서버에 요청 보냄
- 서버 : 이에 대한 응답으로 JSON 데이터 응답
- JS 데이터 안에는 JS에서 쓸수있는 객체모양의 데이터 들어있음
- 그중에서 필요한 속성들만 뽑아 DOM 조작하여 그 부분만 랜더링

- 이벤트 핸들러 → 비동기 프로그래밍의 한 형태 ( 비 동기 프로그램 구현할 것 ) 

- 이벤트 발생할 때마다 호출되는 함수 ( 콜백 함수 ) 제공하는 것 
- http 요청은 응답이 올 때 까지의 시간이 걸릴 수 있는 작업이라 비동기,
  이벤트 핸들러를 XHR 객체에 연결해 요청의 진행 상태 및 최종 완료에 대한 응답 받음

## Callback Promise

- 비동기 처리의 단점 : 
- 비동기 처리의 핵심? : 
WEB API 로 들어오는 순서가 아니라, 작업이 완료되는 순서에 따라 처리한다는 것 
→ 코드의 순서가 불명확하다 → 실행 결과 예상하면서 코드 작성할 수 없게 함 
→ 콜백 함수를 사용하자 !
- 비동기 콜백 : ???????
- 비동기적으로 처리되는 작업이 완료되었을 때에, 실행되는 함수
- **연쇄적으로 발생하는 비동기 작업을 순차적으로** 동작할 수 있게 함
→ 작업의 순서와 동작 제어하거나 결과 처리하는 데에 사용

```jsx
    const asyncTask = function (callback) {
      setTimeout(function () {
        console.log('비동기 작업 완료')
        callback() // **작업 완료 후 콜백 호출**
      }, 2000) // 1초 후에 작업 완료
    }
    
    // 비동기 작업 수행 후 콜백 실행 << 이 안의 인자 f가 callback func 
    asyncTask(function () {
      console.log('작업 완료 후 콜백 실행')
    })
```

- 한계? : 
- 비동기 콜백 함수는 보통 어떤 기능의 실행 결과 받아 다른 기능 수행하기 위해 사용됨
- 이 과정 작성하다 보면, 비슷한 패턴 계속 발생 → 콜백 지옥
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c9924357-475b-4db6-bdd2-94ac547c8302/Untitled.png)
    

- 해결? : 콜백 함수 정리 
- 콜백 함수는 비동기 작업 순차적으로 실행할 수 있게 하는 반드시 필요한 로직
- 비동기 코드 작성하다 보면 콜백 함수로 인한 콜백 지옥은 빈번히 나타나는 문제,
→ 이는 코드의 가독성 해치고 유지보수 어려워짐 → Promise 사용

### Promise :

- JS에서 비동기 작업의 결과를 나타내는 객체
비동기 작업 완료되었을 때에 결과값을 반환하거나 실패 시 에러 처리할 수 있는 기능 제공

- Promise object : 
JS에서 비동기 작업 처리하기 위한 객체
비동기 작업의 성공 / 실패에 관련된 결과 / 값 나타냄
콜백 지옥 문제 해결 위해서 등장한 비동기 처리위한 객체
작업 끝나면 실행시켜 줄게 → 라는 약속

- then() : 성공에 대한 약속

- catch() : 실패에 대한 약속

```jsx
    // promise.html
    const fetchData = () => {
      return new Promise((resolve, reject) => {
          const xhr = new XMLHttpRequest()
          xhr.open('GET', 'https://api.thecatapi.com/v1/images/search')
          xhr.send()
      })
    }
    const promiseObj = fetchData()
    console.log(promiseObj) // Promise object
```

- 비동기 콜백 vs Promise :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c0cff56a-a9a8-4224-b0fe-0ed0ac58cfce/Untitled.png)

- then 메서드 chaning : 

1. 목적? : 
비동기 작업의 순차적인 처리 가능 
코드를 보다 직관적이고 가독성 좋게 작성할 수 있도록 도움 

2. 장점? : 
- 가독성 ( 비동기 작업의 순서, 의존 관계를 명확히 표현할  수 있어 코드 가독성 +
- 에러 처리 : 각각의 비동기 작업 단계에서 발생하는 에러 분할해서 처리 가능
- 유연성 : 각 단계마다 필요한 데이터 가공 / 다른 비동기 작업 수행 가능 
                → 더 복잡한 비동기 흐름 구성 가능 
- 코드 관리 : 비동기 작업 분리하여 구성 → 코드 관리 용이

- Promise 가 보장하는 것 ( vs 비동기 콜백 ) 

1. 콜백 함수는 JS의 Event Loop가 현재 실행 중인 Call Stack 완료 이전에는 절대 호출 XXX
    반면, **Promise callback 함수**는 Event Queue 에배치되는 엄격한 순서로 호출됨 

2. 비동기 작업 성공하거나 실패한 뒤에 then 메서드 이용해 추가한 경우에도 
    호출 순서 보장하며 동작 

3. then을 여러 번 사요ㅕㅇ하여 여러 개의 callback 함수 추가 가능 
ㄴ 각각의 콜백은 주어진 순서대로 하나하나 실행하게 됨
ㄴ chaning은 Promise의 가장 뛰어난 장점

## Axios

- JS에서 사용되는 HTTP 클라이언트 라이브러리

- 정의 : 
클라이언트 및 서버 사이에 HTTP 요청 만들고, 응답 처리하는 데에 사용되는 JS 라이브러리
서버와의 HTTP 요청과 응답 간편하게 처리할 수 있도록 도와주는 도구
브라우저 위한 XHR 객체 생성
간편한 API 제공 / Promise 기반의 비동기 요청 처리
주로 웹 app에서 서버와 통신할 때에 사용

- AJAX 활용한 C - S 간 동작 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0cc15cfd-5e29-4151-b786-c479ec9f4c6f/Untitled.png)

- 사용 : CDN 방식으로 사용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9eb87676-70a2-4e6d-a096-e1181fc5bbe3/Untitled.png)

```jsx
  <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
  <script>
  
    // 1. XHR 객체와 동일한 요소(객체) 주고, axios 객체 생성 
    const promiseObj = axios({
      method: 'get',
      url: 'https://api.thecatapi.com/v1/images/search'
    })

    console.log(promiseObj) // Promise object 출력  
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4d42bc26-3512-414e-b37c-7026a9844d54/Untitled.png)

```jsx
    // promise 객체라서 then 메서드 사용 가능 
    promiseObj.then((response) => {
      console.log(response) // Response object
      console.log(response.data)  // Response data
    })
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ca9836dd-27f2-40b3-9220-5c0f922b7359/Untitled.png)

```jsx
    // 2. 표현법만 다르고, 바로 위의 코드와 동일한 결과 출력 
    axios({
      method: 'get',
      url: 'https://api.thecatapi.com/v1/images/search'
    })
      .then((response) => {
        console.log(response)
        console.log(response.data)
      })
```

```jsx
    axios({
      method :'post',
      url : '/user/1234/'
    })
    .then((response) => {
      console.log(response)
    })
    .catch((error) => {
      console.log(error)
    })
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6eecc296-5622-405b-a2d4-ead5ad369484/Untitled.png)

- 위 코드는 당연히 틀리는 코드. catch로 인해 실패사유 / 오류정보 등 정보 줌

```
// 아래와 같이 응용 가능 
    axios({
      method :'post',
      url : '/user/1234/',
      headers : {
        'Authorization': `Token #{api_key}`
      },
      data : {
        firstName : '',
        lastName: '',
      }
    })
    .then((response) => {
      console.log(response)
    })
    .catch((error) => {
      console.log(error)
    })
```

- then & catch의 chaining 
axios로 처리한 비동기 로직은 항상 promise 객체 반환
즉, then과 catch는 모두 항상 promise 객체 반환 ( 계속 chaining 가능 ) 
then 을 계속 이어 나가면서 작성할 수 있게 됨
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f79b0ec3-6b28-42ed-9f9a-8f2bd1805eb1/Untitled.png)
    

- 예시

```jsx
  <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
  <script>
    URL = 'https://api.thecatapi.com/v1/images/search'
    axios({
      method: 'get',
      url: URL,
    })
      .then((response) => {
        console.log(response)
        console.log(response.data)
      })
      .catch((error) => {
        console.log(error)
        console.log('실패했다옹')
      })
    console.log('야옹야옹')
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1109c83f-4740-4b78-b4ec-18b0d971e0d6/Untitled.png)

```jsx

// 버튼 누르면 click 감지되어서 getCats 콜백 함수 실행 
// axios에게 요청 보내고, 응답 결과에 따라 서로 다른 작업 하도록 

<!-- cat-api-ad.html -->

<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Document</title>
</head>

<body>
  <button>냥냥펀치</button>

  <script src="https://cdn.jsdelivr.net/npm/axios/dist/axios.min.js"></script>
  <script>
    const URL = 'https://api.thecatapi.com/v1/images/search'
    const btn = document.querySelector('button')

    const getCats = function () {
      axios({
        method: 'get',
        url: URL
      })
        .then((response) => {
        // response.data 확인해보면 이미지 url 가지고 있는 형태 / 해당 요소 받아옴
          const imgUrl = response.data[0].url
          return imgUrl
        })
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ad0d66fd-f8c7-491d-878d-dd36918017d6/Untitled.png)

```jsx
        .then((imgData) => {
          const imgElem = document.createElement('img')
          imgElem.setAttribute('src', imgData)
          document.body.appendChild(imgElem)
        })
        // 바로 위 then에서 응답 받은 데이터 
        // img 태그 imgElem 생성 
        // -> setAttribute 통해 src 영역에 넘겨받은 데이터 넣어줌
        // document.body 에 이미지 appendChild 통해서 붙이기 
        // 결과 : 버튼 클릭할때마다 이미지 붙여짐 
        
        .catch((error) => {
          console.log(error)
          console.log('실패했다옹')
        })
      console.log('야옹야옹')
    }

    btn.addEventListener('click', getCats)
  </script>
</body>

</html>
```

- 정리 : 
AJAX - 하나의 특정 기술 의미 X / 비동기적 웹APP 개발에 사용하는 기술들 묶어 지칭 
Axios - CS 사이에 HTTP 요청 만들고 응답처리에 사용하는 JS 라이브러리 (Promise API 지원) 

**FE서 Axios 활용해 DRF 로 만든 API 서버로 요청 보내 데이터 받아오고 처리**하는 로직 작성

---

# Django

- WEB Application : 장고 통해서 api 서버 생성 목표
- WEB Application (= web service) 개발 : 
인터넷을 통해 사용자에게 제공되는 sw 프로그램 구축 과정
다양한 디바이스에서 웹 브라우저 통해 접근하고 사용 가능

## 클라이언트 & 서버 :

### 웹의 동작 방식 :

- 우리가 컴퓨터, 모바일 기기로 웹 페이지 보게될 떄까지 무슨 일이 발생 ?

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a33ec2f7-1534-4181-9ee2-533b9f57b3f1/Untitled.png)

- Client : 서비스를 요청하는 주체 ( 웹 사용자의 **인터넷이 연결된 장치** / **웹 브라우저** )
- Server : 클라이언트의 요청에 응답하는 주체 ( **웹 페이지**, **앱을 저장하는 컴퓨터** )

### 우리가 웹 페이지 보게 되는 과정 :

1. **웹 브라우저 ( 클라이언트 )** 에서 ‘google.com’ 주소 입력
2. 브라우저는 인터넷에 연결된 **전세계 어딘가에 있는 구글 컴퓨터 ( 서버 )** 에게
‘Google 홈페이지.**html’ 파일을 달라**고 요청
3. 요청을 받은 구글 컴퓨터는 **DB**에서 ‘Google 홈페이지.html’ 파일을 찾아 응답
4. 전달받은 ‘Google 홈페이지.html’ 파일을 사람이 볼 수 있도록
**웹 브라우저가 해석(파싱)**해주면서 사용자는 구글의 메인 페이지를 보게 됨 

### 웹 개발에서의 FE & BE :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/02ce9d7a-acdf-41e1-9d32-45e8c87ac3c6/Untitled.png)

- FE : UI를 구성하고, **사용자가 APP과 상호작용**할 수 있도록 함 
**HTML / CSS / JS / FE 프레임워크** 등
- BE : 서버 측에서 동작하며, **클라이언트의 요청에 대한 처리와 DB와의 상호작용** 담당 
서버 언어 ( Python, JAVA ) / **BE 프레임워크 / DB / API / 보안** 등

### 웹 서비스 개발에 필요한 요소 ? :

- 로그인 / 로그아웃 / 회원관리 / DB / 보안 … 너무 많은 기술 필요
- 모든걸 개발자가 하는건 현실적으로 어려우니, 잘 만들어진 것들을 가져와 사용

---

- 모듈 / 패키지 / 라이브러리 ?
1. 모듈 : 파이썬 파일 하나 ( 클래스, 함수 ) 

2. 패키지 : 연관된 모듈들 하나의 dir에 모아놓은 것 // [ 파이썬 파일 여러개 모여있는거 ]
from package import module
from package.module import func(혹은 변수) 

3. 라이브러리 : 패키지들로 이뤄진 기능 하나

cf> 프레임워크 : 특정한 기능 개발을 위해 모아놓은 틀  >  아예 다른 느낌 
                           라이브러리 + 개발도구 + 규칙 

---

# Web Framework :

- 웹 APP 을 빠르게 개발할 수 있도록 도와주는 도구
- 개발에 필요한 기본 구조 / 규칙 / 라이브러리 등 제공
- ex> bootstrap

## Django Framework :

- Python 기반의 대표적인 Web Framework
- 왜 장고? : 
1. 다양성 - Python 기반으로 소셜 미디어 및 빅데이터 관리 등 광범위한 서비스 개발에 적합
2. 확장성 - 대량의 데이터에 대해 바르고 유연하게 확장할 수 있는 기능 제공
3. 보안 - 취약점으로부터 보호하는 보안 기능이 기본적으로 내장되어 있음
4. 커뮤니티 지원 - 개발자를 위한 지원, 문서 및 업데이트를 제공하는 활성화 된 커뮤니티
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0793999d-5cd8-4857-a42c-f246ee0b8127/Untitled.png)
    
- **장고를 사용해서 API 서버를 구현**할 것

## 가상 환경 :

- Python APP과 그에 따른 패키지들을 격리하여 관리할 수 있는 **독립적인** 실행 환경

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f634d636-998c-4c55-b8e2-056bd4ce6471/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/512308b9-1714-4197-9e26-cfb6f291308e/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/673b3114-6714-45b5-9cdb-c4b02230c476/Untitled.png)

```bash
python -m venv venv # 가상 환경 venv 생성
source venv/Scripts/activate # 가상 환경 활성화 
pip list # 환경에 설치된 패키지 목록 확인 
deactivate # 가상환경 비활성화 
```

- 아래와 같이 bash 명령 입력한 곳에 venv 폴더 생기고, 
source 명령 통해서 해당 가상환경의 파이썬 가상환경 activate 시키면
입력하는 곳이 (venv) 가 되어 있는 것을 확인할 수 있음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1f6cd2ff-1add-42cd-8df6-9d658b6b7a26/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c07cd936-c553-450c-a6f9-71332bd12519/Untitled.png)

- + 설치되어 있는 라이브러리 없는 것도 확인 가능

### 패키지 목록이 필요한 이유 ?

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c14b47d0-cfc5-415f-8713-fbbbd477994e/Untitled.png)

### 의존성 패키지 :

- 한 sw 패키지가 다른 패키지의 기능이나 코드를 사용하기 떄문에, 
그 패키지가 존재해야만 제대로 작동하는 관계
- 사용하려는 패키지가 설치되지 않았거나, 호환되는 버전이 아니면 
오류가 발생하거나 예상치 못한 동작을 보일 수 있음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/17f636c8-1f9e-490f-aae4-ec9345a4c844/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3631fe08-5733-4b1c-bb7d-dfb496382f5b/Untitled.png)

- requests 설치 후 추가로 설치되는 패키지 목록 변화
- notebook 설치 이후에 의존성 패키지가 쫘르륵 설치되는 것도 확인 가능

### 의존성 패키지 관리의 중요성 ?

- 개발 환경에서는 각각의 pjt가 사용하는 패키지, 버전을 정확히 관리하는 것이 중요

       >>>> 가상 환경 & 의존성 패키지 관리 

```bash
pip freeze > requirements.txt # 의존성 패키지 목록 생성 
pip install -r requirements.txt # 받아온 txt 파일대로 패키지 설치 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/28d2a228-a465-4c2b-aa9b-813615a7553e/Untitled.png)

---

## Django 프로젝트 :

- APP의 집합 : DB 설정 / URL 연결 / 전체 앱 설정 등을 처리

### 매일 PJT 생성 전 루틴 :

```bash
python -m venv venv # 가상 환경 venv 생성
source venv/Scripts/activate # 가상 환경 활성화 
pip install django # 4.2버전 설치되어야 함! 
pip freeze > requirements.txt # **패키지 설치 시마다 진행** 
# 폴더 내에 .gitignore 생성   >    venv/   입력 후 저장
# 폴더 어디?????????????

# 프로젝트 생성 - 현재 위치에 플젝 이름대로 만들어줘
django-admin startproject pjtname .  

# manage.py 가 현재 위치에 만들어짐 
# 혹시 잘못 만들었으면 rm -rf firstproject/

# 서버 시작
python manage.py runserver
```

---

## Django Design Patern :

- 디자인 패턴 : SW 설계에서 발생하는 문제 해결하기 위한 일반적 해결책 
                      공통적인 문제 해결하는 데에 쓰이는 형식화된 관행 
                      **APP의 구조는 이렇게 구성하자** 라는 약속

- MVC 디자인 패턴 : Model - 데이터 / View - UI / Controller - 비즈니스 로직
                               APP 구조화하는 대표적인 패턴
                               시각적 요소와 뒤에서 실행되는 로직 서로 영향 없이, 
                               독립적이고 쉽게 유지보수 할 수 있는 APP을 만들기 위해

- **MTV** 디자인 패턴 : Model / Template / View **장고에서 사용 / APP 구조화하는 디자인 패턴**
                              MVC <> MTV 대응되는 순서는 그대로 
                              View = Template  /  Controller = View 
                              기존 MVC 패턴과 동일하나 단순히 명칭을 다르게 정의

### Django APP

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8947996c-c7a1-4ee0-8b7e-2c033bdff374/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/59250c4a-7292-4d88-9107-736c2c73d43d/Untitled.png)

- 예시 : APP A : 뉴스 기사 정보  /  APP B : Auth ( 회원 정보 )

- 장고 APP : 독립적으로 작동하는 기능 단위 모듈 
                 각자 특정 기능 담당 / **다른 앱들과 함께 하나의 PJT 구성**

- APP 사용 위한 순서 : 앱 생성 > **앱 등록 ( PJT에 등록 / like 파이썬 import  )**

```bash
# 앱 생성 : 폴더 생성됨
python manage.py startapp articles  
# **앱의 이름은 복수형으로** 지정 권장 

# 앱 등록 : 프로젝트 폴더 내에 settings 
# 가서 installed_apps 리스트에 넣어주기 
#  , 생각  
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/693820d9-f966-4c02-88b0-0ae98fca090a/Untitled.png)

### Django PJT 구조 :

- 프로젝트 폴더 내에 파일 설명

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d02bd6d3-6302-4937-a326-7cbd61c4a30f/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/92f44e5d-4fc5-41a6-882b-6ecbf576e988/2cf37cdb-88a8-489e-9280-b621daa45369.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a5d57ba5-d10c-45ac-bf7f-c2cb3b458696/7c28b37e-77c3-44cf-b0c4-b5fb73e43060.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0060d4f4-c189-4ba8-a7d2-68c24bc98e66/Untitled.png)

- 1. _ _init_ _.py : 이 폴더를 패키지로 인식하도록 설정하는 파일 ( 3.5 이후는 필수는 X ) 
2. asgi . py      : 건드릴 일 X / 서버 만들고 배포하고 싶을 때에 사용 
3. settings. py : 이 PJT 와 관련된 모든 설정 작성 
4. urls. py        : 사용자가 타고 들어올 url 관리 
5. wsgi. py      : 건드릴 일 X / Deploy 할 때에 보게 될 것 

**외부 manage. py** : 장고에 명령 줄 때에 모두 이 파일 통해서 진행 / 명령어 옵션 적어 사용
                              직접 수정 X
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/58ddc657-c66c-4aba-a408-c5d56c0f0699/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/76b3e014-4d92-4f67-97e3-ce6f1553b3ae/Untitled.png)
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d9d0171d-9acb-4726-b587-5de1441c4629/Untitled.png)
    
- APP 폴더 내에 파일 설명 : 
1. _ _init_ _.py : 이 폴더를 패키지로 인식하도록 설정하는 파일 ( 3.5 이후는 필수는 X ) 
2. admin : 관리자와 관련된 설정 
3. apps   : APP 설정과 관련된 모든 정보 작성/ 건드릴 일 X 
4. models : 디자인 패턴에서 모델 ( 데이터 ) 에 해당하는 영역. 
                  여기서 클래스 정의함으로써 데이터를 파이썬 파일로 관리하는 방법에 대해 알아봄 
5. test : 테스트 관련 … 다룰일 X 
6. view : 비즈니스 로직 관련 작성됨 / 사용자의 요청에 대해 어떤 응답을 할 것인지 작성

---

# REST API : API 설계하는 방법론

- 어떤 api 서버를 어떻게 만들 것이냐? REST API 를 만들 것이다.
- API : 두 SW 서로 통신할 수 있게 하는 매커니즘 - application programming interface
        클라이언트 - 서버처럼 서로 다른 프로그램에서 요청-응답 받을 수 있도록 만든 체계
- API : 두 SW 서로 통신할 수 있게 하는 매커니즘 - application programming interface
        클라이언트 - 서버처럼 서로 다른 프로그램에서 요청-응답 받을 수 있도록 만든 체계
- WEB API : 웹서버 / 웹 브라우저 위한 API   <  
                 현대 웹 개발은 직접 개발보다 여러 Open API 활용하는 추세 
                 대표적인 써드파티 Open API : Youtube / Google Map / 파파고 / 카카오맵

- REST : Representational State Transfer 
API 서버를 개발하기 위한 일종의 SW 설계 방법론 
모두가 API 서버 설계하는 구조 다르니, 이렇게 맞춰서 하는거 어때 ? ( 규칙 X )
- RESTful API : 
REST 원리를 따르는 시스템을 RESTful 하다고 부름 
**자원을 정의 / 자원에 대한 주소를 지정** 하는 전반적 방법 서술

- REST API : 
RESET 라는 설계 디자인 약속 지켜 구현한 API

- REST 에서 자원 사용하는 법 3가지
1. 자원의 식별 - URL 

URI : Unifom Resource Identifier ( 통합 자원 식별자 ) 
- 인터넷에서 리소스 식별하는 문자열  /  가장 일반적인 URI : 웹 주소로 알려진 URL 

URL : Uniform Resource Locator ( 통합 자원 위치 ) 
- 웹에서 주어진 리소스의 주소 / 네트워크 상에 리소스 어디 있는지 알려주기 위한 약속 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e2060c4e-1803-4a52-9055-c9ae93bd56b7/Untitled.png)

- Protocal Schema : 
브라우저가 리소스 요청하는 데에 사용해야 하는 규약 
URL의 첫 부분 - 브라우저가 어떤 규약 사용하는지 나타냄
기본적으로 웹 : HTTP(S) 요구, 메일/파일 열기 위한mailto: / ftp: 등 다른 프로토콜도 존재 
cf> HTTPS : 보안과 관련된 추가규칙이 있어 포트번호도 다름
- Domain Name : 
요청중인 웹 서버를 나타냄
어떤 웹 서버가 요구되는지를 가리키며, 직접 ip주소 사용하는 것도 가능 
사람이 외우기 어렵기 때문에 주로 Domain Name 사용 
ex> 도메인 [google.com](http://google.com) == 142.251.42.142
- Port : 
웹 서버의 리소스에 접근하는 데 사용되는 기술적인 문 (Gate)
HTTP(S) 프로토콜의 표준 포트 : HTTP - 80 / HTTPS - 443 
표준 포트만 작성 시 생략 가능 
장고 / 뷰 다 포트번호가 다름
- Path : 
웹 서버의 리소스 경로
**실제 파일이 위치한 물리적 위치 X** > 실제 위치 아닌 추상화된 형태의 구조 표현
- Parameters : 
웹 서버에 제공하는 추가적 데이터
& 기호로 구분되는 key-value 쌍 목록
서버는 리소스 응답 전에 이러한 파라미터 사용해 추가 작업 수행 가능

1. 자원의 행위 - HTTP Methods ( == HTTP verbs ) 
- 리소스에 대한 행위 ( 수행하고자 하는 동작 ) 을 정의 

- 대표 HTTP Request Methods : 
1. GET : 서버에 리소스의 표현 요청 ( 검색, 조회 ) 
2. POST : 내가 작성한 데이터를 지정된 리소스에 제출  >  생성
3. PUT, PATCH : 제출한 데이터 바탕으로 기존 리소스 전체 / 일부 수정 
4. DELETE : 삭제 

- HTTP response status codes : 
특정 http 요청이 성공적으로 완료 되었는지 여부를 응답 
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2347e3e1-e8eb-413a-b28d-968c13ced9c4/Untitled.png)
    

1. 자원의 표현 - JSON 데이터 
- 현재 장고가 응답하는 것 ( 자원을 표현하는 것 ) 
장고는 풀스택 framework  >  기본적으로 사용자에게 페이지 ( **html** ) 응답 
BUT 서버가 응답할 수 있는 것  >  페이지 / 다양한 DTYPE 
REST API 는 이 중에서도 **JSON** 타입으로 응답하는 것을 권장 
>> 응답 데이터 타입의 변화

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b2976714-8e0d-492f-9116-6eec9bbc355b/Untitled.png)

- 장고는 더이상 Template 부분에 대한 역할 담당 X 
본격적으로 FE BE 분리되어 구성됨

---

## 요청 & 응답

### Django REST Framework - DRF 라이브러리 설치

- 장고에서 RESTful API 서버 쉽게 구축할 수 있도록 도와주는 오픈소스 라이브러리

```bash
pip install djangorestframework
pip freeze > requirements.txt # 업데이트 

## settings에도 추가 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/430e3e2a-a623-43c5-a180-3bcc818a48f3/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4115c8f7-f97a-43ba-8e56-e2068ebcfcb7/Untitled.png)

- 장고 코드 작성 순서 : 일반적인 요청 & 응답 흐름 맞춰서

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/745f2782-425c-4c4f-939c-8d6fcbdb14d0/Untitled.png)

1. urls에서 경로 설정 
2. views 에서 응답 처리 
template : 보내줘야 할 html 있다면 방문 
models : 보내줘야 할 데이터 있다면 방문
- 1, 2 반복

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/721664de-7b0b-4bf4-a3ca-8950761bd79b/Untitled.png)

- URLs : 분배기 
URL 패턴 정의하고 해당 패턴이 일치하는 요청 처리할 view 함수를 연결(매핑)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ff138d0f-535c-4944-98b5-d6263096b9dc/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b5ec7f0e-dbbf-4ceb-b9a0-47a9a242b0d1/Untitled.png)

사용자가 만약 index 라고 하는 경로로 요청을 보냈고, 
이에 대한 처리를 articles pkg 안의 index 라는 모듈에 있는 index라는 함수를 실행 
장고가 미리 만들어놓은 path 험수에 인자 2개  >  사용자의 요청 왔을때 실행

- View : 
Json 응답 객체 반환하는 index view 함수 정의
모든 view 함수는 첫번째 인자로 request 요청 객체 필수적으로 받음
매개변수 이름이 request 아니어도 되지만, 그렇게 작성 X

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6f95a282-3244-4428-8fd3-20ce76740850/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c40f99ad-fe75-43d0-a60c-51ac7e59951e/9764687d-1f0c-429e-b257-9ba7c8ed1443.png)

- 해당 view 안에 있는 **api_view 데코레이터** : 
DRF **view 함수에서 필수**로 작성
기본적으로 GET 메서드만 허용 / 다른 메서드 요청 : **405** Method Not Allowed 응답
                                                                                  해당 데코레이터의 인자 추가도 가능하긴 함 
DRF View 함수가 응답해야 하는 HTTP 메서드 목록 작성

- DRF 설치 : RESTful한 API 쉽게 만들기 위해 
                    ㄴ 사용자의 요청을 보냈을 때에 응답하도록 ???????

- 해당 views 안의 index의 조건이 맞을 때에, **return 해주는 형식 : Json**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fe90509a-97a5-4611-bef9-6c4e4d218fd1/Untitled.png)

- JsonResponse() : 장고에 내장된 HTTP 응답 클래스  
                           첫번째 위치 인자로 JSON 으로 변환 가능한 데이터 받아와 응답 객체 반환 
                           필요 시, http response 의 응답 코드도 설정하여 반환 가능

```bash
# 위에 내용 체크해보자 
python manage.py runserver
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/93e62f55-83c3-4643-8100-9a101387e3f4/Untitled.png)

- 아까 로켓이랑 다른 이유 ? : 
APP 등록 이후부터는 모든 경로에 대한 처리들을 직접 만들어줘야 함
포트 8000번으로 요청 들어왔을 때에 응답해줄 어떠한 내용물도 만들지 않았다

- 아랫쪽에는 정해놓은 URL들의 목록 ( 지금은 6개 처리되어있음 )

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ac1858e0-8351-44c7-90e7-e6d94244e863/Untitled.png)

- 이렇게 넘어온 데이터 ? json 형식의 데이터

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2eeb28b8-bcb9-42ad-88e8-47713caaa14f/Untitled.png)

- 위와 같이 어떤 식으로 반응하는지도 출력됨

---

### 변수 & URL :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/802de4a6-76c0-4f51-80e1-15c1ec65053b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/19301858-a37c-40a8-9ce7-40371a0ec04f/Untitled.png)

- ANS : Variable Routing : URL 일부에 변수를 포함시키는 것 
                                       변수는 VIEW 함수의 인자로 전달할 수 있음
- Path converters : URL 변수의 타입 지정 : str, int 등 5가지 타입 지원

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9e3eaf69-d574-49b0-a063-03e4466b9fb1/Untitled.png)

**[ urls / views 변수명 동일하게! ]**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8715ca68-e65d-4c2f-8d6f-5c29f838e030/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7bcb3079-e856-4f51-9846-5671c7b65968/Untitled.png)

이런 식으로, <int:num> 과 같이 변수명  num 과 함께 넘겨줬을 때에, 
넘겨준 변수명을 받는 views에서도 그대로 사용해야 함 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2910e0d9-823e-40ae-a943-7b43ffe3c236/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dbe5990d-fc64-45a8-9e18-2294c053a363/Untitled.png)

- 위 코드의 흐름 :

urls 에 경로 path > 루트경로 뒤에 articles  뒤에 정수형태 > views에 넘겨줌 > 문자열 넘겨주면 펑 

---

### APP & URL :

URL 관리 최종 형태 : APP을 어떻게 관리하냐에 따라서 달라짐 

- APP URL Mapping : 
각 앱에 URL 정의하는 것 
PJT와 각 APP이 URL 나누어 관리 편하게 하기 위함

- PJT 늘어나면 모든 APP은 동일한 구조 가지고 있으므로 
views 모듈이 겹칠 수 있다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f09e80bb-89d2-44d9-894f-ce0c9a277782/Untitled.png)
    

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/9cee2fed-42b6-46c5-8e30-bb205368331a/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2ddc61a9-4dbb-46ab-ac4f-5d22910d3339/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d2066ce6-83df-449b-8753-535966dfaa9b/Untitled.png)

- **include()** : 프로젝트 내부 앱들의 URL 참고할 수 있도록 매핑하는 함수 
                URL의 일치하는 부분까지 잘라내고,  
                남은 문자열 부분은 후속 처리를 위해 include 된 URL 로 전달

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5b6035d5-9716-4c37-9253-3cf2c2438273/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f0460774-1282-40f7-8d3e-6393751ec69f/Untitled.png)

- articles라는 곳으로 요청 오면 
뒷부분들은 articles 의 urls 에서 따로 관리할 것 

ㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹㄹ

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/630c924e-3a64-4951-9373-a9f4458c8ada/Untitled.png)

### 참고 140P ~

# 오늘 내용 요약

1. PJT 의 urls 에서 APP의 views를 불러 와 ( 해당 views 내의 함수명 > views.func 로 ) 
urls 에서 include 를 import 하고, 아래와 같은 식으로 articles 라는 주소를 입력하면 
articles 의 urls 를 본다는 식으로 APP의 urls 를 사용할 수 있다. 

```python
path('articles/', include('articles.urls')),
```

1. APP 의 views 에서는 GET 을 받았을 때에, 특정 함수를 실행시키고, 

```bash
def index(request):
	return JsonResponse({"message": "Hello, world!"})
```

와 같은 식으로 웹의 화면에 출력 가능

---

Django Model & ORM 

## Model :

- Django Model :
DB의 테이블을 정의하고, 데이터를 조작할 수 있는 기능들을 제공 
웹 서비스의 디자인 패턴 **MTV중 model** / 테이블 구조를 설계하는 청사진

- **Models. py 통한 DB 관리** 
**APP의 models.py 에 class 정의**  >  객체 생성 후 메서드 사용해 DB 컨트롤

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ef0b5f94-9b8f-4742-a484-4547d0638f61/d6e5240c-3c0c-4486-b374-aab547d66e39.png)

- **Model 클래스** 작성 : 
위의 코드로 Table 구조 만듦  ( **id 필드는 Django 가 자동 생성** )
장고가 제공해주는 model 클래스를 상속받아 옴으로써 기능들을 사용 가능
아래와 같이 코드 사용하면, **articles_article 이라는 테이블 생성**

```bash
# articles APP의 models.py
class Article(**models.Model**):
		title = models.CharField(max_length = 10)
		content = models.TextField()
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/31262f57-09f5-4eaf-befe-40fe8ff61f5c/Untitled.png)

1. models.Model :   django.db.models 모듈의 Model 이라는 부모 클래스를 상속받음
                            **Model : 모델에 관련된 모든 코드가 이미 작성되어있는 클래스** 
                            개발자는 가장 중요한 Table 구조를 어떻게 설계할지에 대한 
                            코드만 작성하도록 하기 위한 것 ( 상속을 활용한 Framework의 기능 제공 ) 
2. title / content : **테이블의 각 필드(열) 이름**
3. CharField / TextField : **테이블 필드의 데이터 타입** 
4. max_length = 10 : 테이블 필드의 **제약조건** 관련 설정
- 제약조건 : 데이터가 올바르게 저장되고 관리되도록 하기 위한 규칙
ex > 숫자만 저장되도록 / 문자가 100자까지만 저장되도록 하는 등

APP의 [views.py](http://views.py) 에서 **Article 클래스의 객체 a1 생성** 후, 해당 객체의 title / content 에 값 주고
장고가 DB에 저장하는 과정 거치면 요소 삽입 가능  << ORM 파트 

---

## Migtations

- class 를 DB로 그대로 넘기기에는, ID를 알아서 넣어준다던가 
APP명_클래스명 으로 테이블명을 넘긴다던가 하는 과정이 생략되어있음

- model **클래스의 변경사항 ( 필드 생성 / 수정 / 삭제 등 ) 을 DB에 최종 반영**하는 방법

- Migrations 과정 : 
0001_initial 과 같은 migration 파일에 있는 migration 클래스는 자동으로 생성되니 신경 ㄴㄴ 
이 파일을 보면 id를 자동으로 추가해주거나 테이블 명을 설정하고있음

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ee9f9358-b108-4fc2-9c60-e4ac39bf0347/Untitled.png)

- Migrations 핵심 명령어 2가지 :

```bash
# model class 기반으로 **최종 설계도 ( migration ) 작성**
python manage.py **makemigrations**

# 최종 **설계도를 DB에 전달**하여 반영
python manage.py **migrate**

# db.sqlite3 파일 생성 > vscode 의 db 가서 connect ( sqlite 파일은 모든파일찾기해서 ) 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f180d07c-fb8f-4db6-b07a-eb080cd6dcd4/Untitled.png)

- 추가 Migrations :
1. 이미 생성된 테이블에 필드를 추가해야 한다면 ? : **추가 모델 필드 작성**

```bash
# articles APP의 models.py 에 추가 
class Article(models.Model):
		title = models.CharField(max_length = 10)
		content = models.TextField()

		## 추가 내용		
    # 생성되는 시점을 기록 - 옵션 : 추가되는 시점에 자동으로 현재시간+
    created_at = models.DateField(auto_now_add = True)
    # 수정되는 시점을 기록 - 수정되는 순간 현재시간 알아서 추가 
    updated_at = models.DateTimeField(auto_now = True)
    
# 변동사항 생겼으므로 makemigrations  >  설계도 추가 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d6b247c3-c1f6-460a-a96b-718debd661f2/Untitled.png)

- 이미 기존 테이블 존재하기 때문에, 필드를 추가할 때 필드의 기본값 설정 필요
**1번** 현재 대화를 유지하면서 **직접 기본값을 입력**하는 방법 > 엔터 누르면 알잘딱
**2번** 현재 대화에서 나간 후, **models. py 에 기본 값 관련 설정**을 하는 방법

- migrations 폴더에 새로운 파일 생성 ( 수정했으므로 )  <  git 과 같은 느낌

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1c7cbb0b-b916-4d26-b1d0-7c883f8ab5f0/Untitled.png)

- migrate 명령어 이용해서 db에 반영

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f781a19b-53e2-46ff-ba86-888860f2124c/Untitled.png)

### Migrations 기타 명령어 :

```python
python manage.py showmigrations
# migrate 됐는지 여부 확인
# [X] 표시 : migrate 완료되었음

python manage.py sqlmigrate articles 0001
# 해당 migrations 파일이 SQL언어 (DB에서 사용하는 언어) 로 
# 어떻게 번역되어 DB에 전달되는지 확인하는 명령어 
```

### CF> DB 초기화 :

1. Migration 파일 삭제
2. db.sqlite3 파일 삭제
- **init / migrations 폴더 삭제 X**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cfc3071e-090a-4d2b-ab59-387e9a2592ba/Untitled.png)

---

### Model Field :

- DB테이블 필드(열)을 정의해 해당 필드에 저장되는 dtype 과 제약조건 정의
이미 장고에서 다 구현해놔서, 가져다 쓰기만
1. CharField() : 길이의 제한 있는 문자열 넣을 떄에 사용 ( max_length 필수 ) 
2. TextField() : 글자의 수가 많을 때에 사용 
3. DateTimeField() : 날짜 / 시간 넣을 떄에 사용 

---

- cf> migrate 하면서 article 외에 다른 테이블도 많이 만들어 지는데, 이건 무엇?
      >> 유저에 대한 정보 처리할 수 있는 테이블 ( admin / auth … )

## Admin site

### Automatic admin interface :

- Djanho는 추가 설치 및 설정 없이 자동으로 관리자 인터페이스 제공
데이터 확인 / 테스트 등 진행하는 데에 매우 유용

1. admin 계정 생성 : email은 선택사항이라 입력 안하고 진행 가능  
                             비밀번호 입력 시 보안상 터미널 출력 x > 무시하고 입력 고고 

```bash
python manage.py createsuperuser
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/717e66dd-53c6-42e3-a944-b0cae16572c7/Untitled.png)

- auth_user table 에 admin 생성된 모습 / pw 16진수로 난수화 되어있다… 장고GOAT

pr

1. 서버 run 후 admin/  > 로그인 > 내 pjt db를 관리하는 관리자 페이지로 이동 가능 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/397511f3-e91d-4d6d-8a79-7a872be5a8b8/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c797b136-a886-469c-98ad-91b02c2f5d1f/Untitled.png)

- 이 페이지에서 article 테이블 정보 확인하는법?

```bash
# articles APP admin. py

from django.contrib import admin
# 내 dir(articles)의 models.py서 Article 클래스 가져와 
from . models import Article 

admin.site.register(Article)# 관리자.사이트.등록(모델)
```

[]()

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/017bc9bb-fc19-43ae-ae0d-028a6c3ecd6e/Untitled.png)

- 사이트의 add article 이용해 데이터 생성 / 수정 도 가능

---

## ORM

- 테이블에 데이터 삽입?
- Object Relational Mapping : 객체 지향 프로그래밍 언어 사용해
                                             호환되지 않는 유형의 시스템 간에 데이터를 변환하는 기술 

>> 파이썬(장고) <> SQL(DB) 간의 언어가 달라 소통 불가? 장고의 ORM이 가능하게 해준다

### Query :

- DB에 특정한 데이터 보여달라는 요청
- 쿼리문 작성 : 원하는 데이터 얻기 위해 DB에 요청 보낼 코드 작성한다
- 파이썬으로 작성한 코드가 ORM에 의해 SQL로 변환되어 DB에 전달되며,
DB의 응답데이터를 ORM이 QuerySet이라는 자료형태로 바꿔 우리에게 전달

### QuerySet :

- DB에게 전달 받은 객체 목록 (데이터 모음) / iterable 하고, 1개 이상의 데이터 불러와 사용 가능
- Django ORM 통해 만들어진 자료형
- DB가 **단일**객체 반환할때 : **QuerySet 아닌 모델(class)의 인스턴스로 반환**

### OMR / QuerySet API 사용 이유 :

- DB 쿼리를 추상화 > Django 개발자가 DB와 직접 상호작용하지 않아도 되도록 함
- DB와의 결합도 낮추고, 개발자가 더욱 직관적이고 생산적으로 개발할 수 있도록 도움

### QuerySet API

- ORM에서 데이터를 검색 / 필터링 / 정렬 / 그룹화 하는 데에 사용하는 도구 
API 사용하여 SQL 아닌 Python 코드로 데이터 처리
- 파이썬의 모델 클래스와 인스턴스 활용해 DB에 데이터 CRUD

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2cb9e5b0-417a-4845-a2df-12602eef784d/Untitled.png)

- 데이터 보낼때: QuerySet API / 들여올때: QuerySet / 인스턴스 형태로 받음

- QuerySet API 구문 : 
Article 이라는 모델 class : Model 클래스 상속받음
Model 클래스의 objects 라는 요소의 all ( QuerySet API ) 메서드 이용

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/548421a4-37dc-4905-a7e0-1170401b5f3b/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/688c0876-07ed-4e93-a440-fa10905dab26/Untitled.png)

### QuerySet API 실습

- CRUD : DB 생성 / 조회 / 수정
- Postman 사용

- 스켈레톤 코드 메서드 설계 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7af2dd23-3c25-441b-b6f5-cee8acda3ea9/Untitled.png)

```python
# 1. PJT의 urls.py 
urlpatterns = [
    path("admin/", admin.site.urls),
    # 요청이 articles/로 시작하면 뒤로 이어지는 URL path는 
    # articles/urls.py에서 처리하도록 설정
    path("articles/", include("articles.urls")),   
]

# 2. articles APP의 urls.py
from django.urls import path
from . import views
urlpatterns = [
    path('new/', views.article_create), # 생성 C 
    path("list/", views.article_list),
    path("<int:pk>/", views.article_detail), # 상세 조회 R 
    path("<int:pk>/edit/", views.article_update), # 수정 U
    path("<int:pk>/delete/", views.article_delete), # 삭제 D 
] 

# 3. articles APP의 views.py
from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework import status
from .models import Article

# Article 객체를 딕셔너리로 변환하는 함수 param article: Article 객체
# return: Article 객체를 딕셔너리로 변환한 결과
# JsonResponse 를 통해서 웹에 띄울꺼라 이걸 쓴거고, 이후에는 안쓸것 

def model_to_dict(article):
    return {
        'id': article.id,
        'title': article.title,
        'content': article.content
    }

@api_view(["POST"]) # POST 할때만 실행 
def article_create(request):
    # 데이터 객체를 만드는 방법
    ### 1] models.py 에 아까 만들어 놓은 table 클래스 > 이의 객체 생성 
    article = Article()
    article.title = **request.data.get**('title') 
    article.content = request.data.get('content')
    # 나머지 2개의 요소 [생성 시점 / 수정 시점]은 옵션덕에 알아서 들어가므로 입력받기 ㄴㄴ 
    ### 2] article = Article(title=request.data.get('title'), content=request.data.get('content')

    # 데이터베이스에 저장
    article.save()
    
    ### 3] **QuerySetAPI : create**
    # 위 두 방법과 달리 **객체 생성과 동시에 데이터베이스에 저장** - save 필요 X 
    # Article.objects.**create**(title=request.data.get('title'), content=request.data.get('content'))
    
    # 반환값으로 article.id와 status.HTTP_201_CREATED를 응답 코드로 반환
    return JsonResponse({'id': **article.id**}, **status=status.HTTP_201_CREATED**)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/52cce5ba-8465-4015-b636-7559894c72e8/Untitled.png)

- 제대로 동작하는 것 확인 가능

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/548421a4-37dc-4905-a7e0-1170401b5f3b/Untitled.png)

- 조회 메서드 : **QuerySetAPI**

- Return new QuerySets :  

all() : 전체 데이터 조회 
filter() : 특정 조건 데이터 조회  
            >  조건에 해당하는 값 없어도 오류 X  >  빈 쿼리셋 반환 

- Do not return QuerySets : 

get() : 특징 -  **단일 객체 반환 ( 쿼리셋 X )** 
                      객체 찾을 수 없으면 DoesNotExist 예외 발생 
                      둘 이상의 객체 찾으면 MultipleObjectsReturned 예외 발생 
                      이런 특징 > primary key (pk) 와 같이 unique 보장하는 조회에서 사용해야

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/059a31fe-3351-4efd-9d43-a8b675cdd591/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0e2fb382-c155-480d-8507-9b9d4f847668/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0ae4f68b-465e-4fe6-91e7-0f851e5d4f73/Untitled.png)

```python
# 모든 객체에 대해 정보 보여줌
@api_view(['GET'])
def article_list(request):
    articles = Article.objects.**all()**  # 저장된 모든 article 데이터를 가져옴
    # articles = Article.objects.filter(title='title1')
    
    # 출력 위해 **QuerySet 객체 순회하며 각각의 데이터 딕셔너리로 변환**
    if articles:  # **articles가 존재하면**
        data = [**model_to_dict**(article) for article in articles]
        # 위에서 정의한 함수 - 이후에는 잘 안씀  
        return JsonResponse({'data': data})  # articles가 존재하면 데이터를 반환
    return JsonResponse({'data': []})  # **articles가 존재하지 않으면 빈 리스트 반환**
    
    # return JsonResponse(data, safe=False)  
    # 딕셔너리가 아닌 데이터를 반환할 때 safe=False로 설정
    
    
# 특정 **객체 하나**에 대해 모든 정보 보여줌 
@api_view(['GET'])
def article_detail(request, **pk**): 
    article = Article.objects.get(**pk=pk**) # **get 메서드 : 단일 객체** 
    # article = Article.objects.get(title='title')  
    # title 필드에 저장된 값이 'title'인 데이터를 가져옴
    
    return JsonResponse(**model_to_dict**(article)) 
    # 단일 객체 dict 형태로 바꿔 return
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7920dddd-69cd-49e9-af35-733d4d7408df/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1671aa83-1f25-4328-85d0-962dce4d61bb/4561a7c8-51cd-4536-a448-5d98d98ca1c7.png)

```python
# 수정
@api_view(['PUT'])
def article_update(request, **pk**): 
		# 게시글 수정 역시 어떤거 수정 원하는지 알아야 함 
    article = Article.objects**.get(pk=pk)** 
    # 수정할 article 데이터를 가져옴 = 이것도 get이라서 하나 조회 
    # **수정할 데이터를 request.data로부터 가져와서 수정**
    article.title = request.data.get('title') 
    article.content = request.data.get('content') 
    article.save() # 수정된 데이터를 저장
    return JsonResponse(model_to_dict(article))
    
# 삭제
@api_view(['DELETE'])
def article_delete(request, pk):
		# 삭제할 대상 찾아야 함 ( pk 받아옴 ) 
    article = Article.objects.get(pk=pk)  
    # 삭제할 article 데이터를 가져옴
    article.delete() # 데이터 삭제
    
    # 삭제 후 삭제 해줬다고 무언가를 리턴 해줘야 함 
    # article 객체는 이미 db에서 삭제되었으므로,
    # article.pk = None but 매개변수는 안사라지니 괜찮음
    return JsonResponse({'delete':f'{pk}번 게시글 삭제되었습니다.'}, 
										    status = status.HTTP_204_NO_CONTENT)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/206f518f-464b-4a32-9ed5-142cd4be7585/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/23c3b311-09af-48b4-a9bf-0ae0adf6c4f9/Untitled.png)

- **model_to_dict** 함수 안쓰면? : 출력이 non-dict 형이라 **serialized ( 직렬화 과정 ) 거쳐달라고 요청** 
                                                pretty 아니라 preview 로 보면, 한글 깨지는 것도 볼 수 있음 
                                                APP 많아지면 하나하나 바꾸기 너무 별로다

---

## Django Serializer : 유효성 검사 /

### 개요

- **Serialization : 직렬화**
**내부적으로 저장된 데이터를 다른 시스템이나 클라이언트가 이해할 수 있도록 변환하는 과정**
- 의미 : 여러 시스템에서 활용하기 위해 데이터 구조나 객체 상태를 나중에
             재구성할 수 있는 포맷으로 변환하는 과정 
             어떠한 언어나 환경에서도 나중에 다시 쉽게 사용할 수 있는 포맷으로 변환하는 과정 
- 예시 : 데이터 구조나 객체 상태를 나중에 재구성할 수 있는 포맷으로 변환하는 과정

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/32e2247c-0b10-4280-8949-3d109c8f00ec/Untitled.png)

- 객체 데이터 > … Serialization : Serialization Class가 담당 > Serialized data > JSON 파일

- Serializer : Serialization을 진행하여 Serialized data를 반환해주는 클래스

- ModelSerializer : Django 모델과 연결된 Serializer 클래스 
- 일반 Serializer 와 달리, 사용자 **입력 데이터 받아 자동으로 모델 필드에 맞춰 Serialization 진행**

---

### Serializer class

- articles APP의 [**serializers.py**](http://serializers.py) 에서 rest 프레임워크가 가지고 있는 serializer,
내가 모델에 정의해 놓은 article 도 가져와서 클래스 정의

```python
# articles APP의 [**serializers.py**](http://serializers.py) 

'''
메타 클래스 : 시리얼라이저의 설정을 정의하는 데 사용
              시리얼라이저가 사용할 모델 그 자체 / 필드 / 옵션 등을 명시
              시리얼라이저와 모델 간의 연결을 설정하는 중요한 역할
'''

class ArticleListSerializer(**serializers.ModelSerializer**):
    class Meta:
        model = Article          
        # Model: 직렬화할 대상 모델
        # APP의 models.py > Article class에 컬럼 정의되어         
        # fields = ('id', 'title', 'content') # 보여줄 field
        exclude = ('created_at', 'updated_at',) # 뺄 field 
        
class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__' 
        # 상세 조회 : __all__ 식으로 
        # # fields = ('id', 'title',)
        # exclude = ('created_at', 'updated_at',)
```

---

### CRUD with ModelSerializer

- 본 실습에서의 APP 구조 확인 :

```python
###################################################################################

# **APP의 urls.py**  /  PJT 에서 특정 링크가 나오면 여기서 확인하라고 설정해 두었음

from django.urls import path
from . import views

urlpatterns = [
    path(**''**, views.article_list), # URL 이 비어있으면 
    path('**<int:article_pk>/**', views.article_detail), # URL에 숫자 들어오면 
    
    # APP의 views.py 가서 함수 실행해 
]

###################################################################################

# 아래부터는 APP의 views.py

from rest_framework.decorators import api_view
from . models import Article
from . serializers import ArticleSerializer, ArticleListSerializer

# from django.http import JsonResponse << 이제 안쓰고 밑에꺼 씀 
from rest_framework.response import Response 

from rest_framework import status

@api_view(['GET', 'POST'])
def **article_list**(request):
    if request.method == 'GET':
        articles = Article.objects.all()  # 모든 Article 객체를 가져옴
        
        # **Serializer를 통해 직렬화**
        # 직렬화 할 객체가 여러개 일 경우, many=True
        # 위 코드블록에서 시리얼라이저 만들 때에 메타데이터 중요 
        serializer = **ArticleListSerializer**(articles, **many=True**) 
        
        return Response(**serializer.data**)   # ********************
        # 직렬화한 데이터를 기존에 Json으로 변환해서 하는게 아닌
        # **장고 rest framework 가 가지고 있는 Response 사용해 반환**
        
        
    elif request.method == 'POST': # **새로운 article을 생성**
        serializer = ArticleSerializer(data=request.data)  
        # request로 전달받은 데이터를 serializer에 담음
        
        # **데이터 생성 :: 조회와 다르게 유효성 검사 필요** 
        if serializer.**is_valid()**:  
        # serializer를 통해 입력 데이터의 유효성 검사 > 통과 못하면 errors

            serializer.save() # 유효성 검사를 통과하면 데이터베이스에 저장            
            # return 해주는 요소들 (데이터, 상태) 등등은 예전과 동일 
            return Response(serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.**errors**, status=status.HTTP_400_BAD_REQUEST)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/82dde0cf-26e1-4f64-bdaf-37ecc8acd13e/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2464225b-95a7-4169-8d4c-b0b1ed28d704/Untitled.png)

- raise_exception :  is_vaild() 의 선택 인자 
                            유효성 검사 통과 못할 경우 ValidationError 예외 발생시킴
                            DRF에서 제공하는 기본 예외 처리기에 의해 자동으로 처리. 
                            기본적으로 HTTP 400 응답을 반환

                                  해당 인자 넣으면 아래 RETURN 문 안넣어도 됨 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4b30cd42-012b-4f8f-bd49-d00edd33c1c1/Untitled.png)

```python
python [manage.py](http://manage.py) migrate # db 생성 
python [manage.py](http://manage.py) runserver # 서버 start
```

```python
from django.urls import path
from . import views
urlpatterns = [
    path(**''**, views.article_list),
    path('**<int:article_pk>/**', views.article_detail),
]

"""
path 통해서 받아온 pk 이용해
아래부터는 views.py
"""
@api_view(['GET', 'DELETE', 'PUT'])
def article_detail(request, article_pk): # 넘겨받은 pk값 

    article = Article.objects.get(pk=article_pk)

    if request.method == 'GET': 
    # article 객체를 직렬화하여 반환
        serializer = ArticleSerializer(article) # many = false 
        return Response(serializer.data)

    elif request.method == 'DELETE':
        article.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

    elif request.method == 'PUT':
        # instance: 수정할 article 객체 . data: 업데이트할 데이터
        # article 안넣으면 새로 만들음 
        serializer = ArticleSerializer(article, data=request.data)
        
        # :partial = True로 설정하면 일부 필드만 업데이트 가능 // request Method Patch 일 때
        # serializer = ArticleSerializer(instance=article, data=request.data, partial=True)

        if serializer.is_valid():  
        # serializer를 통해 입력 데이터의 유효성 검사
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f9cf4f24-6108-4961-bcd9-ecdab7fd4bdb/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c0d8cd17-5bcd-486d-a544-f83c7f4ed694/Untitled.png)

---

# Django Relationships :

## Many to one relationships :

- 한 테이블의 0개 이상의 레코드가 다른 테이블의 레코드 한개와 관련된 관계
- ex> Comment [ N ] - Article [ 1 ]  : 0개 이상의 댓글은 1개의 게시글에 작성될 수 있다. 
외래키 : unique 한 값 ( id ) 

**인자가 많은 요소 ( 여기서는 comment ) 에 외래키를 둔다**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6a8809ed-0af2-4c86-9109-faa2614f0725/Untitled.png)

- cf> 정규화 : 각 테이블에서 해당 테이블의 정보만을 올바르게 구조화 해야 한다

### 댓글 모델 : 게시글과 댓글간의 관계 정의

- 외래키 설정 위해 장고에서 어떤 것 필요로 하는지 ?

- ForeignKey() : N : 1 관계 설정 모델 필드 

- models.**ForeignKey(to, on_delete="")** 
   to : "articles.Article" 와 같이 app.model_class 와 같은 형태로도 가능 ???????? 
****
- [ article - **ForeignKey 클래스의 인스턴스** 명 ] : **참조하는 모델 class명 의 단수형** 작성 권장

- 외래키 : ForeignKey 클래스 작성 위치 상관없이, **table 에서는 항상 마지막**에 

- on_delete : 외래키가 참조하는 객체(1) 사라졌을 때에, 
                     외래키를 가진 객체(N)을 어떻게 처리할 지를 정의하는 설정 [ 데이터 무결성 ] 
                     **CASCADE** : 부모 객체(참조된 객체) 삭제되었을 때에, **이를 참조하는 객체도 삭제**

```python
# articles/models.py
from django.db import models

# 기존 Article ( 게시물 ) 
class Article(models.Model):
    title = models.CharField(max_length=120)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    def __str__(self):  # 매직 메서드 정의해놔야 인스턴스 출력했을 때에 원하는 값이 나옴 
        return self.title

# 추가하는 댓글         
class Comment(models.Model):
    """
    하나의 게시글에 여러 댓글 가능 > Article을 ForeignKey로 참조
    on_delete=models.CASCADE : 참조하는 객체가 삭제될 때 참조하는 객체도 삭제되도록
    
    sql에서 볼때는 외래키를 보고 join 하는 식으로 하면 정상인데,
    파이썬 객체로 확인할 때에는 게시글 테이블을 그대로 받아오는 식으로 
    
    >> 조회 위해서 쿼리 날리지 말고, c1.article.title 과 같은 식으로 접근 가능 
    >> 이거때문에 역참조 생각해봐야  ????????? 
    """
    **article** = models.**ForeignKey**(**Article**, **on_delete=models.CASCADE**)    
    content = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

- Migration 이후 댓글 테이블 :  참조 대상 클래스명 + _ + 클래스명
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/96dcb640-7762-4994-a4d5-2f3110c398ff/Untitled.png)
    

---

## 관계 모델 참조 :

### 역참조 : 1 → N

- N:1 관계에서, 1에서 N을 참조하거나 조회하는 것
- **N [ Comment ] 는 외래 키를 가지고 있어 물리적으로 참조가 가능**하지만, 
**1 [ Article ] 은 N에 대한 참조 방법 X** → 역참조 기능 필요

### 역참조 사용 예시 :

- [  모델 **인스턴스** . **역참조 이름**(related_manager) . **QuerySetAPI**  ]
    - ex> [  article . comment_set . all()  ]
    특정 게시글에 작성된 댓글 전체를 조회하는 명령 
    1번 게시글을 참조하고 있는 대상들로부터 데이터 받아오는 형식의 api
    - **related manager** : [**참조하는 model class 소문자**]_set 
    - N:1 OR M:N 관계에서 역참조 시에 사용하는 매니저 
    - objects 매니저를 통해 QuerySetAPI 사용했던 것처럼, related manager 통해 사용
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/0ca2acfd-1774-499f-9b66-29d2edbbaf6b/Untitled.png)
    

---

## DRF with N:1 Relation :

### 사전 준비 :

- Comment 모델 정의 : 
1. Comment 클래스 정의 및 DB 초기화  <<  위에서 했음 
2. migrate 작업 진행 ( makemigrations > migrate )

- URL 및 HTTP request method 구성 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6d05dfdc-f7aa-4091-960a-ea128e218e7d/Untitled.png)

### 댓글 CRUD :

### 1. POST :

```python
# 단일 댓글 생성 위한 url / view [ articles 의 ] : 
path('<int:article_pk>/comments/', views.comment_create),

@api_view(['POST'])
def comment_create(request, **article_pk**):
    article = Article.objects.get(**pk=article_pk**)
    serializer = CommentSerializer(data=request.data) # 인풋 데이터 시리얼라이저 넣고
    if serializer.is_valid(raise_exception=True):     # 유효성검사 하고 (예외처리 해주고)
        serializer.save(**article=article**)              # 댓글 저장 
        return Response(serializer.data, status=status.HTTP_201_CREATED)
```

- 댓글 생성 위한 CommentSerializer 정의 :  아래의 CommentSerializer

```python
# articles/serializers.py

from rest_framework import serializers
from . models import Article, Comment

class ArticleListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        exclude = ('created_at', 'updated_at',)

# 게시글 조회 할 때 해당 게시글의 댓글도 함께 조회
class ArticleSerializer(serializers.ModelSerializer):
    class CommentSerializer(serializers.ModelSerializer):
        class Meta:
            model = Comment
            fields = ('id', 'content',)   # 댓글의 id와 content만 보여준다.
		"""
    read_only=True 옵션을 통해 해당 필드를 읽기 전용으로 설정할 수 있다.
    comment_set = CommentSerializer(many=True, read_only=True) # related_name을 통해 역참조
    댓글 갯수 표시
    source 옵션을 통해 특정 필드를 지정하거나, 메서드를 호출할 수 있다.
    comment_count = serializers.IntegerField(source='comment_set.count', read_only=True)
		"""
		
    class Meta:
        model = Article
        fields = '__all__'
        # # fields = ('id', 'title',)
        # exclude = ('created_at', 'updated_at',)

# **댓글 조회 시 게시글 정보도 함께 조회**
class CommentSerializer(serializers.ModelSerializer):
        class ArticleSerializer(serializers.ModelSerializer):
             class Meta:
                    model = Article
                    fields = ('title',)
        
        article = ArticleSerializer(read_only=True)
        class Meta:
            model = Comment
            fields = '__all__'
            # read_only_fields = ('article',)
```

- 게시글 하나 생성 이후 댓글 하나 넣는 첫번째 ex

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4327c06d-1bc7-4157-91bf-fea99589f301/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/56ae5167-99e3-4380-98ad-ecd6af046ae3/Untitled.png)

- 하지만, 위와 같은 방식은 논리가 맞지 않는다.
유저가 게시글에 댓글을 달 때에 article 번호를 직접 주는 일은 없기 때문에 
아래와 같이 동작이 되어야 하는데,

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f27ae358-93c3-4ee2-aaa7-6b6af7987cdc/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f38b9085-a9fe-48b8-8f7d-1f7ea75e6b61/Untitled.png)

- 위의 comment serializer 생성 코드와 comment_create 함수에 들어가 있는 부분이 이를 해결
이 두 부분을 뺀다면 , 위의 우측 이미지와 같이 article field 가 비어있다 / 필수라고 말해줌

```python
        class ArticleSerializer(serializers.ModelSerializer):
             class Meta:
                    model = Article
                    fields = ('title',)
        
        article = ArticleSerializer(read_only=True) # 읽기 전용 상태 
        
    # comment_create 함수에서도 save 할 때에 바로 위의 article 을 읽어오는듯         
    article = Article.objects.get(pk=article_pk)
        serializer.save(article=article) 
```

- 바로 위에서 처리한 법도 물론 정석이지만, 시리얼라이저의 맨 마지막에서
read_only_fields 를 정의해주면 작동은 잘 하지만, 위의 왼쪽 이미지에서 
article 에 대한 정보를 보여줄 때에 title 을 같이 보여준 것과는 다르게, 
article 의 id만을 보여준다

```python
class CommentSerializer(serializers.ModelSerializer):
        class Meta:
            model = Comment
            fields = '__all__'
            read_only_fields = ('article',)
```

### 2. GET

- 게시글과 차이점 ? **참조 대상에 대한 처리**

```python

# 모든 댓글 조회 < 게시들 전체 조회와 동일 
    path('comments/', views.comment_list),

@api_view(['GET'])
def comment_list(request):
    comments = Comment.objects.all()
    serializer = CommentSerializer(comments, many=True)
    return Response(serializer.data)
    
# 댓글 상세 조회 < 게시글과 동일 댓글 pk 값으로 조회, 삭제, 수정
    path('comments/<int:comment_pk>/', views.comment_detail),

@api_view(['GET', 'DELETE', 'PUT'])
def comment_detail(request, comment_pk):
    comment = Comment.objects.get(pk=comment_pk)
    if request.method == 'GET':
        serializer = CommentSerializer(comment)
        return Response(serializer.data)
    elif request.method == 'DELETE':
        comment.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
    elif request.method == 'PUT':
        serializer = CommentSerializer(instance=comment, data=request.data, partial=True)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data) 
```

---

## 역참조 데이터 구성 :

### Article → Comment 간 역참조 관계 활용한 JSON 데이터 재구성 :

1. 단일 게시글 조회 시 해당 게시글에 작성된 **댓글 목록**  /  **댓글 갯수** 붙여 응답 

- 두개 다 view 변함 x / **시리얼라이저 변경** 

- 모든 댓글 목록 담고있는 comment_set 요소 추가 
  articles 에는 댓글 관련 컬럼이 없지만, 역참조를 실제로 한걸 확인 가능

```python
# 게시글 조회 할 때 해당 게시글의 댓글도 함께 조회
class ArticleSerializer(serializers.ModelSerializer):
    class CommentSerializer(serializers.ModelSerializer):
        class Meta:
            model = Comment
            fields = ('id', 'content',)               # 댓글의 id와 content만 보여준다.

		## 댓글 목록.. 
    **comment_set** = CommentSerializer(**many=True, read_only=True**) 
    # 바로 위에서 정의한 comment 시리얼라이저 이용해서 related_name 생성해 역참조
    # 나를 참조하는 댓글 많을 수 있으므로 many = True
    # read_only=True 옵션을 통해 해당 필드를 읽기 전용으로 설정

    # 댓글 갯수 표시
    # source 옵션을 통해 특정 필드를 지정하거나, 메서드를 호출할 수 있다.
    # field 새로 생성 + 댓글 개수이니까 intfield 사용 + 위에서 만들어놓은 set의 갯수  
    comment_count = **serializers.IntegerField**(source='comment_set.count', **read_only=True**)

		# 추가적인 필드 
		other_field = serializers.SerializerMethodField()

		def get_other_field(self, obj):
				return f'게시글 제목은 "{obj.title}" 이렇게 나타납니다.'

    class Meta:
        model = Article
        fields = '__all__'
        # 새롭게 추가한 set / count 추가해서 출력 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2482a135-cadd-4957-9248-054aba51b315/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/00e01396-ef54-4eef-9d0e-eb0bfafa663d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cffbf06d-1633-481b-99cd-d6b9e60a0ac4/Untitled.png)

### read_only_fields 속성과 read_only 인자의 사용처 :

- read_only_fields : 기존 외래 키 필드값을 그대로 응답 데이터에 제공하기 위해 지정하는 경우
- read_only : 기존 외래 키 필드값의 결과를 다른 값으로 덮어쓰는 경우
                  새로운 응답 데이터 값을 제공하는 경우

---

## Many to Many relationships :

- 한테이블의 0개 이상의 레코드가 다른 테이블의 0개 이상의 레코드와 관련된 경우 
**양쪽 모두에서 N:1** 관계를 가짐

### N:1 한계 :

### 중개 모델 :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c41cbc06-31f1-4d3a-917d-60d7fc609cad/Untitled.png)

### ManyToManyField : M:N 관계 설정 모델 필드

- Django 에서는 ManyToManyField 로 중개모델을 자동으로 생성
- courses / teachers APP 2개 M:N 관계

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/86145850-8af5-4d3f-8d36-54f2033967f7/Untitled.png)

- Teacher model << name 만
- Course model :

```python
from django.db import models
from teachers.models import Teacher

# Create your models here.
class Course(models.Model): # course / teacher n : 1  && n : m
    name = models.CharField(max_length=100)
    main_teacher = models.**ForeignKey(Teacher**, on_delete=models.CASCADE)
    #                               "teachers.Teacher" 로도 가능 < 이건 import 안해도 됨 
    assistant_teachers = models.ManyToManyField(Teacher)
    # assistant_teachers = models.ManyToManyField(Teacher, related_name="assistant_courses")
    # assistant_teachers = models.ManyToManyField(Teacher, related_name="assistant_courses", through='CourseInfo')

# 참조 : Through 참고 (교재에는 없음)
# N:M 관계에서 추가적인 정보를 저장하고 싶은 경우에는 중개 모델을 사용할 수 있음
# 이 경우에는 중개 모델을 정의하고 through 옵션에 중개 모델을 지정하면 됨
# 중개모델에는 반드시 관계 설정을 원하는 두 모델을 ForeignKey로 지정해야 함
class CourseInfo(models.Model):
    course = models.ForeignKey(Course, on_delete=models.CASCADE)  # Course 모델과의 관계
    teacher = models.ForeignKey(Teacher, on_delete=models.CASCADE)  # Teacher 모델과의 관계
    max_student = models.PositiveIntegerField(default=20)
    is_nessary = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ('course', 'teacher')
```

- 이러면 에러남 ; Add or change a related_name argument to the definition for 'courses.Course.assistant_teachers' or 'courses.Course.main_teacher'.

- 위와 같이 짜면 참조 받는 teacher 입장에서 역참조할게 2개라 충돌 
>> 둘중에 하나는 역참조명 넣어줘

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7fb6b9db-54aa-4a8a-a1a4-a8fde56517e4/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f275359f-fc61-428d-be60-15ff5273b61e/Untitled.png)

```python
assistant_teachers = models.ManyToManyField(Teacher, related_name="assistant_courses")
```

- 테이블 생성 가능해지고
각각의 테이블 + 중개 테이블 만들어진다  -  양쪽의 id 가지고 참조관계 나타내줌

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/ea849400-b063-4a4f-b2d2-d224be4243c3/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cfbe1135-d16b-42a8-b0ac-4815458700d4/Untitled.png)

- m:n 관계 나타내는 manytomany field 안만들어놓으면, courseinfo 객체로부터 정보 받아와야함
assistant_teachers 매니저 만들어 놓아서, 이거 써서 데이터 받기 쉬워짐

- 그럼에도 중개모델 써야하는 경우 ? 

추가 field 정의될 때에 > courseinfo table 에 추가 이후 
through 이용해서 CourseInfo 통해서 중개모델 만들 것이라고 선언

### ‘through’ argument :

```python
assistant_teachers = models.ManyToManyField(Teacher, 
                                            related_name="assistant_courses", 
                                            through='CourseInfo')
```

```python
# pjt urls
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("courses/", include("courses.urls")),
    path("teachers/", include("teachers.urls")),
]

# courses urls 
from django.urls import path
from . import views

urlpatterns = [
    path("<int:teacher_pk>/new/", views.create_course), # 강사 pk를 받아서 새로운 강좌를 생성하는 페이지로 이동
    path("<int:teacher_pk>/assistant/<int:course_pk>/", views.assistant),  # 강사 pk와 강좌 pk를 받아서 강좌의 부강사를 지정하는 페이지로 이동
]

# courses views 
@api_view(["POST"])
def assistant(request, course_pk, teacher_pk):
    '''
    path parameter로 받은 course_pk와 teacher_pk를 이용해 해당 강좌에 참여할 부강사를 지정
    이미 부강사로 지정되어 있는 경우에는 해당 강좌 부강사에서 제외
    부강사 지정/해제 후 강좌 정보를 반환
    M:N 관계에서는 add()와 remove() 메서드를 사용해 중개 모델을 통해 관계를 생성/해제
    '''
    course = get_object_or_404(Course, pk=course_pk) # course_pk에 해당하는 Course 객체를 가져옴
    teacher = get_object_or_404(Teacher, pk=teacher_pk) # teacher_pk에 해당하는 Teacher 객체를 가져옴
    if teacher in course.assistant_teachers.all(): # 이미 m:n 관계라면 (부강사로 지정되어 있는 경우
        **course.assistant_teachers.remove**(teacher) # 해당 강좌의 부강사에서 제외
    else: # 부강사로 지정되어 있지 않은 경우
        course.assistant_teachers**.add**(teacher) # 해당 강좌의 부강사로 지정
    return Response(CourseSerializer(course).data, status=status.HTTP_200_OK)
    
    
# courses 시리얼라이저

    # 여러 명의 부강사를 나타내는 필드이므로 many=True로 설정
    assistant_teachers = TeacherSerializer(many=True, read_only=True)
```

---

## 좋아요 기능 구현 :

### 모델 관계 설정 :

### 기능 구현 :

---

# Django Authentication :

## Cookie & Session :

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d0c04cda-86ce-4e6f-8f42-6c95075e9958/Untitled.png)

- 서버로부터 받은 페이지 둘러볼 때에, 서버와 연결되어 있는 상태 X

- **HTTP** : HTML 문서와 같은 **리소스들을 가져올 수 있도록 해주는 규약**
           웹(WWW) 에서 이뤄지는 모든 데이터 교환의 기초 
           
      특징 : 1. **비연결지향**(Connectionless) - 서버는 요청에 대한 응답 보낸 이후 연결 끊음
                 2. **무상태**(Stateless) - 연결 끊는순간 클라이언트<>서버 통신 끝나며, 상태정보 유지 X 
                                                    상태가 없다 - 장바구니 상품, 로그인 상태 유지 X 
                                                    >> 상태를 유지하기 위한 기술이 필요하다

---

### 쿠키 :

- 서버(S)가 사용자(C)의 웹 브라우저에 전송하는 작은 데이터 조각 
**클라이언트 측에서 저장**되며, 사용자 인증/추적/상태 유지 등에 사용되는 데이터 저장 방식 

**서버로부터 받은 이후에, 같은 서버의 다른 페이지로 재 요청시마다 받은 쿠키 함께 전송**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/94b2ff14-ccb2-4dc0-b084-30c5365b4ec5/ec6af7bd-cf51-449e-8c2f-01e4932fb0db.png)

- 사용 원리 : 

브라우저(클라이언트 C) 는 쿠키를 **K - V** 데이터 형식으로 저장 
이렇게 쿠키 저장해 놓았다가, **동일한 서버**에 **재요청** 시 저장된 **쿠키 함께 전송**

쿠키는 두 요청이 **동일한 브라우저**에서 들어왔는지 아닌지 **판단**할 때에 주로 사용됨 
- 이를 이용해 사용자의 **로그인 상태 유지** 가능 
- 상태가 없는 HTTP 프로토콜에서 **상태 정보를 기억**시켜주기 때문

- 예시 : 쿠팡 장바구니 담기 ( 로그인 X일 때 ) 

개발자도구 > Network 탭 > cartView.pang 확인 
서버는 응답과 함께 Set-Cookie 응답 헤더를 브라우저에게 전송
- 이 헤더는 클라이언트에게 쿠키 저장하라고 전달하는 것 

쿠키 데이터 자세히 확인하고, 메인 페이지 이동해도 유지되는지 확인 
개발자 도구 - Application tab - Cookies - 우클릭 - Clesr - 새로고침 > 장바구니 빔

- 사용 목적 : 
1. **세션 관리** : 정보관리 [ 로그인 / 아이디 자동완성 / 공지 하루 안보기 / 팝업 체크 / 장바구니 ]
2. 개인화 : 사용자 선호 / 테마 등의 설정 
3. 트래킹 : 사용자 행동 기록 및 분석 

- **세션**? : 서버 측에서 생성 / 클라이언트<>서버 상태 유지 / 상태정보 저장하는 데이터 저장방식
              **쿠키에 세션 데이터 저장하여 매 요청시마다 세션 데이터 함께 보냄** 

- 세션 작동 원리 : 
1. C 로그인 > S가 **session 데이터 생성 ( 서버의 DB에 )**  후 저장 
2. 생성된 session 데이터에 인증할 수 있는 session id 발급 > C에게 응답
3. C는 응답받은 **session id 쿠키에 저장** 
4. C가 다시 동일한 서버에 접속 > 요청과 함께 쿠키(session id 가 저장된) 을 서버에 전달
5. 쿠키는 요청 때마다 서버에 함께 전송 > 서버에서 **session id 확인해 로그인 여부** 알도록  함 

요약 :  S에서 세션 데이터 생성 후 저장 > 이 데이터에 접근 가능한 세선 ID 생성 
           이 ID를 C측으로 전달 > C는 쿠키에 이 ID 저장 
           이후 C가 같은 서버에 재요청 시마다 저장해뒀던 쿠키도 요청과 함께 전송 
(EX> 로그인 유지 위해 로그인 되어있다는 사실을 입증하는 데이터 매 요청마다 계속 보내는 것 )

```python
# PJT의 urls.py
path('accounts', include('accounts.urls'))  
# 세션 기반 인증 확인을 위한 URL, 후반부 과정에서 주석 처리 됨

# accounts APP의 urls.py 
from django.urls import path
from . import views
urlpatterns = [
    path("login/", views.login),
]
########################################################################################

# accounts APP의 models.py
# 장고에서 기본적으로 제공해주는 유저모델 상속받음
from django.db import models
from django.contrib.auth.models import AbstractUser
class User(AbstractUser):
    pass

########################################################################################

# accounts APP의 views.py
# **DRF 없이 장고 순수 기능만 우선 사용** 
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.forms import **AuthenticationForm**  # 장고 **내장 로그인 폼
# 시리얼라이저 아닌 폼 사용!**
from django.contrib.auth import **login** as auth_login  # 장고 내장 **로그인 함수**

# Session Authentication test function
@api_view(["POST"])
def login(request):
    form = **AuthenticationForm**(request, request.**POST**)  # 사용자가 입력한 데이터를 폼에 넣음
    if form.**is_valid():  # 유효성 검사**
        user = form.**get_user()**  # form에서 **user 객체를 가져옴**
        **auth_login**(request, user)  # **로그인 함수** 실행 >> **DB에 세션 만들어짐**
        session_id = request.session.session_key  # **세션 키를 가져옴 << 세션 생성확인**
        response_data = {
            'message': 'Login successful',
            'session_id': session_id
        }
        return Response(response_data, status=status.HTTP_200_OK)
    return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)

########################################################################################

# 이후 migrate > createsuperuser  
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5fd0b5db-ba0e-40e6-9710-8c2d0f64e877/Untitled.png)

- 세션 생성된 것 볼 수 있음.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/565f2314-04ce-4929-ac9b-9e92f715dd54/Untitled.png)

>>> 쿠키 & 세션의 목적 : S & C 간의 상태를 유지 

- 참고 
쿠키 종류별 수명 - session cookie  <> persistent cookies 
장고에서의 세션 : 굉장히 편하다~

---

## Authentication with DRF :

- 시작 전

```python
# 인증 로직 진행 위해 User 모델 관련 코드 활성화 
# articles/models.py
from django.db import models
from django.conf import settings
class Article(models.Model):
		"""
		ForeignKey to 방법 2개:Class 직접 전달 vs **"app.Model" 문자열** 호출 순서때문 후자추천 
		PJT의 settings 에서 installed_apps 테스트 시 articles > accounts 순서로 호출 가정
		articles 가 accounts 테이블 참조한다면 ? 올바른 실행 순서 아닐수도 
		
		여기서 to에 들어가는건 PJT의 settings.py 에 있는 요소. 우리가 직접 추가해 준 것 
		결국 동일한 형태인데 왜 이런식으로 ? 이미 장고의 내부 함수에 AUTH_USER_MODEL이 존재. 
		우리만의 User 모델 커스터마이징 위해서 하는데, pjt 진행하다가 중간에 커스터마이징한다면
		이런 과정들이 꼬이게 되어서 설정 힘들어짐 확장성을 생각해 ...  
		
		아래와 같은 코드 + accounts.models 에 설정해놓은 요소 check 
		"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4365e0f5-fd3f-4fdc-a465-724e147d6cd1/Untitled.png)

### 인증 ( Authentication )  :

- 수신된 요청을 해당 요청의 사용자 or 자격 증명과 연결하는 매커니즘 
 = **누구인지 확인**하는 과정

- 권한 ( Permissions ) : **요청에 대한 접근 허용 or 거부** 여부 결정

- 인증 & 권한 : 
- 순서상 **인증 먼저 진행**
- 수신 요청을 [ 해당 요청의 사용자 or 서명된 토큰과 같은 자격 증명 자료 ] 와 연결

- 그런 다음 권한 및 제한 정책 : 
인증이 완료된 해당 자격 증명을 사용해 요청을 허용해야 하는 지를 결정

- **DRF에서의 인증** : 
인증은 항상 **view 함수 시작 시, 권한 및 제한 확인이 발생하기 전, 
다른 코드의 진행이 허용되기 전**에 실행됨 

인증 자체로는 들어오는 요청 허용 or 거부 불가 
**단순히 요청에 사용된 자격 증명만 식별**한다는 점에 유의
- 승인되지 않은 응답 및 금지된 응답 : 
- 인증되지 않은 요청이 권한 거부하는 경우 해당되는 2가지 오류 코드 응답 
1. HTTP 401 Unauthorized : 요청된 리소스에 대한 유효한 인증 자격 증명 없기 때문에, 
                                     C 요청이 완료되지 않았음을 나타냄 ( **누구인지 증명할 자료가 없음** ) 

2. HTTP 403 Forbidden (Permission Denied) : 
- 서버에 요청 전달되었지만, 권한때문에 거절되었음을 의미 
- 401과 차이점 : 서버는 **C가 누구인질 알고있음**

- **Session 대신 Token 인증** 사용하는 이유 ? 
- 세션 기반 인증 : 서버에 세션 데이터 저장함 ( **상태 저장 방식** ( stateful ) ) 
- 서버가 사용자의 상태를 유지해야 
                   > 여러 서버 사용하는 **분산 시스템 구축/서버간의 부하 분산 어려움**

- 토큰 기반 인증 : **RESTful 한 방식으로 C와 S간의 독립성을 유지**하며 인증 처리할 수 있음 

1. C > S 인증 성공 > S가 넘겨받은 데이터 암호화(token 형식) > C에게 돌려줌 
2. 다음번 요청 시에 Token 과 함께 넘겨줌 > 서버가 받은 Token 복호화 > 인증정보 확인 
>> DB 공간 안쓰고 가능

### 인증 체계 설정 :

- 방법 2가지 : 
1. 전역 설정 
- https://www.django-rest-framework.org/api-guide/authentication/ 참고
- PJT의 settings : 아래 REST_FRAMEWORK 확인 

2. View 함수별 설정 :  아래 img 확인

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6476df70-131a-4a12-86f3-5291852de17f/Untitled.png)

```
REST_FRAMEWORK = {
    # Authentication
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.**TokenAuthentication**',
    ],
    # permission
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}
```

- **TokenAuthentication
-** token 기반 HTTP 인증 체계 : 기본 데스크탑 및 모바일 C와 같은 C-S 서버 설정에 적합 
- S가 인증된 C에게 토큰 발급 / C는 매 요청마다 발급받은 토큰 요청과 함께 보내 인증

---

### Token 인증 설정

- **TokenAuthentication 적용 과정** 
1. 인증 클래스 설정 
2. INSTALLED_APPS 추가 
3. Migrate 진행
4. 토큰 생성 코드 작성

- accounts 의 signals : 
특정한 상황 발생 시 아래의 함수 해라

```python
from django.db.models.signals import post_save # 유저 생성된 순간 감지 
from django.dispatch import receiver           # 유저 생성된 순간 감지 
from **rest_framework**.authtoken.models import **Token**
from django.conf import settings

# AUTH_USER에서 POST_SAVE 된거 감지했을 때에 함수 실행해라 
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created: # 유저 생성되면 유저정보 토대로 토큰 만들겠다 
        Token.objects.create(user=instance)
```

---

### Dj-Rest-Auth 라이브러리 :

- 회원가입 / 인증 / 비번 재설정 / 사용자 세부 검색 / 회원 정보 수정 / … 해주는 **써드파티 패키지**
- 타 라이브러리 비해 보안 / 업데이트 등등 좋음

```python
# PJT settings / urls 주석 풀기 
```

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8e0238c0-10aa-4447-ba65-50dd17768466/Untitled.png)

- 다양한 기능 확인 가능

### Token 발급 및 활용

- [http://127.0.0.1:8000/accounts/signup/](http://127.0.0.1:8000/accounts/login/) 이용해서 회원가입 
POST - username / password1 / password2 줘야
- http://127.0.0.1:8000/accounts/login/ 이용해서 로그인
POST - username / password 줘야 >> 토큰값 받아옴
- http://127.0.0.1:8000/api/v1/articles/ 에 title / content POST 
header 에 Authorization - Token 토큰값 안주면 에러
- 이랬을 때의 문제 ? 권한에 대한 설정이 x > 각 APP의 view 확인

### 권한 정책 설정 :

- 권한 설정 방법 : 
1. 전역 설정 : 

2. view 함수별 설정 :

```python
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny', # 
    ],
    
# permission Decorators
from rest_framework.decorators import permission_classes
from rest_framework.permissions import IsAuthenticated

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def article_list(request):
    if request.method == 'GET':
        articles = get_list_or_404(Article)
        serializer = ArticleListSerializer(articles, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            # serializer.save()
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
```

---

### DRF 가 제공하는 권한 정책 :

1. **IsAuthenticated 권한** 설정 :
- 인증되지 않은 사용자에 대한 권한 거부 / 인증되었으면 허용 
- 등록된 사용자만 API 엑세스 가능하도록 하려는 경우에 적합 
- 위에서 주석 해제한 것들 < 전체 게시글 조회 및 생성시에만 인증된 사용자인지 체크(권한)
2. IsAdminUser
3. IsAuthenticatedOrReadOnly
4. … 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f3368c35-821e-4bf6-b2e6-193f9ea029ec/Untitled.png)

---

### Django Signals :

- 이벤트 알림 시스템
- APP 내에서 특정 이벤트 발생 > 다른 부분에게 신호를 보내 이벤트 발생 여부 알리기 가능
- 주로 모델의 데이터 변경 / 저장 / 삭제 작업에 반응 > 추가적인 로직 실행하고자 할 때에 사용 
- EX> 사용자가 새로운 게시글 작성할 때마다 특정 작업 수행하려는 경우

### 금일 강의코드 체크 :

```python
## PJT의 settings
INSTALLED_APPS [ # 추가 또 있나?? + 어디부터 어디까지 어떤거인지 check 

     # Authentication and Authorization (인증 및 권한 부여):
    'rest_framework.authtoken',
    'dj_rest_auth',
    'dj_rest_auth.registration',

    'corsheaders',
    'django.contrib.sites', # 다중 사이트 관리 
    
     # Social Authentication (소셜 인증):
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
]

SITE_ID = 1 # 다중 사이트 관리 + 소셜 인증과의 통합 위해 사용 

# DRF 의 인증 및 권한 부여 시스템을 구성하는 설정
REST_FRAMEWORK = {

    # Authentication < 요청에 포함된 토큰을 사용하여 사용자를 인증
    'DEFAULT_AUTHENTICATION_CLASSES': [
		"""
		REST framework의 인증 방식 지정 
		여기에 설정된 클래스 : 요청 처리할 때 사용자의 인증 정보 확인
		
		현재 설정된 'rest_framework.authentication.TokenAuthentication' 클래스 : 
		
		토큰 기반 인증을 사용 > C 요청 보낼 때 HTTP 헤더에 인증 토큰을 포함시키면, 
		이 토큰을 통해 사용자의 인증 여부를 확인
		"""
        'rest_framework.authentication.TokenAuthentication',
    ],
    
    # permission < 모든 사용자에게 모든 리소스에 대한 접근을 허용
    'DEFAULT_PERMISSION_CLASSES': [
    """
    REST framework의 권한 부여 방식 지정 
    여기에 설정된 클래스 : 요청 처리할 때 사용자가 요청한 리소스에 접근권한이 있는지 확인
    
    현재 설정된 'rest_framework.permissions.AllowAny' 클래스
    모든 사용자에게 요청된 리소스에 대한 접근 허용
    즉, 인증 여부와 상관없이 모든 요청 허용
    """
        'rest_framework.permissions.AllowAny',
    ],
}

ACCOUNT_EMAIL_VERIFICATION = 'none'
 
MIDDLEWARE = [    # 추가
		'django.contrib.auth.middleware.AuthenticationMiddleware',
		# Django의 기본 인증 시스템을 지원 / 사용자 인증 정보를 request.user로 제공
		
		'allauth.account.middleware.AccountMiddleware',
		# 계정 관련 기능을 지원
    ]
    
AUTH_USER_MODEL = 'accounts.User'
"""
Django의 사용자 모델을 커스터마이즈하기 위해 설정하는 항목
기본적으로 Django는 auth.User라는 기본 사용자 모델을 제공
그러나 프로젝트의 요구사항에 맞게 사용자 모델을 변경해야 할 경우가 있습니다. 
이 설정을 통해 커스터마이즈된 사용자 모델을 사용하도록 Django에 지시할 수 있습니다.
"""

# PJT 의 URLS 
"""
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('articles.urls')),
    
    # path('accounts/', include('accounts.urls')),  
    # 세션 기반 인증 확인을 위한 URL, 

    path('accounts/', include('dj_rest_auth.urls')),
    path('accounts/signup/', include('dj_rest_auth.registration.urls')),
]
"""

# accounts 의 models
"""
from django.db import models
from django.contrib.auth.models import AbstractUser
class User(AbstractUser):
    pass
"""

# accounts 의 signals
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from rest_framework.authtoken.models import Token
from django.conf import settings

@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)
"""

# accounts 의 views
"""
from rest_framework.decorators import api_view
from rest_framework import status
from rest_framework.response import Response
from django.contrib.auth.forms import AuthenticationForm  # 장고 내장 로그인 폼
from django.contrib.auth import login as auth_login  # 장고 내장 로그인 함수

# Session Authentication test function
@api_view(["POST"])
def login(request):
    form = AuthenticationForm(request, request.POST)  # 사용자가 입력한 데이터를 폼에 넣음
    if form.is_valid():  # 유효성 검사
        user = form.get_user()  # form에서 user 객체를 가져옴
        auth_login(request, user)  # 로그인 함수 실행
        session_id = request.session.session_key  # 세션 키를 가져옴
        response_data = {
            'message': 'Login successful',
            'session_id': session_id
        }
        return Response(response_data, status=status.HTTP_200_OK)
    return Response(form.errors, status=status.HTTP_400_BAD_REQUEST)

"""

# articles 의 models
"""
from django.db import models
from django.conf import settings

class Article(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE
    )
    title = models.CharField(max_length=100)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

"""

# articles 의 serializers
"""
from rest_framework import serializers
from .models import Article

class ArticleListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ('id', 'title', 'content')

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = '__all__'
        read_only_fields = ('user',)
"""

# articles 의 urls
"""
from django.urls import path
from . import views

urlpatterns = [
    path('articles/', views.article_list),
    path('articles/<int:article_pk>/', views.article_detail),
]
"""

# articles 의 views
"""
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework import status

# permission Decorators
from rest_framework.decorators import permission_classes
from rest_framework.permissions import IsAuthenticated

from django.shortcuts import get_object_or_404, get_list_or_404

from .serializers import ArticleListSerializer, ArticleSerializer
from .models import Article

@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def article_list(request):
    if request.method == 'GET':
        articles = get_list_or_404(Article)
        serializer = ArticleListSerializer(articles, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = ArticleSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            # serializer.save()
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def article_detail(request, article_pk):
    article = get_object_or_404(Article, pk=article_pk)

    if request.method == 'GET':
        serializer = ArticleSerializer(article)
        print(serializer.data)
        return Response(serializer.data)

"""
```

---

# Vue

## Frontend Development - FE 개발

- 웹 사이트와 웹 app의 UI와 UX를 만들고 디자인하는 것
- HTML / CSS / JS 등을 활용해 사용자가 직접 상호작용하는 부분 개발

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e06458cf-0064-4ac7-bf80-5bfbb5a120a6/image.png)

### Client-side frameworks ( vue / react / anguler )

- Client 측에서 UI와 상호작용을 개발하기 위해 사용되는 **JS 기반 Framework**
- 필요 이유 : 웹에서 하는 일이 많아졌다
- 무언가를 읽는 곳 > 하는 곳 
음악 스트리밍 / 영화보기 / 텍스트, 영상채팅 ( 즉시 통신 ) 
이처럼 현대적이고 복잡한 대화형 웹 사이트 : **웹 페이지 X / WEB APP O** 
**JS 기반의 Client side Framework 등장하며 매우 동적인 대화형 app 구현 쉬워짐** 

- 다루는 데이터가 많아졌다 
facebook 에서 친구 이름 변경? 
친구 목록 / 타임라인 / 스토리 등 출력되는 모든 곳에서 함께 변경되어야 함 
→ app의 기본데이터 안정적으로 추적 & 업데이트 (랜더링, 추가, 삭제 등) 하는 도구필요
→ app의 상태를 변경할 때 마다 일치하도록 UI 업데이트 해야 함 

- Vanila JS도 가능은 하지만, 쉽지 않음

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3a85bc0d-833a-4718-9326-678a0755aced/image.png)

### SPA : Single Page Application - 단일 페이지로 구성된 app

- **하나의 HTML 파일**로 시작 
→ 사용자가 상호작용할 때마다 **페이지 전체 새로 로드X 화면의 필요한 부분만 동적갱신**
- 대부분 JS Framework 사용해 Client 측에서 UI와 렌더링 관리 ( CSR 방식 사용 )

### **CSR : Client - side Rendering** → 클라이언트에서 화면 랜더링 하는 방식

- 동작 과정 :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/68532ada-6b4e-44a1-93f1-5cab88699395/image.png)

1. 브라우저는 서버로부터 **최소한의 HTML 페이지와 해당 페이지에 필요한 JS** 응답 받음
2. 그런 다음 **Client측에서 JS를 사용하여 DOM을 업데이트 → 페이지 랜더링** 
3. 이후 서버는 **더이상 HTML을 제공하지 않고, 요청에 필요한 데이터 응답** 
- 구글맵스 / 페이스북 / 인스타 등의 서비스에서 페이지 갱신 시 새로고침 없는 이유 

- **장점**? :
1. 빠른 페이지 전환 : 
- 페이지가 **처음 로드된 후 → 필요한 데이터만** 가져오면 됨 
**JS는 전체 페이지 새로 고칠 필요 없이 페이지의 일부를 다시 랜더링할 수 있기 때문**
- **서버로 전송되는 데이터의 양을 최소화** ( 서버 부하 방지 ) 

지도 웹 APP : 
실시간 네비 지원한다면 출발 도착 찍고 경로 나오면 전세계 지도 X 
사용자가 움직이는 방향의 지도 데이터만 응답 
그와중에 출발, 도착은 랜더링 X 

2. 사용자 경험 : UX **새로고침 발생하지 않아** 네이티브 앱과 유사한 사용자 경험 제공 

1. FE / BE의 명확한 분리 : → 대규모 APP을 더 쉽게 개발 / 유지 관리 가능
- FE : **UI 렌더링 및 사용자 상호 작용 처리 담당** 
- BE : 데이터 및 API 제공 담당 
 

- **단점**? :
1. 느린 초기 로드 속도 : 넷플릭스 입장시 전체 페이지 보기 전에 약간의 지연 느낄 수 있음 
**JS가 다운로드 / 구분 분석 및 실행될 때까지 페이지가 완전히 랜더링 되지 않기 때문** 

1. **SEO ( 검색 엔진 최적화 ) 문제** : 
페이지를 나중에 그려 나가는 것이기 때문에, 검색에 잘 노출되지 않을 수 있음 
검색엔진 입장에서 HTML 읽어 분석해야 하는데, 아직 콘텐츠가 모두 존재하지 않아서

- SPA  VS  MPA   /    CSR  VS  SSR 
- MPA ( Multi Page App ) 
여러개의 HTML 파일이 서버로부터 각각 로드 
사용자가 다른 페이지로 이동할 때 마다 새로운 HTML 파일 로드됨 

- SSR ( Server-side Rendering ) 
서버에서 화면을 렌더링 하는 방식 
모든 데이터가 담긴 HTML을 서버에서 완성 후 Client에게 전달

## Vue & Tutorial

### What is Vue ( Vue 3 )

- **Vue.js : UI 구축 위한 JS Framework**
- Vue 학습 이유 : 
쉬운 학습 곡선 - 간결하고 직관적인 문법을 가지고 있어 빠르게 익힐 수 있음 
잘 정리된 문서 기반으로 어렵지 않게 학습 가능 

확장성과 생태계 - **다양한 플러그인과 라이브러리 제공**하는 높은 확장성 
전세계적으로 활성화된 커뮤니티 기반으로 많은 개발자들이 새로운 기능 개발 & 공유중

유연성 및 성능 - 작은 규모의 PJT ~ 대규모 APP 까지 다양한 PJT에 적합 

가장 주목받는 Client - side Framework

```jsx
<body>
  <div id="app">
    <h1>{{ message }}</h1>
    <button v-on:click="count++">
      Count is: {{ count }}
    </button>
  </div>

	// CDN 작성 
  <script src="https://unpkg.com/vue@3/dist/vue.global.js"></script>
  <script>
    const { createApp, ref } = Vue // Vue 객체 가져옴 -> 객체구조 분해하여 할당
    const app = createApp({ // 가져온 메서드 가지고 Vue 인스턴스 생성 {} 객체 하나 넘겨주는 형식 
      setup() { // 객체 내부에 setup 메서드 통해 인스턴스가 사용할 데이터셋 생성
      
        const message = ref('Hello vue!') // ref Obj 생성 
        const count = ref(0)
        
        return { // 셋업은 메서드이기 때문에 return 값 있어야 
          message, // 객체 단축 표현 적용된 모습 
          count // return한 변수 위의 html에서 변수명만 동일하게 응용하면 될듯 
        }
      }
    })
    app.mount('#app') // 만든 app인스턴스 -> css id가 app인 div 태그에 mount(올리기)
    // {{ }} 이용해 보낸 인스턴스의 리턴값들 사용하는 모습 확인 가능 
  </script>
</body>

</html>
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/98af4a78-a760-4845-9391-9170a0a6bcaf/image.png)

- Vue 개발자도구 이용해서 컨트롤도 가능 ( 확장 프로그램 설치 필요 )
- 새로고침 없어도 변함

- Vue의 2가지 핵심 기능 :
1. **선언적 렌더링** {{Declarative Rendering }} 
표준 HTML 확장하는 **템플릿 구문 
vue서 html에 넘겨준 변수 사용할 때에 {{}} 쓰는거**
사용해 JS 상태 ( 데이터 ) 를 기반으로 화면에 출력될 HTML **선언적으로** 작성 

2. **반응성** ( Reactivity ) : 
**JS 상태 ( 데이터 ) 변경 추적 & 변경사항 발생하면 자동으로 DOM 업데이트** 
위의 코드에서 클릭하면 숫자 변하는 것

### Vue Style Guide

- Vue의 스타일 가이드 규칙은 우선순위에 따라 4가지 규칙 범주로 나눔
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dfe09115-733e-41c7-ba84-6511d9908492/image.png)
    

A : 오류 방지하는 데에 도움이 되므로, 어떤 경우에도 규칙 학습하고 준수 

B : 가독성 & 개발자 경험 향상시킴 
규칙 어겨도 코드는 여전히 실행되겠지만, 정당한 사유 있어야 규칙 위반할 수 있음 

C : 일관성을 보장하도록 임의의 선택을 할 수 있음 

D : 잠재적 위험 특성 고려 

- Vue 사용 방법 :
1. CDN 방식 : 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fe030ac0-8175-4903-9d52-95deb5a926ff/image.png)

```python
<body>
  <div id = "app">
    <h1>{{message}}</h1>
    <!--뷰가 랜더링할 때에 message.value를 message로만 적어서 html에 써도 
    value를 띄워줌 오히려 message.value 하면 이상한 값이 나옴  -->
  </div>

  <script **src="https://unpkg.com/vue@3/dist/vue.global.js"**></script>
  <script>
    const {createApp, ref } = Vue
    const app = createApp({
      setup() {
        const message = ref('hello, world!')
        console.log(message) // 객체 형태의 정보 확인 
        console.log(message.value) // 위에서 선언한 문자열 가져옴 
        return {
          //template 영역에서 사용할 각종 변수와 함수 반환 
          // message : message 단축 가능 
          message,
        }
      }
    })
    app.mount('#app')    // Vue 인스턴스 하나당 한번의 mount 만 진행 
  </script>
</body>
</html>
```

- ***const로 생성해서 넘겼으므로, 개발자 도구에서도 글자 변경 불가 
let도 조금 문제가 있어서, ref를 사용***

- ref ( ) : 반응형 상태를 선언하는 함수
- 교재 내용 추가 ㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱㄱ

1. NPM 설치 방식 : Node Package Manager 

### Component : 재사용 가능한 코드 블록

- 특징 : UI 독립적&재사용 가능한 일부분으로 분할 / 각 부분을 개별적으로 다룰 수 있음
           자연스럽게 APP은 중첩된 Component의 트리 형태로 구성
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1ac14ade-e073-440b-984c-165db3b90685/image.png)
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/d74653e3-ff25-404c-b6f1-6479577086b3/image.png)
    

## SFC ( Single File Components )

- 컴포넌트의 템플릿, 로직 및 스타일을 하나의 파일로 묶어낸 특수한 파일 형식 .vue파일
    - **.vue** 파일은 3가지 유형의 최상위 언어 블록 가짐
    - **template / script / style** 로 구성 & 순서도 왼쪽과 같이
    
    ### SFC 문법
    
    1. **<template>** : 각 vue 파일에 무조건 **한개만** / html 형식으로 입력 
    2. <script> : 본래 HTML에서는 JS 입력 가능한 부분 
    
    <**script setup**> ** 블록은 무조건 **하나만**! 
    **컴포넌트의 setup 함수로 사용**  &  컴포넌트의 ***각 인스턴스에 대해 실행*** 
    변수 및 함수는 동일한 컴포넌트의 템플릿에서 자동으로 사용 가능 
    
    1. <style> : 여러개 사용 가능 / **scoped 옵션** 지정되면 CSS는 현재 컴포넌트에만 적용 
    
    ```jsx
    <!-- MyComponent.vue -->
    
    <template>
      <div class="greeting">{{ msg }}</div>
    </template>
    
    <script setup>
    import { ref } from 'vue'
    const msg = ref('Hello World!')
    </script>
    
    <style scoped>
    .greeting {  color: red;}
    </style>
    ```
    
    - 컴포넌트 사용하기 : https://play.vuejs.org/ 에서 Vue컴포넌트 코드 작성 및 미리보기
    - Vue SFC는 일반적인 방법으로 실행 X / 컴파일러 통해 컴파일 된 후 빌드되어야 함 
        ㄴ 실제 PJT에서는 Vite와 같은 공식 빌드 도구 이용
    
    ## Node JS
    
    - Chrome의 V8 JS엔진을 기반으로 하는 Server-Side 실행 환경
    - 기존에 브라우저 안에서만 동작 가능했던 JS를 서버 측에서도 실행할 수 있게 함 
    FE / BE에서 동일한 언어로 개발할 수 있게 됨
    - NPM : Node Package Manager : Node.js의 기본 패키지 관리자
    - ㄴ 활용해 많은 오픈소스 패키지&라이브러리 제공 → 개발자들 쉽게 코드 공유 & 재사용
    

## SFC build tool

### 모듈과 번들러

- 모듈 - 프로그램 구성하는 독립적인 코드 블록 ( .js 파일 ) 
ㄴ 필요성 : 개발하는 app 크기 커지고 복잡 → 파일 하나에 모든 기능 담기 어려움
                  자연스럽게 파일 여러개로 분리해 관리 → 분리된 각 파일이 모듈 
ㄴ 한계 : app 발전함에 따라 처리해야 하는 **JS 모듈의 개수 극적으로 증가**
              성능 병목 현상 발생 / **모듈간의 의존성** 깊어짐 → 문제 근원 찾기 어려워짐 
              복잡 & 깊은 모듈 간 의존성 문제 **해결하기 위한 도구 필요 (Bundler**)
- 번들러 - 여러 모듈 & 파일 하나 or 여러개의 번들로 묶어 최적화 ( Bundling )  
               → app에서 사용할 수 있게 만들어주는 도구 
ㄴ 역할 : 의존성 관리 / 코드 최적화 / 리소스 관리 등… 
ㄴ **Vite : Rollup 번들러** 사용 → 개발자가 별도 기타 환경설정 신경쓰지 않도록 설정해둠
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/aa6beb31-f0f5-4173-a597-0347a33fa0ae/image.png)
    
- 모듈화 통해 타 모듈의 객체 함수 import 해오는 형식 
( **파이썬과 from / import 순서 반대** )

**넘겨주는 쪽에서 이런 식으로 export** / 객체 형태로 반환하면 객체 형태로 받아야
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1474c57e-1975-4adc-b20f-b299e2620cc6/image.png)
    

- 모듈 형태로 사용 원한다면 package.json 영역에 아래와 같이 선언해 놔야 사용 가능하다
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/28b05edb-8c4e-4349-9310-9b59edd47b1e/image.png)
    

- export 받을 대상 명확히 안정해주면, default 로 설정해둔 요소가 보내짐

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4546c2b7-92b5-4805-beac-c1140683c76e/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7012c08b-2323-43a5-aaca-0578c86bd6bb/image.png)

### Vite

- 프론트 엔드 개발 도구
- 빠른 개발 환경 위한 **빌드 도구 / 개발 서버** 제공

- 빌드 : 
- PJT의 소스코드 최적화 & 번들링 → 배포할 수 있는 형식으로 변환하는 과정 
- 개발중에 사용되는 여러 소스 & 리소스 ( JS / CSS/ img ) 최적화된 형태로 조합 
     → 최종 SW제품 생성 
- Vite : 이러한 빌드 프로세스 수행하는 데에 사용되는 도구

## Vue PJT

- PJT 생성 :

```python
npm create vue@latest
npm install 
"""
npm install → 패키지 설치되면서 생성
장고 venv = node_modules 폴더
- vite / rollup / … 

장고 requirements =  바로 아래의 json 파일에 적혀 있음 
  ㄴ freeze 필요없이 알아서
  ㄴ gitignore도 알아서
"""
npm run dev # 서버 실행  ( http://localhost:5173/ )
"""
npm install 진행 후 audit 뜨는 이유? 
package-lock에 있는 버전들이 최신이 아닐 때에 
"""
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a4a013fc-9ddb-439f-ab3b-3acdaf87dbad/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/12f4d1cc-9b4d-43d1-a96e-ce337fb40be5/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/729eb009-9bfb-482f-9b9c-691f19e97e9c/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a39786d9-0029-43b9-bd76-4977bd9c9be5/image.png)

### PJT 구조

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b205b2cc-1727-4f78-a114-20f5e7754580/image.png)

- public dir : 정적인 파일들 → 소스코드에서 참조하지 않을 내용물들 / 항상 같은 형태를 유지할 파일들 import 할 필요 없는 것들 → 사용 잘 X / favicon.ico : Vue 로고

 

- **src dir : 가장 많이 사용** / App.vue / components 폴더 자주 사용  

- **App.vue** : 컴포넌트 중 **최상위 컴포넌트** 
components 폴더에서 만들어져 있는 vue 모듈들을 가져와서 사용 

- **main.js : 컴포넌트가 화면에 어떻게 그려지는지** 다루는 부분 
**node_modules에서 vue / App.vue에서 객체 들고와** 
html의 id app인 요소에 mount 하는 형식 / 기존 html은 index.html에 저장되어 있 
<div id="app"></div> 이 부분에서 app이라는 **id에 컴포넌트를 mount** 하는 느낌 

- assets 폴더 : css / image 

- components 폴더 : 각 요소들에 해당하는 vue 파일 존재 
             ㄴ 컴포넌트 명 : PascalCase로 
             ㄴ 기본 컴포넌트 구조 : extension sniffet 이용해 기본 구조 불러오기 가능

```python
<!-- ctrl + shift + p -> snippets -> vue.json 수정 -->
<template>
  <div>

  </div>
</template>

<script setup>

</script>

<style scoped>

</style>
###################################################################
{
	// Place your snippets for vue here. Each snippet is defined under a snippet name and has a prefix, body and 
	// description. The prefix is what is used to trigger the snippet and the body will be expanded and inserted. Possible variables are:
	// $1, $2 for tab stops, $0 for the final cursor position, and ${1:label}, ${2:another} for placeholders. Placeholders with the 
	// same ids are connected.
	// Example:
	"Vue for SSAFY": {
		"prefix": "vue",
		"body": [
			"<template>",
				"  <div>",
				"",
				"  </div>",
			"</template>",
			"",
			"<script setup>",
			"",			
			"</script>",
			"",
			"<style scoped>",
			"",
			"</style>",
		],
		"description": "Log output to console"
	}
}

```

- PJT 초기화 : 
assets 지우고, components 에 필요없는 파일 / 폴더들 지우고 
app.vue에 기본 구조 빼고 다 날리고 
main.js에 assets import 해오는거 지우고

## Vue Component 활용

- Scripts setup에서 import 받아옴
- scoped 태그여서 : style은 하위 vue에는 영향 미치지 않는다 ( 본인만 영향받음 )

```jsx
#################################################
<!-- App.vue -->

<template>
  <h1>App Vue</h1>
  <MyComponenet />
</template>

<script setup>
import MyComponenet from './components/MyComponent.vue'
</script>

<style scoped>
  p{    color: red;  }
</style>

#################################################

<!-- MyComponent.vue -->

<template>
  <div>
    <h2>MyComponenet2</h2>
    <p>{{ message }}</p>
    <MyComponenetItem />
    <MyComponenetItem />
    <MyComponenetItem />
    <MyComponenetItem />
  </div>
</template>

<script setup>
  import {ref} from 'vue'
  const message = ref('hello, world!')
  import MyComponenetItem from '@/components/MyComponentItem.vue' 
                             // @ == src 
</script>

<style scoped>

</style>
#################################################
<!-- MyComponentItem.vue -->
<template>
  <div>
    <p>안녕하세요</p>
  </div>
</template>

<script setup>
</script>

<style scoped>
</style>
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/adac9705-2f9a-4849-9c80-22167b2a919f/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/65b99381-8903-478e-b6b3-bcf65a1f58a2/image.png)

### Vue 작성하는 2가지 스타일 ?

1. Composition API : Vue3 권장 
- import 해서 가져온 API 함수들을 사용해 컴포넌트의 로직 정의
- JS문법 그대로 사용

1. Option API : Vue2 까지 쓴거 ( 복잡 ) 

## Template Syntax :

- 의미 : DOM을 기본 구성 요소 인스턴스의 데이터에 선언적으로 **바인딩 ( VUE의 인스턴스와 DOM을 연결 )** 할 수 있는 HTML 기반 템플릿 구문( 확장된 문법 제공 ) 사용

- 종류 :
1. Text Interpolation : 문자열 보간법 
- 데이터 바인딩의 가장 기본적 형태 
- 콧수염 구문 : 이중 중괄호 구문 / 해당 구성 요소 인스턴스의 msg 속성 값으로 대체 
- **msg  속성이 변경될 때 마다 업데이트** 됨 
- <p>Message : {{msg}} </p>

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/db9c61a1-5410-45ac-9902-1a8668ecce30/image.png)

1. Raw HTML : v-html
- 콧수염 구문은 데이터를 일반 텍스트로 해석 → 실제 HTML 출력은 v-html을 사용해야 함 
- div에 attribute 주듯 
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/7cb43d34-4a28-417c-b274-51a0733f3dbe/image.png)
    

1. Attribute Bindings : v-bind
- HTML Attribute에 내가 원하는 값 넣는 느낌 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/384fd05f-3564-4bb5-9a27-c0be7315d762/image.png)

1. JS Expressions : 데이터 바인딩 내에서 JS의 모든 표현식 지원 
- 콧수염 구문 내부
- V-로 시작하는 특수 속성의 값 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a123dcf1-d14a-41f5-ae7a-b54e8d360e7b/image.png)

- 표현식 주의사항 : 각 바인딩에는 하나의 단일 표현식만

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5bc232b5-eba3-412b-8508-7d34df3928de/image.png)

### Directive : v- 접두사가 있는 특수 속성

- 특징 : 속성값이 단일 JS 표현식이어야 함 **[ v-for / v-on 제외 ]** 
          표현식 값이 변경될 때 DOM이 반응적으로 업데이트 적용
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8520c378-b7dd-4db5-8e37-288b05de05de/image.png)
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6143d296-f1b8-43e0-b03a-72c060ebe83f/image.png)

- submit 이벤트 발생 → value에 넣은 표현식의 평가 결과를 실행하겠다
modifiers : preventdefault와 같이 html의 기본 event 동작 취소하는 메서드를 한번에

- 일부 directive만 : 사용 가능 /

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/e65eea0c-b569-4106-884f-91ccbdab51f5/image.png)

- 시작 순서 :

npm install : 만들어진 vue pjt 에서 패키지 설치 

npm run dev : vue로 만든 서버 실행 

```jsx
// app.vue
<template>
  <div class="wrapper">
    <header class="sidebar">
    <!-- v-for 이용해 list로 되어 있는 객체들 버튼으로 만들어 주는 모습.
         각 for문이 돌때마다 components가 호출되는느낌?   -->
      <div class="nav" v-for="item in componentsList" :key="item.name">
        <button @click="selected = item.name">{{item.name}}</button>
      </div>
    </header>
    <section class="content">
      <component :is="currentComponent" v-if="currentComponent"></component>
    </section>
  </div>
</template>

<script setup>
import { defineAsyncComponent, ref, computed } from 'vue'

const componentsList = [
  { directoryName: '01_basic_syntax', name: 'TemplateSyntax' },
  { directoryName: '02_bind', name: 'VBind' },
  { directoryName: '02_bind', name: 'BindingHTMLClasses' },
  { directoryName: '02_bind', name: 'BindingInlineStyles' },
  { directoryName: '03_event', name: 'EventHandling' },
  { directoryName: '03_event', name: 'FormInputBinding' },
  { directoryName: '04_v_model', name: 'VModel' },
  { directoryName: '05_conditional_rendering', name: 'ConditionalRendering' },
  { directoryName: '06_list_rendering', name: 'ListRendering' },
  { directoryName: '06_list_rendering', name: 'RenderingWithKey' },
  { directoryName: '06_list_rendering', name: 'RenderingWithIf' }
]
```

## Dynamically data binding :

### v-bind

- 동적 데이터 바인딩을 가능하게 함
- 하나 이상의 속성 or 컴포넌트 데이터를 표현식에 동적으로 바인딩

1. **Attribute Bindings**
- HTML의 속성 값을 Vue의 상태 속성 값과 동기화 되도록 함
shortcut : 콜론(:)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/65139734-84a9-4daf-bf85-fd6e103706fd/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f3715ffd-5973-4755-a4aa-82d215fd73a8/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cfaf8f9c-8226-4d91-8187-05d898a8255f/image.png)

1. Class and Style Bindings
- Class, Style = HTML 속성 → 다른 속성과 동일 v-bind 사용 → 동적으로 문자열 값 할당 가능
- Vue : class, style 속성 값을 v-bind로 사용할 때 **객체 또는 배열을 활용**해 작성할 수 있도록 
( 단순히 문자열 연결을 사용해 이러한 값 생성하는 것 번거롭고, 오류 발생하기 쉽기 때문 )

2-1. Binding HTML Classes : 

- Binding to Objects & Arrays : 객체를 :class에 전달해 클래스를 동적으로 전환할 수 있음

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/83c2350a-eaad-44ab-85a7-8a6c0a7ee90d/image.png)

```jsx
<template>
  <div>
    <!-- Binding to Objects -->
    <div :class="{ active: isActive }">Text</div>
    <div class="static" :class="{ active: isActive, 'text-primary': hasInfo }">Text</div>
    <div class="static" :class="classObj">Text</div>

    <!-- Binding to Arrays -->
    <div :class="[activeClass, infoClass]">Text</div>
    <div :class="[{ active: isActive }, infoClass]">Text</div>
  </div>
</template>

<script setup>
  import { ref } from 'vue'
  const isActive = ref(false)
  const hasInfo = ref(true)
  
  // ref는 반응 객체의 속성으로 액세스되거나 변경될 때 자동으로 래핑 해제
  const classObj = ref({
    active: isActive,
    'text-primary': hasInfo
  })
  const activeClass = ref('active')
  const infoClass = ref('text-primary')
</script>
```

2-2. Binding Inline Styles :  

Binding to Objects & Arrays : 

```jsx
<template>
  <div>
    <!-- Binding to Objects -->
    <div :style="{ color: activeColor, fontSize: fontSize + 'px' }">Text</div>
    <div :style="{ 'font-size': fontSize + 'px' }">Text</div>
    <div :style="styleObj">Text</div>

    <!-- Binding to Arrays -->
    <div :style="[styleObj, styleObj2]">Text</div>
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  const activeColor = ref('crimson')
  const fontSize = ref(50)

  const styleObj = ref({
    color: activeColor,
    fontSize: fontSize.value + 'px'
  })
  
  const styleObj2 = ref({
    color: 'blue',
    border: '1px solid black'
  })
</script>

<style scoped>

</style>
```

## Event Handling :

### v-on : DOM 요소에 이벤트 리스너를 연결 및 수신 ( addEventListner )

- v-on:event=”handler”
shortcut : @event=”handler

- handler 종류 :
1. Inline handlers : 이벤트가 트리거 될 때 실행 될 JS 코드 / 간단한 상황에 사용 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/256f6aa5-8912-42e2-9d35-d1a211dfadbb/image.png)

2. Method handlers : 컴포넌트에 정의된 메서드 이름 / 인라인 불가능할 시 사용 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8fd76fa5-72c4-40fe-a29e-980b57968e42/image.png)

```jsx
<template>
  <div>
    <!-- Inline Handlers -->
    <button @click="count++">Add 1</button>
    <p>Count: {{ count }}</p>

    <!-- Method Handlers -->
    <button @click="myFunc">Hello</button>

    <!-- Calling Methods in Inline Handlers -->
    <button @click="greeting('hello')">Say hello</button>
    <button @click="greeting('bye')">Say bye</button>

    <!-- Accessing Event Argument in Inline Handlers -->
    <button @click="warning('경고입니다.', $event)">Submit</button>
<!-- 
$ 접두어가 붙은 변수 : Vue 인스턴스 내에서 제공되는 내부 변수
사용자가 지정한 반응형 변수나 메서드 구분 위해
주로 Vue 인스턴스 내부 상태 다룰 때에 사용 -> 유저가 덮어쓰기하는거 방지
-->
    <!-- event modifiers -->
    <form @submit.prevent="onSubmit">...</form>
    <a @click.stop.prevent="onLink">...</a>

    <!-- key modifiers -->
    <input @keyup.enter="onSubmit">
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  const count = ref(0)

  const name = ref('Alice')
  const myFunc = function (event) { // 여기서 핸들러에게서 온 이벤트 받음 
    console.log(event)
    console.log(event.currentTarget)
    console.log(`Hello **${name.value}**!`)
  }

  const greeting = function (message) { // 넘겨받은 인자 따라서 메세지 출력 
    console.log(message)
  }
  
  const warning = function (message, event) {
    console.log(message)
    console.log(event)
  }
</script>

<style scoped>

</style>
```

- Event Modifiers :
1. Vue는 v-on에 대한 event modifiers 제공 →  event.preventDefault() 와 같은 구문을 메서드에서 작성하지 않도록 함 
2. stop / prevent / self 등 다양한 modifiers 제공 
3. 메서드는 DOM 이벤트에 대한 처리보다 데이터에 관한 논리를 작성하는 것에 집중할 것 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/f69d94f5-c074-4949-89e8-65c983bed386/image.png)

1. Key Modifiers : vue는 키보드 이벤트 수신 시 특정 키에 관한 별도 modifiers를 사용 가능 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/20024736-6b2d-4466-8ca7-c633341f5bb6/image.png)

## Form Input Bindings : 양방향 바인딩

- form 처리시 사용자가 input에 입력하는 값 실시간으로 JS 상태에 동기화해야 하는 경우

- 방법 :

1. v-bind & v-on 함께 사용
- v-bind 사용해 input 요소의 value 속성 값을 입력 값으로 사용
- v-on 사용해 input 이벤트 발생 시 마다 
  input 요소의 value 값을 별도 반응형 변수에 저장하는 핸들러 호출 

```jsx
<template>
  <div>
    <p>{{ inputText1 }}</p>
    <input :value="inputText1" @input="onInput">

    <p>{{ inputText2 }}</p>
    <input v-model="inputText2">
  </div>
</template>

<script setup>
  import { ref } from 'vue'
  
  const inputText1 = ref('')
  const onInput = function (event) {
    inputText1.value = event.currentTarget.value
  }
  
  const inputText2 = ref('')
  
</script>
```

 

1. v-model 사용

### v-model :

- form input 요소 또는 컴포넌트에서 양방향 바인딩을 만듦
- 사용자 입력 데이터 / 반응형 변수를 실시간 동기화
- IME 가 필요한 언어 ( 일어 중국어 한국어 ) V-MODEL 업데이트 제대로 X 
→ 해당 언어에 대해 올바른 응답 위해 1번 방법 써야
- 하지만 text뿐 아니라 checkbox / radio / select 등 다양한 타입의 입력에서 사용 가능하기 때문에 한글 이외에는 활용 잘 가능

```jsx
<template>
  <div>
    <!-- single checkbox -->
    <input type="checkbox" id="checkbox" v-model="checked">
    <label for="checkbox">{{ checked }}</label>

    <!-- multiple checkbox -->
    <div>Checked names: {{ checkedNames }}</div>

    <input type="checkbox" id="alice" value="Alice" v-model="checkedNames">
    <label for="alice">Alice</label>

    <input type="checkbox" id="bella" value="Bella" v-model="checkedNames">
    <label for="bella">Bella</label>

    <!-- single select -->
    <div>Selected: {{ selected }}</div>

    <select v-model="selected">
      <option disabled value="">Please select one</option>
      <option>Alice</option>
      <option>Bella</option>
      <option>Cathy</option>
    </select>
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  const checked = ref(false)
  const checkedNames = ref([])
  const selected = ref('')
</script>
```

## Conditional Rendering : 조건문

### v-if / v-else / v-else-if :

- 표현식 값의 tf를 기반으로 요소를 조건부 랜더링
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/75577527-b929-400a-99b4-ec9add72e351/image.png)
    

```jsx
<template>
  <div>
    <!-- if else -->
    <p v-if="isSeen">true일때 보여요</p>
    <p v-else>false일때 보여요</p>
    <button @click="isSeen = !isSeen">토글</button>

    <!-- else if -->
    <div v-if="name === 'Alice'">Alice입니다</div>
    <div v-else-if="name === 'Bella'">Bella입니다</div>
    <div v-else-if="name === 'Cathy'">Cathy입니다</div>
    <div v-else>아무도 아닙니다.</div>

    <!-- v-if on <template> -->
    <template v-if="name === 'Cathy'">
      <div>Cathy입니다</div>
      <div>나이는 30살입니다</div>
    </template>

    <!-- v-show -->
    <div v-show="isShow">v-show</div>
    <button @click="isShow = isShow">토글</button>
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  const isSeen = ref(true)

  const name = ref('Cathy')
  
  const isShow = ref(false)
</script>

<style scoped>

</style>
```

### v-if   VS   v-show

- v-show : 표현식 값의 tf 기반으로 요소의 **가시성 전환**
- if 는 조건 틀리면 랜더링조차 되지 않음
- v-show는 스위치같은 느낌 ( 랜더링은 무조건 됨 )

## List Rendering : 반복문

### v-for

- 소스 데이터 기반으로 요소 / 템플릿 블록을 여러 번 랜더링
- alias in expression 형식의 특수 구문 사용

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/dc02e490-2c7a-4c56-996b-42260ba322a0/image.png)

- 인덱스 ( 객체에서는 키 ) 에 대한 별칭 지정 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/17fdbf90-b419-4169-888b-799189e44afd/image.png)

```jsx
<template>
  <div>
    <div v-for="(item, index) in myArr">
      {{ index }} / {{ item**.name** }}
    </div>
    <div v-for="(value, key, index) in myObj">
      {{ index }} / {{ key }} / {{ value }}
    </div>

    <!-- v-for on <template> -->
    <ul>
      <template v-for="item in myArr">
        <li>{{ item.name }}</li>
        <li>{{ item.age }}</li>
        <hr>
      </template>
    </ul>

    <!-- nested v-for -->
    <ul v-for="item in myInfo">
      <li v-for="friend in item.friends">
        {{ item.name }} - {{ friend }}
      </li>
    </ul>
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  const myArr = ref([
    { name: 'Alice', age: 20 },
    { name: 'Bella', age: 21 }
  ])
  
  const myObj = ref({
    name: 'Cathy',
    age: 30
  })

  // nested v-for
  const myInfo = ref([
    { name: 'Alice', age: 20, friends: ['Bella', 'Cathy', 'Dan'] },
    { name: 'Bella', age: 21, friends: ['Alice', 'Cathy'] }
  ])
</script>
```

- 반드시 v-for와  key를 함께 사용한다 
→ 내부 컴포넌트의 상태를 일관되게 하여 데이터의 에측 가능한 행동 유지하기 위함
- key : 반드시 각 요소에 대한 고유한 값을 나타낼 수 있는 식별자여야 함 
- number / string 으로만 사용해야 
- vue 내부 가상 DOM 알고리즘이 이전 목록과 새 노드 목록 비교할 때 각 NODE를 식별하는 용도로 사용
- Vue 내부 동작 관련 부분이기에, 최대한 작성하려고 노력할 것

```jsx
<template>
  <div>
    <!-- Maintaining State with key -->
    <div v-for="item in items" :key="item.id">
      <p>{{ item.id }}</p>
      <p>{{ item.name }}</p>
    </div>
  </div>
</template>

<script setup>
  import { ref } from 'vue'

  let id = 0

  const items = ref([
    { id: id++, name: 'Alice' },
    { id: id++, name: 'Bella' },
  ])
</script>

<style scoped>

</style>
```

- 동일 요소에 v-for / v-if 함께 사용하지 않는다
**동일한 요소에서 v-if가 우선순위 더 높기 때문**

```jsx
<template>
  <div>
    <!-- [Bad] v-for with v-if -->
    <!-- 에러뜸. if가 더 세서 <ul>
      <li v-for="todo in todos" v-if="!todo.isComplete" :key="todo.id">
        {{ todo.name }}
      </li>
    </ul> -->

    <!-- [Good] v-for with v-if (computed)-->
    <ul>
      <li v-for="todo in completeTodos" :key="todo.id">
        {{ todo.name }}
      </li>
    </ul>

    <!-- [Good] v-for with v-if  << template 요소 사용해 v-if 위치 이동 
    근데 이거보다는 금요일날 배우는 computed 많이 사용함 -->
    <ul>
      <template v-for="todo in todos" :key="todo.id">
        <li v-if="!todo.isComplete">
          {{ todo.name }}
        </li>
      </template>
    </ul>
  </div> 
</template>

<script setup>
  import { ref, computed } from 'vue'

  let id = 0

  const todos = ref([
    { id: id++, name: '복습', isComplete: true },
    { id: id++, name: '예습', isComplete: false },
    { id: id++, name: '저녁식사', isComplete: true },
    { id: id++, name: '노래방', isComplete: false }
  ])

  const completeTodos = computed(() => {
    return todos.value.filter((todo) => !todo.isComplete)
  })
</script>

<style scoped>

</style>
```

- 배열 변경 관련 메서드 : v-for과 배열 함께 사용 시 메서드 주의해서 사용해야

1. 변화 메서드 : 호출하는 원본 배열 변경 
- push pip shift unshift splice sort reverse

1. 배열 교체 : 원본 배열 수정하지 않고 항상 새 배열 반환
- filter concat slice

- v-for 와 배열 이용해 “필터링 / 정렬’ 활용 : 
- 원본 데이터 수정/교체 하지 않고 필터링하거나 정렬된 새로운 데이터 표시하는 방법

1. computed 활용

1. 메서드 활용 ( computed가 불가능한 중첩된 v-for 의 경우 사용 ) 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8d79e8e3-705c-4b64-be16-e614a89571db/image.png)

## Passing Props

- 같은 데이터지만 다른 컴포넌트에 있다면 ( ex> facebook의 profile 사진 )

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/88d4185c-82cb-4522-a928-7c9bed3d8df8/image.png)

해당 페이지를 구성하는 컴포넌트 각각이 개별적으로 동일한 데이터 관리? 
→ 사진 변경할 때에 모든 컴포넌트에 대해 변경요청 해야 하므로 까다로움 
→ **공통된 부모 컴포넌트에서 관리하자** 

- Passing Props : 부모는 자식에게 **데이터를 전달 ( Passing Props )** 하며, 
자식은 자신에게 일어난 일을 **부모에게 알림 ( Emit Events )**

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c8de8666-0200-4f31-9762-19bddb9564f7/image.png)

### Props

- 부모 컴포넌트로부터 자식 컴포넌트로 데이터 전달하는 데에 사용되는 속성

- 특징 : 
부모 속성 업데이트 → 자식으로 전달 ( 반대는 XX ) 
즉, 자식 컴포넌트 내부에서 props 변경하려고 시도해서는 안되며, 불가능
부모 컴포넌트 업데이트 → 사용하는 자식 컴포넌트의 모든 props 최신 값으로 업데이트 
**부모 컴포넌트에서만 변경하고, 이를 내려 받는 자식 컴포넌트는 자연스럽게 갱신**

- One Way Data Flow : 모든 props는 자식 <> 부모 사이에 하향식 단방향 바인딩 형성 
→ 단방향인 이유? 하위 컴포넌트가 실수로 상위 컴포넌트의 상태 변경하여 앱에서의 데이터 흐름 이해하기 어렵게 만드는 것 방지하기 위함 ( 데이터 흐름의 일관성 & 단순화 )

- 초기설정 :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/23a891c9-4909-4489-acbf-0ecd39e1e970/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c6f7cac3-7f93-4a80-9548-d1dcd1a977ea/image.png)

### Props 활용

- 선언 : 부모 컴포넌트에서 내려 보낸 props 를 사용하기 위해서는 
자식 컴포넌트에서 명시적인 props 선언이 필요

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2f3b80e3-23d1-45e3-b84e-a2eeb5796b09/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/3b3be0c8-0907-4399-9c13-2beb30256589/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1e22dd22-3a91-4e18-9a24-ae27fc066691/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/4751ed96-c33d-47b9-b2a8-c321486e8d2f/image.png)

- 이러면 문자열 그대로 와버리는데, 우리는 prop 된 데이터 message를 받고싶음

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a6043a19-7a1f-4352-955f-24ed3e903e3e/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/179fcb65-7a37-4761-b136-c3a74a11e854/image.png)

- 동일한 구성에서 : 만 사용해 바인딩을 통한 변수명에 들어있는 데이터 가져옴

- defineProps는 호출결과로, props 객체를 반환  →  const로 변수 저장해 사용
- 변수명: dtype과 같은 느낌으로 들어가는데, 
required: true  /  default:10 등 조건 달아주기 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/1f836151-d9ae-4cea-b71f-67607d56e314/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/493f6f57-ee17-417f-9c35-947195cfc8af/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/01be0f4e-7d10-47bb-b0d9-d3285f70a0c9/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/6855f197-c3ed-451d-8b28-3f3f00f6a51c/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/68aa7d5e-688a-4153-a0c6-cf159a8818ef/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/fbda0e0d-cf6c-46e8-abe0-e1902972b247/image.png)

- parent의 변수인 name을 dynamic-prop으로 넘겨
- props 통해 온 dynamicProp라는 변수로 받아와서 출력

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/2c405899-5360-45a3-8fb0-ea997b64ebb9/image.png)

배열 items 만들고, ParentItem이라는 컴포넌트 3번 실행 ? 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a4bbb60b-fcfd-432f-b453-2eb9816697cc/image.png)

- ParentChild와 동일한 레벨의 item component 3개 생성

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/07d07547-313b-40c4-b2ee-780033563bc0/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a30160d4-6911-4fb5-a96d-b64266e349b1/image.png)

## Component Events

### Emit

- 부모가 자식에게 데이터를 전달(prop) 하는 것과 마찬가지로, 자식이 자신에게 일어난 일을 부모에게 알림(emit) 한다 → 부모가 props 데이터를 변경하도록 소리쳐야 한다
- 이벤트에 반응하기 위함

- $emit() : 
자식 컴포넌트가 이벤트 발생시켜 → 부모 컴포넌트로 데이터 전달하는 역할의 메서드 

구조 : $emit(event, …args) 
- event : 커스텀 이벤트 이름
- args : 추가 인자

### 이벤트 발신 및 수신 :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c8b4bab5-1d30-4d7c-8e59-23219857edea/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/c5057973-652b-4c61-bcc5-c05dc726f7ee/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5cc74b46-9e0d-4b18-bb84-11fa9d0ec7e2/image.png)

- 아래와 같은 형식으로도 emit 선언 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/26318563-1afd-46f7-86b2-693369e60228/image.png)

- 아래는 인자 넘기는 예시

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/53d5baf3-9c8b-4864-ab1f-97375b260355/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/615de0bc-cb03-4768-a9e2-e62c3906fc84/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/8dd2361c-fc74-4dc1-ab06-be7d65d8c0bc/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/020565d4-8798-497e-9d43-a0ea95326519/image.png)

- 이런 식으로 넘기면 배열과 같이 출력

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cb78110a-715b-43fe-9878-6678cfec8499/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/44b8b787-d46c-4292-b689-97e8b2fdfe19/image.png)

### emit 이벤트 활용

- 최하단 컴포넌트 grandchild에서 parent 컴포넌트의 name 변수 변경요청하기

```jsx
<!-- Parent.vue -->

<template>
  <div>
    <!-- 하위 컴포넌트에게 넘겨줄 데이터를 어떤 변수에 담을 것인지 지정 -->
    <ParentChild 
    my-msg="message" 
    :dynamic-prop="name"
    @some-event="someCallbak"
    @emit-args="getNumbers"
    @update-name="updateName"
    />
    
    <ParentItem 
      v-for="item in items"
      :key="item.id"
      :my-prop="item"
    />
  </div>
</template>

<script setup>
import ParentChild from '@/components/ParentChild.vue'
import ParentItem from './ParentItem.vue'
import {ref} from 'vue'

const name = ref('Alice')
const items = ref([
  {id: 1, name: '사과'},
  {id: 2, name: '바나나'},
  {id: 3, name: '딸기'},
])

const someCallbak = function (){
  console.log('ParentChild 컴포넌트가 발신한 이벤트를 수신하였음')
}

const getNumbers = function (...args) {
  console.log(`ParentChild가 넘겨준 인자들 ${args} 출력 `)
}

const updateName = function() {
  name.value = 'Bella'
}

</script>

<style scoped>
 
</style>
///////////////////////////////////////////////
<!-- ParentChild.vue -->

<template>
  <div>
    <p> {{  myMsg }} </p>
    <ParentGrandChild 
    :my-msg="myMsg"
    @update-name="updateName"
    />
    <p> {{  dynamicProp }} </p>
    <!--<button @click="$emit('someEvent')">클릭</button>-->
    <button @click="buttonClick">클릭</button>
    <button @click="emitArgs">추가 인자 전달</button>
  </div>
</template>

<script setup>
import ParentGrandChild from './ParentGrandChild.vue';
// props 선언 : 배열
// defineProps(['myMsg'])
// props 선언 : 객체 

defineProps({
  myMsg: String,
  dynamicProp: String
})

// emit 선언 & 실행 
const emit = defineEmits(['someEvent', 'emitArgs', 'updateName'])
const buttonClick = function() {
  emit('someEvent')
}

const emitArgs = function () {
  emit('emitArgs', 1, 2, 3)
}

const updateName = function() {
  emit('updateName')
}

</script>

<style scoped>

</style>

/////////////////////////////////////////////////////
<!-- ParentGrandChild.vue -->
<template>
  <div>
    <p> {{  myMsg }} </p>
    <button @click="updateName">이름 변경</button>
  </div>
</template>

<script setup>
  const emit = defineEmits(['updateName'])
  const props = defineProps({
                  myMsg:String
                })
                
  const updateName = function(){
    emit('updateName')
  }
</script>

<style scoped>

</style>
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/5cd1e2da-bdf6-4e0a-bca3-abe8076bdd37/image.png)

- 바인딩 이후 “”는 무조건 JS 표현식이라 숫자 1이 감!

## Computed Properties

### Computed

- 계산된 속성을 정의하는 함수 : 
미리 계산된 속성 사용해 템플릿에서 표현식 단순하게 하고 불필요한 반복 줄임

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/89381fd2-4461-4702-9344-408703e9d108/image.png)

- 기존 코드 : 뭔가 할때마다 todos 하나 빼는?

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/bfd32d90-1c13-4057-a4fd-ec1261119bf8/52d65774-f20a-439c-8c18-dfed1922654f.png)

- computed 적용 : 반환값을 변수에 할당

```jsx
<template>
  <div>
    <h2>남은 할 일</h2>
    <p>{{ restOfTodos }}</p>
    <p>{{ getRestOfTodos() }}</p>
  </div>
</template>

<script setup>
  import { ref, computed } from 'vue'

  const todos = ref([
    { text: 'Vue 실습' },
    { text: '자격증 공부' },
    { text: 'TIL 작성' }
  ])

  const restOfTodos = computed(() => {
    console.log('computed!!!')
    return todos.value.length > 0 ? '아직 남았다' : '퇴근!'
  })

  const getRestOfTodos = function () {
    console.log('method!!!')
    return todos.value.length > 0 ? '아직 남았다' : '퇴근!'
  }
</script>

<style scoped>

</style>
```

- 특징 : 
반환되는 값은 computed ref이며, 
일반 refs와 유사하게 계산된 결과 .value로 참조 가능 ( 템플릿에서는 .value 생략 가능 ) 

computed 속성은 의존된 반응형 데이터 자동으로 추적
 
**의존하는 데이터 변경될 때만 재평가** 
1. restOfTodos 의 계산은 todos에 의존하고 있음 
2. → todos 변경될 때만 restOfTodos가 업데이트 됨

- 장점 : 
의존하는 데이터 따라 결과가 바뀌는 게산된 속성 만들 때에 유용 
동일한 의존성 가진 여러 곳에서 사용할 때에 계산 결과 캐싱하여 중복 계산 방지

### Computed  VS  Methods

- computed 속성 : 의존된 반응형 데이터 기반으로 **캐시됨** 
의존하는 데이터가 변경된 경우에만 재평가됨 
즉, 의존된 반응형 데이터가 변경되지 않는 한 이미 계산된 결과에 대한 여러 참조는 다시 평가할 필요 없이, 이전에 계산된 결과 즉시 반환 

반면, method 호출은 **다시 랜더링 발생할 때마다 항상 함수 실행**

- 아래 보면 차이 확연히 보임. 
computed는 restOfTodos가 변화가 있어야 다시 실행되는데
변경이 없이 바로 불러졌으므로 출력되지 않음 ( 2번째 computed )

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/93240c4f-41c8-49a9-8e37-fab7b8096c8b/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b92e1b9c-a8ec-437a-bd1a-a53beb589a6f/image.png)

- 주의사항 : component의 반환값 / 원본 배열 은 변경하지 말 것

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/29157628-dbbc-4f28-94c4-19147260f957/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/b74977e1-a735-4e88-95f3-9b1868513695/image.png)

## Watchers

### Watch() :

- 하나 이상의 **반응형 데이터 감시** / 감시하는 데이터 **변경되면 콜백 함수 호출**

- 구조 :

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/a5cc2ff2-be6f-49c8-bbbc-fc6333497efc/image.png)

```jsx
<template>
  <div>
    <!-- 1 -->
    <button @click="count++">Add 1</button>
    <p>Count: {{ count }}</p>

    <!-- 2 -->
    <input v-model="message">
    <p>Message length: {{ messageLength }}</p>
  </div>
</template>

<script setup>
  import { ref, watch } from 'vue'
  
  const count = ref(0)

  watch(count, (newValue, oldValue) => {
    console.log(`newValue: ${newValue}, oldValue: ${oldValue}`)
  })
  
  const message = ref('')
  const messageLength = ref(0)

  watch(message, (newValue) => {
    messageLength.value = newValue.length
  })

</script>

<style scoped>

</style>
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/88326bb3-be6a-47b8-8eda-368249a1ead3/image.png)

- 예시 : 감시대상의 변동사항을 체크해 다른 객체에 업데이트

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/301f6ec2-f23d-4739-b0ef-506ce1261fc8/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/cd6dba31-c316-4f66-a201-78e0128a3c09/image.png)

- 여러 source 감시는 배열을 통해서 진행 가능

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/804bc384-ffc7-40ee-9bdc-6b9d8d05202a/image.png)

### Computed  VS  Watch

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/7ab80420-dc7e-4fc5-8f76-006851503a18/12194fee-2550-4853-b24d-f939cfec3f2a/image.png)

## LifeCycle Hooks :

- vue 인스턴스의 생애주기 동안 특정 시점에 실행되는 함수 

인스턴스의 생애 주기 중간에 함수 제공 
→ 개발자가 특정 단계에서 원하는 로직 작성할 수 있도록 함
1. vue 컴포넌트 인스턴스가 초기 렌더링 및 DOM요소 생성 완료된 후 특정 로직 수행하기 

```jsx
<template>
  <div>
    <button @click="count++">Add 1</button>
    <p>Count: {{ count }}</p>
    <p>{{ message }}</p>
  </div>
</template>

<script setup>
  import { ref, onMounted, onUpdated } from 'vue'
  
  onMounted(() => {
    console.log('mounted!!')
    // console.log(count.value)
    // console.log(message.value)
    updateMessage()
  })
  
  const updateMessage = function() {
	  message.value = 'mounted!!!'
	  alert()
  }

  const count = ref(0)
  const message = ref(null)

  onUpdated(() => { // 변동사항 발생하면 실행됨 
    console.log('updated!!')
    // message.value = 'updated!'
  })

</script>

<style scoped>

</style>
```

---