from django.shortcuts import render

# Create your views here.
"""
책 리스트 조회 기능을 구현하시오.
책 리스트 조회는 인증 정보가 필요 없다.



책 대여 기능은 IsAuthenticated 권한을 설정하여 
인증된 사용자만 접근 가능하도록 설정하시오.
책 대여는 POST 요청만 처리할 수 있다. 
요청 데이터는 대여할 도서의 ISBN 값을 포함한다.
"""