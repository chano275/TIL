import pandas as pd

# 데이터 로드
file_path = ___________
sales_data = ___________

# 월별 매출 총합 계산
monthly_revenue = ______________

# 가장 높은 매출을 기록한 달 찾기
max_month = monthly_revenue._________

# 해당 달의 데이터 필터링
max_month_data = sales_data[sales_data['Month'] == max_month]

# 해당 달에 가장 많이 팔린 제품 찾기
top_product = max_month_data.groupby('Product')[______].________________
top_quantity = max_month_data.groupby('Product')[_______].______________

# 결과 출력
print(f"{max_month}에 가장 많이 팔린 제품: {top_product} (판매량: {top_quantity} 개)")
