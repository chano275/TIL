import pandas as pd

# 3. 새로운 피처 생성
filtered_data = ______________

# 새로운 피처 생성: Price per Advertising (가격 대비 광고비)와 DiscountedPrice (할인된 가격)
filtered_data['PricePerAd'] = _______________
filtered_data['DiscountedPrice'] = _______________

# 새로운 피처를 추가한 데이터 저장
filtered_data.to_csv(_____________, index=False)
print(filtered_data.head())
