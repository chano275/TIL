import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = ____________
font_name = ___________
rc(__________)

# 데이터 로드
file_path = ___________
movies = ___________

# 'Steven Spielberg' 감독의 영화 필터링
spielberg_movies = ____________

# 평점 분포 시각화
sns.histplot(__________)
plt.title('Steven Spielberg 감독 영화 평점 분포')
plt.xlabel('평점')
plt.ylabel('영화 수')
plt.show()
