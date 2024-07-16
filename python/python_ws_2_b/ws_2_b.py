# 아래에 코드를 작성하시오.
class Post():
    post_count = 0

    def __init__(self, title, content):
        self.title = title
        self.content = content
        Post.post_count += 1

    def show_content(self):
        print(self.content)
    
    def total_posts(self): # 게시물 총 개수 출력
        print(f'Total posts: {Post.post_count}')

    @staticmethod
    def description():
        print('SNS 게시물은 소셜 네트워크 서비스에서 사용자가 작성하는 글을 의미합니다.')

a = Post('제주도', '제주도 여행 왔어요~')
b = Post('일본', '일본 여행 왔어요~')


print(f'Title: {a.title}')
print(f'Content: {a.content}')
print(f'Title: {b.title}')
print(f'Content: {b.content}')
a.total_posts()
Post.description()