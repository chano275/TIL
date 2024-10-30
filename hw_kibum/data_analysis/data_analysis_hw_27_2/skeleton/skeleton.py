from langchain_community.document_loaders import TextLoader

# 1. Langchain의 TextLoader를 사용하여 파일 로드
# TODO: TextLoader를 사용하여 금융 기사 파일을 로드하세요.
# "../data/financial_articles.txt" 경로의 파일을 불러와 loader 변수에 저장합니다.
# TextLoader() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
loader = TextLoader('../data/financial_articles.txt')

# 2. 파일 내용을 로드
# TODO: loader.load() 메서드를 사용하여 문서 내용을 로드하세요.
# 로드된 문서를 documents 변수에 저장합니다.
# loader.load() 참고: https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/
documents = loader.load()

# 3. 각 문서의 텍스트 길이와 토큰 수를 출력
for i, doc in enumerate(documents):
    print(f"문서 {i+1}의 길이: {len(doc.page_content)} 문자")
    print(f"문서 {i+1}의 토큰 수: {len(doc.page_content.split())} 토큰")

# 4. 첫 번째 문서의 내용을 출력
print("\n첫 번째 문서의 내용:")
print(documents[0].page_content)
