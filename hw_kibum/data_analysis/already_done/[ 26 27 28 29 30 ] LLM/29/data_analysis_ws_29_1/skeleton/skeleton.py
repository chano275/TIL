# pip install datasets==2.20.0 pymupdf

from langchain_community.document_loaders import PyMuPDFLoader

# PDF 파일에서 텍스트 조각을 추출하는 함수입니다.
# 이 함수는 문서를 정교하게 분할하기보다는, 데이터를 생성하기 위해 간단히 청킹작업을 수행하기 위한 목적으로 사용합니다.
def extract_pdf_elements(filepath):
    """
    LangChain의 PyMuPDFLoader를 사용하여 PDF 파일에서 텍스트 조각을 추출합니다.
    """
    # langchain_community.document_loaders 모듈의 PyMuPDFLoader 클래스는 PyMuPDF를 사용하여 PDF 파일의 페이지를 로드하고, 각 페이지를 개별 document 객체로 추출합니다. 
    # 특히 PDF 문서의 자세한 메타데이터를 추출하는 데 강점이 있습니다.
    # TODO: PyMuPDFLoader 객체를 통해 document 추출을 완성하세요.
    loader = ______________
    documents = ______________  # PDF를 LangChain Document 형식으로 로드합니다.
    return documents


# PDF 파일 로드 및 텍스트 추출
elements = extract_pdf_elements("../data/canada_av-68-90.pdf")

# 로드한 텍스트 청크 수 확인
print(f"로드된 텍스트 청크 수: {len(elements)}")

# 추출된 텍스트의 첫 번째 청크 확인
print(elements[0])