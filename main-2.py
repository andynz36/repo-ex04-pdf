# pip install -U langchain-text-splitters pypdf
# pip install -U langchain-community pypdf

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF 파일 로더 인스턴스 생성
loader = PyPDFLoader("unsu.pdf")
# PDF 파일에서 페이지를 로드하고 분할하여 페이지 객체 리스트로 반환
pages = loader.load_and_split()

# Split 단계 (텍스트 청크 쪼개기)
# LLM이 처리하기 좋게 문서를 더 작은 단위(chunk)로 잘게 쪼갠다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, # 하나의 텍스트 조각에 들어갈 최대 글자 수
  
    chunk_overlap = 20, #·앞뒤· 텍스트 · 조각 · 간에 ·겹칠 · 글자 ·수, · 문맥이 · 끊기는 · 것을 ·방지하기 · 위해 · 보통 · 10~20% · 정도 · 겹치게 · 설정함

    length_function = len,

    is_separator_regex = False, # 구분자를 정규표현식으로 해석할지 여부 판단
)

# 설정한 chunk_size(300자) 기준에 맞춰 최종 텍스트 조각들로 쪼갠다.
texts = text_splitter.split_documents(pages)


if texts:
    print("--- [첫 번째 텍스트 조각(Chunk) 객체 출력] ---")
    print(texts[0])
    
    print("\n--- [첫 번째 조각의 실제 텍스트 내용만 출력] ---")
    print(texts[0].page_content)
else:
    print("분할된 텍스트 조각이 없습니다. PDF 파일 내용을 확인해 주세요.")