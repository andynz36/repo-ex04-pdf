import os
import streamlit as st
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Streamlit 앱에서 문서를 매번 로드하지 않도록 캐싱 사용
@st.cache_resource
def init_rag_chain():
    # 1. 문서 로드
    loader = PyPDFLoader("unsu.pdf")
    pages = loader.load_and_split()

    # 2. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)

    # 3. 임베딩 모델 및 벡터 저장소 설정
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(texts, embeddings_model)

    # 4. LLM 설정
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5. 검색기(Retriever) 설정
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(), 
        llm=llm
    )

    # 6. 프롬프트 및 체인 생성
    system_prompt = (
        "너는 질문-답변을 돕는 유능한 비서야. "
        "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
        "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)
    
    return rag_chain

# Streamlit UI
st.set_page_config(page_title="PDF Q&A", page_icon="📄")
st.title("📄 PDF 질문-답변 봇")
st.markdown("`unsu.pdf` 파일에 대한 질문을 남겨주시면 AI가 답변해드립니다.")

# RAG 체인 초기화 (캐싱됨)
with st.spinner("문서와 AI 모델을 불러오는 중입니다..."):
    rag_chain = init_rag_chain()

# 사용자 입력
question = st.text_input("질문을 입력하세요:", placeholder="예: 아내가 사달라고 했던 음식 알려주고, 그외 언급된 음식들도 알려줘")

# 버튼 클릭 시 동작
if st.button("답변 받기", type="primary"):
    if question.strip():
        with st.spinner("문서를 검색하고 답변을 생성하는 중입니다..."):
            response = rag_chain.invoke({"input": question})
            
            st.success("답변이 생성되었습니다!")
            st.write(response['answer'])
            
            # 참조된 문서 내용 확인 (아코디언 형태)
            contexts = response.get('context', [])
            with st.expander(f"검색된 참조 문서 확인하기 (총 {len(contexts)}개)"):
                for idx, doc in enumerate(contexts):
                    st.markdown(f"**참조 {idx+1}** (페이지: {doc.metadata.get('page', 'N/A')})")
                    st.info(doc.page_content)
    else:
        st.warning("질문을 입력해주세요.")
