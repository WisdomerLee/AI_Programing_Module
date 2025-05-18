**RAG(Retrieval Augmented Generation, 검색 증강 생성)** 파이프라인의 핵심
LLM이 외부 데이터를 효과적으로 활용하여 답변의 정확성과 최신성을 높이는 데 중요한 역할을 합니다.

-----

# 1\. 인덱스(Index)와 리트리버(Retriever)의 역할

LLM은 학습 데이터에 없는 최신 정보나 특정 도메인의 전문 지식에 대해서는 답변하기 어렵고, 때로는 잘못된 정보를 생성(환각 현상)하기도 합니다.
RAG는 이러한 한계를 극복하기 위해 외부 데이터 소스에서 관련 정보를 검색하여 LLM에게 컨텍스트로 제공하는 방식입니다.

  * **인덱싱(Indexing)**: 외부 데이터(텍스트 파일, PDF, 웹 페이지 등)를 LLM이 쉽게 검색하고 활용할 수 있도록 **구조화하고 저장하는 과정**입니다. 이 과정에는 문서 로드, 텍스트 분할, 임베딩 생성, 벡터 스토어 저장이 포함됩니다.
  * **리트리버(Retriever)**: 사용자의 질문(쿼리)이 주어졌을 때, 인덱싱된 데이터 중에서 **관련성이 높은 정보를 검색하여 가져오는 역할**을 합니다.

이 두 가지를 통해 LLM은 마치 "오픈 북 시험"을 치르듯, 주어진 참고 자료(검색된 문서)를 바탕으로 답변을 생성할 수 있게 됩니다.

-----

# 2\. 인덱싱(Indexing) 과정 상세 설명 및 코드 예시

인덱싱은 보통 다음 단계로 진행됩니다.

## 2.1. Document Loaders (문서 로더)

다양한 소스로부터 텍스트 데이터를 로드하여 Langchain의 표준 `Document` 객체 형식으로 변환합니다. `Document` 객체는 `page_content` (텍스트 내용)와 `metadata` (출처 등의 부가 정보)를 가집니다.

**주요 로더:**

  * `TextLoader`: `.txt` 파일 로드
  * `PyPDFLoader`: PDF 파일 로드 (`pypdf` 라이브러리 필요)
  * `WebBaseLoader`: 웹 페이지 내용 로드 (`beautifulsoup4` 라이브러리 필요)
  * `CSVLoader`: CSV 파일 로드
  * `DirectoryLoader`: 특정 디렉토리 내의 여러 파일 로드 (다른 로더와 조합 가능)

**코드 예시:**

```python
import os
from langchain_community.document_loaders import TextLoader, WebBaseLoader # langchain-community 설치 필요

# API 키 설정 (OpenAI 임베딩 사용 시 필요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 예시 1: 텍스트 파일 로드
# 'example.txt' 파일 생성 (내용: "이것은 텍스트 파일 예시입니다. Langchain은 강력합니다.")
try:
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("이것은 텍스트 파일 예시입니다. Langchain은 강력합니다.\nLangchain을 사용하면 LLM 애플리케이션을 쉽게 만들 수 있습니다.")

    loader_txt = TextLoader("./example.txt", encoding="utf-8")
    documents_txt = loader_txt.load()
    print(f"--- TextLoader 결과 ---")
    for doc in documents_txt:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
except Exception as e:
    print(f"TextLoader 예외: {e}")


# 예시 2: 웹 페이지 로드 (실행 시 네트워크 연결 필요)
try:
    # beautifulsoup4 설치 필요: pip install beautifulsoup4
    loader_web = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/") # 예시 URL
    documents_web = loader_web.load() # 페이지의 주요 텍스트를 가져옴
    print(f"\n--- WebBaseLoader 결과 (첫 200자) ---")
    if documents_web:
        print(f"Content (첫 200자): {documents_web[0].page_content[:200]}")
        print(f"Metadata: {documents_web[0].metadata}")
    else:
        print("웹 페이지 로드에 실패했거나 내용이 없습니다.")
except ImportError:
    print("\nWebBaseLoader를 사용하려면 'pip install beautifulsoup4'를 실행해주세요.")
except Exception as e:
    print(f"\nWebBaseLoader 예외: {e}")

# 임시 파일 삭제
if os.path.exists("example.txt"):
    os.remove("example.txt")
```

## 2.2. Text Splitters (텍스트 분할기)

로드된 문서를 LLM이 한 번에 처리하기 좋고, 임베딩 모델의 토큰 제한을 넘지 않으며, 검색 효율성을 높이기 위해 작은 의미 단위(청크)로 분할합니다.

**주요 분할기:**

  * `CharacterTextSplitter`: 지정된 문자 수와 구분자(기본값: `\n\n`)를 기준으로 분할.
  * `RecursiveCharacterTextSplitter` (권장): 다양한 구분자 리스트(`["\n\n", "\n", " ", ""]` 등)를 사용하여 의미를 최대한 유지하며 재귀적으로 분할.
  * `TokenTextSplitter`: 토큰 수를 기준으로 분할 (모델의 토큰 계산 방식과 일치시키기 좋음).

**주요 파라미터:**

  * `chunk_size`: 각 청크의 최대 크기 (문자 수 또는 토큰 수).
  * `chunk_overlap`: 인접한 청크 간에 겹치는 부분의 크기. 문맥 유지를 위해 중요.

**코드 예시:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 예시 텍스트 (실제로는 Document 객체의 page_content를 사용)
long_text = """Langchain은 대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 위한 오픈소스 프레임워크입니다.
이 프레임워크는 개발자들이 LLM을 외부 데이터 소스 및 다른 계산 도구와 쉽게 통합할 수 있도록 설계되었습니다.
주요 구성 요소로는 Models, Prompts, Chains, Indexes, Agents, Memory 등이 있습니다.
각 요소는 모듈식으로 설계되어 필요에 따라 조합하여 사용할 수 있습니다.
Langchain을 사용하면 챗봇, 질의응답 시스템, 요약 등 다양한 LLM 기반 애플리케이션을 효율적으로 구축할 수 있습니다."""

# RecursiveCharacterTextSplitter 사용
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 청크 크기 (문자 수)
    chunk_overlap=20, # 겹치는 부분 크기
    length_function=len, # 길이 계산 함수
    is_separator_regex=False, # 구분자가 정규 표현식인지 여부
)

# Document 객체 분할 (Document 객체 리스트를 입력으로 받음)
# 여기서는 간단히 문자열을 Document 객체로 변환하여 사용
from langchain_core.documents import Document
docs_to_split = [Document(page_content=long_text)]
split_texts = text_splitter.split_documents(docs_to_split)

print(f"\n--- TextSplitter 결과 ---")
print(f"원본 텍스트 길이: {len(long_text)}")
print(f"분할된 청크 수: {len(split_texts)}")
for i, chunk_doc in enumerate(split_texts):
    print(f"청크 {i+1}: \"{chunk_doc.page_content}\" (길이: {len(chunk_doc.page_content)})")
```

## 2.3. Text Embedding Models (텍스트 임베딩 모델)

분할된 텍스트 청크를 고차원의 숫자 벡터(임베딩)로 변환합니다. 이 벡터는 텍스트의 의미를 담고 있어, 벡터 간의 유사도를 통해 의미적 유사성을 계산할 수 있습니다. (이전 `Model` 설명에서 자세히 다룸)

**코드 예시 (간단히 복습):**

```python
from langchain_openai import OpenAIEmbeddings # langchain-openai 설치 필요

# OpenAI API 키가 환경 변수에 설정되어 있어야 함
try:
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    example_embedding = embeddings_model.embed_query("이것은 임베딩 테스트 문장입니다.")
    print(f"\n--- Text Embedding 결과 (첫 5개 차원) ---")
    print(example_embedding[:5])
    print(f"임베딩 벡터 차원: {len(example_embedding)}")
except Exception as e:
    print(f"\nOpenAIEmbeddings 예외: {e}. API 키 설정을 확인하세요.")
```

## 2.4. Vector Stores (벡터 스토어)

텍스트 청크와 해당 임베딩 벡터를 함께 저장하고, 유사도 검색을 효율적으로 수행할 수 있는 데이터베이스입니다.

**주요 벡터 스토어:**

  * **인메모리/로컬 파일 기반:**
      * `FAISS`: Facebook AI Research에서 개발한 빠르고 효율적인 유사도 검색 라이브러리. (`faiss-cpu` 또는 `faiss-gpu` 설치 필요)
      * `Chroma`: 오픈 소스 임베딩 데이터베이스. (`chromadb` 설치 필요)
  * **클라우드/서버 기반:** `Pinecone`, `Weaviate`, `Redis`, `Milvus` 등

**코드 예시 (`FAISS` 사용):**

```python
from langchain_community.vectorstores import FAISS # langchain-community, faiss-cpu/faiss-gpu 설치 필요
# split_texts와 embeddings_model이 이전 단계에서 정의되었다고 가정

# 실제 사용 시에는 documents_txt 또는 documents_web을 분할한 split_texts를 사용합니다.
# 여기서는 위에서 정의한 split_texts (long_text를 분할한 것)를 사용하겠습니다.

if 'split_texts' in globals() and 'embeddings_model' in globals():
    try:
        print(f"\n--- Vector Store (FAISS) 생성 중 ---")
        # FAISS.from_documents()는 내부적으로 embed_documents를 호출하여 임베딩을 생성하고 저장합니다.
        vectorstore_faiss = FAISS.from_documents(documents=split_texts, embedding=embeddings_model)
        print("FAISS 벡터 스토어 생성 완료!")

        # (선택 사항) 로컬에 저장하고 불러오기
        # vectorstore_faiss.save_local("my_faiss_index")
        # loaded_vectorstore = FAISS.load_local("my_faiss_index", embeddings_model, allow_dangerous_deserialization=True) # 최신 버전에서는 allow_dangerous_deserialization 필요할 수 있음
        # print("FAISS 인덱스 로드 완료!")

    except ImportError:
        print("FAISS를 사용하려면 'pip install faiss-cpu' 또는 'pip install faiss-gpu'를 실행해주세요.")
    except Exception as e:
        print(f"FAISS 벡터 스토어 생성 중 예외: {e}")
else:
    print("\nFAISS 예제를 실행하기 위한 이전 단계의 변수(split_texts, embeddings_model)가 정의되지 않았습니다.")

```

**코드 예시 (`Chroma` 사용):**

```python
from langchain_community.vectorstores import Chroma # langchain-community, chromadb 설치 필요

if 'split_texts' in globals() and 'embeddings_model' in globals():
    try:
        print(f"\n--- Vector Store (Chroma) 생성 중 ---")
        # Chroma.from_documents()는 임시적인 인메모리 Chroma를 사용합니다.
        # 영구 저장을 위해서는 persist_directory를 지정합니다.
        vectorstore_chroma = Chroma.from_documents(documents=split_texts, embedding=embeddings_model, persist_directory="./chroma_db")
        print("Chroma 벡터 스토어 생성 및 저장 완료! (./chroma_db)")

        # 저장된 Chroma DB 로드
        # loaded_vectorstore_chroma = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)
        # print("Chroma 인덱스 로드 완료!")

    except ImportError:
        print("Chroma를 사용하려면 'pip install chromadb'를 실행해주세요.")
    except Exception as e:
        print(f"Chroma 벡터 스토어 생성 중 예외: {e}")
else:
    print("\nChroma 예제를 실행하기 위한 이전 단계의 변수(split_texts, embeddings_model)가 정의되지 않았습니다.")
```

-----

# 3\. 리트리버(Retriever) 상세 설명 및 코드 예시

리트리버는 인덱싱된 벡터 스토어에서 사용자의 쿼리와 관련된 문서를 검색하는 인터페이스입니다.

**생성 방법:**

  * 일반적으로 벡터 스토어 객체의 `as_retriever()` 메서드를 통해 생성합니다.

**주요 파라미터 (`as_retriever()` 호출 시):**

  * `search_type`: 검색 전략을 지정합니다.
      * `"similarity"` (기본값): 쿼리와 가장 유사한 K개의 문서를 반환합니다.
      * `"mmr"` (Maximal Marginal Relevance): 유사도와 다양성을 모두 고려하여 K개의 문서를 반환합니다. 검색 결과가 너무 유사한 내용으로만 채워지는 것을 방지합니다.
      * `"similarity_score_threshold"`: 유사도 점수가 특정 임계값 이상인 문서만 반환합니다.
  * `search_kwargs`: `search_type`에 따른 추가 인자를 딕셔너리 형태로 전달합니다.
      * `k`: 반환할 문서의 개수 (기본값: 4).
      * `score_threshold` (`similarity_score_threshold` 타입 사용 시): 유사도 점수 임계값.
      * `Workspace_k` (MMR 사용 시): 초기 유사도 검색으로 가져올 문서 수 (MMR 계산 대상).
      * `lambda_mult` (MMR 사용 시): 다양성 조절 파라미터 (0에 가까울수록 다양성 중시, 1에 가까울수록 유사도 중시).

**주요 메서드:**

  * `invoke(query_string)` 또는 `get_relevant_documents(query_string)`: 쿼리 문자열을 입력받아 관련 문서 리스트를 반환합니다.

**코드 예시 (FAISS 벡터 스토어가 생성되었다고 가정):**

```python
if 'vectorstore_faiss' in globals():
    print(f"\n--- Retriever (FAISS 기반) 테스트 ---")
    query = "Langchain의 주요 구성 요소는 무엇인가요?"

    # 1. 기본 유사도 검색 리트리버
    retriever_similarity = vectorstore_faiss.as_retriever(search_kwargs={"k": 2}) # 상위 2개 문서 검색
    relevant_docs_similarity = retriever_similarity.invoke(query)
    print(f"\n[유사도 검색 결과 (k=2)] - 쿼리: {query}")
    for doc in relevant_docs_similarity:
        print(f"  - {doc.page_content} (Score: {doc.metadata.get('_score', 'N/A')})") # FAISS는 기본적으로 스코어를 메타데이터에 추가하지 않을 수 있음

    # 2. MMR 검색 리트리버
    # MMR은 유사하면서도 다양한 결과를 보여주려고 함
    retriever_mmr = vectorstore_faiss.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 2, 'fetch_k': 5, 'lambda_mult': 0.75}
    )
    relevant_docs_mmr = retriever_mmr.invoke(query)
    print(f"\n[MMR 검색 결과 (k=2, fetch_k=5, lambda=0.75)] - 쿼리: {query}")
    for doc in relevant_docs_mmr:
        print(f"  - {doc.page_content}")

    # 3. 유사도 점수 임계값 기반 검색 (FAISS는 score_threshold 직접 지원이 다를 수 있음, Chroma에서 더 명확)
    # FAISS의 경우, as_retriever에서 score_threshold를 직접 지원하지 않고,
    # similarity_search_with_score를 호출 후 수동 필터링하거나,
    # 다른 벡터 스토어(예: Chroma)를 사용할 때 더 명확하게 적용됩니다.
    # 여기서는 개념만 설명합니다.
    # retriever_threshold = vectorstore_faiss.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.7})

    # Chroma를 사용한 경우의 예시 (위에서 vectorstore_chroma가 생성되었다면)
    if 'vectorstore_chroma' in globals():
        retriever_chroma_threshold = vectorstore_chroma.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 3, 'score_threshold': 0.5} # Chroma는 score를 반환하며, 이 값이 높을수록 유사 (0~1 사이)
        )
        relevant_docs_chroma_threshold = retriever_chroma_threshold.invoke(query)
        print(f"\n[Chroma 유사도 점수 임계값 검색 결과 (k=3, threshold=0.5)] - 쿼리: {query}")
        for doc in relevant_docs_chroma_threshold:
            # Chroma는 score를 Document의 metadata에 _score로 저장하거나, search_with_score 메서드를 통해 직접 튜플로 반환
            # as_retriever().invoke()는 Document 리스트만 반환하므로, 점수 확인을 위해서는 내부 구현을 보거나
            # vectorstore.similarity_search_with_score()를 직접 사용해야 할 수 있습니다.
            # Langchain 버전 및 VectorStore 구현에 따라 다를 수 있습니다.
            print(f"  - {doc.page_content}")


else:
    print("\nRetriever 예제를 실행하기 위한 이전 단계의 변수(vectorstore_faiss 또는 vectorstore_chroma)가 정의되지 않았습니다.")

```

## 고급 리트리버 (개념 소개)

  * **`MultiQueryRetriever`**: 사용자 쿼리를 LLM을 사용하여 여러 관점의 유사한 쿼리로 변형하고, 각 변형된 쿼리에 대한 검색 결과를 통합하여 반환합니다. 더 넓은 범위의 관련 문서를 찾을 수 있습니다.
    ```python
    # from langchain.retrievers.multi_query import MultiQueryRetriever
    # from langchain_openai import ChatOpenAI # LLM 필요

    # # multi_query_retriever = MultiQueryRetriever.from_llm(
    # #     retriever=vectorstore_faiss.as_retriever(), llm=ChatOpenAI(temperature=0)
    # # )
    # # docs_multi_query = multi_query_retriever.invoke("Langchain 구성 요소에 대해 알려줘")
    ```
  * **`ContextualCompressionRetriever`**: 기본 리트리버가 문서를 검색한 후, LLM을 사용하여 해당 문서에서 실제로 쿼리와 관련된 핵심 내용만 추출하거나 요약하여 반환합니다. LLM에 전달되는 컨텍스트의 노이즈를 줄일 수 있습니다.
    ```python
    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain.retrievers.document_compressors import LLMChainExtractor
    # from langchain_openai import OpenAI

    # # compressor = LLMChainExtractor.from_llm(OpenAI(temperature=0))
    # # compression_retriever = ContextualCompressionRetriever(
    # #     base_compressor=compressor, base_retriever=vectorstore_faiss.as_retriever()
    # # )
    # # compressed_docs = compression_retriever.invoke("Langchain의 Indexes는 무엇인가?")
    ```
  * **`SelfQueryRetriever`**: 사용자 쿼리에서 메타데이터 필터 조건을 자동으로 추출하고, 이를 벡터 검색과 결합하여 더욱 정교한 검색을 수행합니다. (예: "2023년에 작성된 Langchain 관련 문서 찾아줘")

-----

# 4\. 인덱스/리트리버와 체인(Chain)의 연동

리트리버가 검색한 문서는 `RetrievalQA`나 `ConversationalRetrievalChain`과 같은 체인에 전달되어, LLM이 이 정보를 바탕으로 최종 답변을 생성합니다.

**코드 예시 (`RetrievalQA` 복습):**

```python
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

if 'vectorstore_faiss' in globals():
    try:
        # RetrievalQA 체인 생성
        qa_chain_rag = RetrievalQA.from_chain_type(
            llm=OpenAI(model_name="gpt-3.5-turbo-instruct"), # 답변 생성 LLM
            chain_type="stuff",                             # 검색된 문서를 처리하는 방식
            retriever=vectorstore_faiss.as_retriever(search_kwargs={"k": 3}), # 리트리버 사용
            return_source_documents=True                    # 출처 문서도 함께 반환
        )

        query_rag = "Langchain의 인덱싱 과정은 어떻게 되나요?"
        result_rag = qa_chain_rag.invoke({"query": query_rag})

        print(f"\n--- RetrievalQA (RAG) 결과 ---")
        print(f"질문: {query_rag}")
        print(f"답변: {result_rag['result']}")
        print(f"참고한 문서:")
        for doc in result_rag['source_documents']:
            print(f"  - {doc.page_content[:100]}...") # 내용이 길 수 있으므로 일부만 출력
    except Exception as e:
        print(f"\nRetrievalQA 예외 발생: {e}")
else:
    print("\nRetrievalQA 예제를 실행하기 위한 vectorstore_faiss가 정의되지 않았습니다.")

# 임시 Chroma DB 디렉토리 정리 (실제 운영 환경에서는 필요에 따라 관리)
import shutil
if os.path.exists("./chroma_db"):
    try:
        shutil.rmtree("./chroma_db")
        print("\n임시 Chroma DB 디렉토리 (./chroma_db) 삭제 완료.")
    except Exception as e:
        print(f"\n임시 Chroma DB 디렉토리 삭제 중 오류: {e}")
```

-----
