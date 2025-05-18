# **대규모 언어 모델(LLM)을 활용한 애플리케이션 개발을 간소화하고 강력하게 만들어주는 프레임워크**

복잡한 LLM 파이프라인을 손쉽게 구축
다양한 외부 데이터 소스 및 도구와 연동

# Langchain 핵심 키워드 및 개념

Langchain의 모듈식 구성 요소

## 1\. Models (모델)

LLM(Large Language Model)
  * **LLMs**: 텍스트 문자열을 입력받아 텍스트 문자열을 반환하는 기본 모델입니다.
    ```python
    from langchain_openai import OpenAI

    # OPENAI_API_KEY 환경 변수 설정 필요
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    prompt = "대한민국의 수도는 어디인가요?"
    response = llm.invoke(prompt)
    print(response)
    # 출력 예시: 대한민국의 수도는 서울입니다.
    ```
  * **Chat Models (챗 모델)**: 채팅 메시지 목록을 입력받아 채팅 메시지를 반환하는 모델로, 대화형 인터페이스에 적합합니다.
    ```python
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage

    # OPENAI_API_KEY 환경 변수 설정 필요
    chat = ChatOpenAI(model_name="gpt-4")
    messages = [
        SystemMessage(content="You are a helpful assistant that translates English to Korean."),
        HumanMessage(content="I love programming."),
    ]
    response = chat.invoke(messages)
    print(response.content)
    # 출력 예시: 저는 프로그래밍을 사랑합니다.
    ```
  * **Text Embedding Models (텍스트 임베딩 모델)**: 텍스트를 숫자 벡터로 변환하여 의미적 유사성을 비교하는 데 사용됩니다.
    ```python
    from langchain_openai import OpenAIEmbeddings

    # OPENAI_API_KEY 환경 변수 설정 필요
    embeddings_model = OpenAIEmbeddings()
    text = "안녕하세요, Langchain입니다."
    embedded_text = embeddings_model.embed_query(text)
    print(embedded_text[:5]) # 벡터의 처음 5개 차원만 출력
    ```

-----

## 2\. Prompts (프롬프트)

LLM에 전달할 입력을 동적으로 생성하고 관리하는 도구

  * **Prompt Templates (프롬프트 템플릿)**: 사용자 입력, 다른 동적 정보 등을 포함하는 프롬프트 문자열을 생성합니다.
    ```python
    from langchain_core.prompts import PromptTemplate

    prompt_template = PromptTemplate.from_template(
        "{country}의 수도는 어디인가요?"
    )
    filled_prompt = prompt_template.format(country="프랑스")
    print(filled_prompt)
    # 출력: 프랑스의 수도는 어디인가요?

    # LLM과 함께 사용
    # llm = OpenAI(model_name="gpt-3.5-turbo-instruct") # 위에서 정의한 llm 사용 가능
    # response = llm.invoke(filled_prompt)
    # print(response)
    ```
  * **Chat Prompt Templates (챗 프롬프트 템플릿)**: 챗 모델을 위한 메시지 목록을 생성합니다.
    ```python
    from langchain_core.prompts import ChatPromptTemplate

    chat_template = ChatPromptTemplate.from_messages([
        ("system", "Translate the following from {input_language} to {output_language}."),
        ("human", "{text}")
    ])
    messages = chat_template.format_messages(
        input_language="English",
        output_language="Spanish",
        text="Hello, how are you?"
    )
    print(messages)
    # chat = ChatOpenAI(model_name="gpt-4") # 위에서 정의한 chat 사용 가능
    # response = chat.invoke(messages)
    # print(response.content)
    ```

-----

## 3\. Chains (체인)

가장 기본적인 구성 요소로, 여러 LLM 호출 또는 다른 유틸리티를 순차적으로 연결하여 실행합니다.

  * **LLMChain**: 프롬프트 템플릿과 LLM을 결합하여 간단한 체인을 만듭니다. 가장 흔하게 사용되는 체인입니다.
    ```python
    from langchain_openai import OpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # OPENAI_API_KEY 환경 변수 설정 필요
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    prompt = PromptTemplate.from_template(
        "{product}을 만드는 새로운 회사 이름을 3개 추천해주세요."
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.invoke({"product": "친환경 세제"})
    print(response)
    # 출력 예시: {'product': '친환경 세제', 'text': '\n\n1. 에코퓨어 (EcoPure)\n2. 그린솝 (GreenSoap)\n3. 네이처클린 (NatureClean)'}
    ```
  * **Sequential Chains (순차 체인)**: 여러 체인을 순서대로 실행하며, 이전 체인의 출력을 다음 체인의 입력으로 전달할 수 있습니다.

-----

## 4\. Indexes (인덱스) & Retrievers (리트리버)

LLM이 외부 데이터를 효과적으로 사용할 수 있도록 텍스트 데이터를 구조화하고 검색하는 기능입니다. \*\*RAG (Retrieval Augmented Generation)\*\*의 핵심 요소입니다.

  * **Document Loaders (문서 로더)**: 텍스트 파일, PDF, 웹 페이지 등 다양한 소스에서 문서를 로드합니다.
    ```python
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader("./my_document.txt") # 예시 파일 경로
    # documents = loader.load()
    # print(documents[0].page_content[:100]) # 문서 내용의 처음 100자 출력
    ```
    (실행하려면 `my_document.txt` 파일이 필요합니다.)
  * **Text Splitters (텍스트 분할기)**: 긴 문서를 LLM이 처리하기 좋은 작은 청크로 분할합니다.
    ```python
    from langchain_text_splitters import CharacterTextSplitter

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents) # 위에서 로드한 documents 사용
    # print(len(texts))
    ```
  * **Vector Stores (벡터 스토어)**: 텍스트 청크의 임베딩(숫자 벡터 표현)을 저장하고 검색합니다. (예: FAISS, ChromaDB, Pinecone)
    ```python
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    # embeddings = OpenAIEmbeddings() # 위에서 정의한 embeddings_model 사용 가능
    # # texts 객체가 정의되어 있다고 가정 (TextSplitter를 통해 생성)
    # # vectorstore = FAISS.from_documents(texts, embeddings)
    ```
  * **Retrievers (리트리버)**: 사용자의 질문과 관련된 문서를 벡터 스토어에서 검색합니다.
    ```python
    # retriever = vectorstore.as_retriever()
    # query = "Langchain의 주요 기능은 무엇인가요?"
    # relevant_docs = retriever.invoke(query)
    # print(relevant_docs[0].page_content)
    ```

RAG를 활용한 체인 (예: `RetrievalQA`):

```python
from langchain.chains import RetrievalQA
# llm, retriever가 정의되어 있다고 가정

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff", # 다른 체인 타입: "map_reduce", "refine", "map_rerank"
#     retriever=retriever
# )
# query = "내 문서에서 Langchain에 대해 설명해줘."
# result = qa_chain.invoke({"query": query})
# print(result["result"])
```

-----

## 5\. Memory (메모리)

체인이나 에이전트가 이전 대화 내용을 "기억"하여 컨텍스트를 유지할 수 있도록 합니다.

  * **ConversationBufferMemory**: 대화의 전체 기록을 버퍼에 저장합니다.
    ```python
    from langchain.memory import ConversationBufferMemory
    from langchain_openai import OpenAI
    from langchain.chains import LLMChain
    from langchain_core.prompts import PromptTemplate

    # OPENAI_API_KEY 환경 변수 설정 필요
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
    # 대화형 프롬프트 템플릿
    template = """You are a nice chatbot having a conversation with a human.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)

    # ConversationBufferMemory는 "chat_history"라는 입력 변수를 사용합니다.
    memory = ConversationBufferMemory(memory_key="chat_history")

    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True, # 실행 과정 출력
        memory=memory
    )

    # 첫 번째 대화
    response1 = conversation.invoke({"question": "제 이름은 홍길동입니다."})
    print(response1['text'])

    # 두 번째 대화 (이전 대화 내용이 memory에 저장되어 활용됨)
    response2 = conversation.invoke({"question": "제 이름을 기억하시나요?"})
    print(response2['text'])
    # 출력 예시: 네, 홍길동님이라고 기억합니다.
    ```

-----

## 6\. Agents (에이전트)

LLM이 어떤 \*\*도구(Tools)\*\*를 사용할지, 어떤 순서로 사용할지, 그리고 그 결과를 어떻게 처리할지 스스로 결정하도록 하는 강력한 기능입니다. LLM을 추론 엔진으로 사용합니다.

  * **Tools (도구)**: 에이전트가 사용할 수 있는 특정 기능입니다. (예: 구글 검색, 계산기, 데이터베이스 조회 등)
  * **Agent Executor (에이전트 실행기)**: 에이전트와 도구들을 받아 실제로 실행하는 런타임입니다.

<!-- end list -->

```python
from langchain_openai import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun # SerpAPI 대신 DuckDuckGo 사용

# OPENAI_API_KEY 환경 변수 설정 필요
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

# 도구 로드 (DuckDuckGo 검색 도구)
# tools = load_tools(["ddg-search"], llm=llm) # 이전 방식
search = DuckDuckGoSearchRun()
tools = [search]


# 에이전트 초기화
# ZERO_SHOT_REACT_DESCRIPTION: 상황에 따라 어떤 도구를 사용할지 LLM이 결정
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# 에이전트 실행
question = "오늘 대한민국의 수도 날씨는 어떤가요?"
response = agent.invoke({"input": question})
print(response["output"])
# 출력 예시: (검색 결과에 따라 다름) 오늘 서울의 날씨는 [날씨 정보]입니다.
```

**참고**: `load_tools`에서 `serpapi`를 사용하려면 `SERPAPI_API_KEY` 환경 변수가 필요합니다. 예시에서는 무료로 사용 가능한 `DuckDuckGoSearchRun`을 사용했습니다.

-----

## Langchain의 장점 👍

  * **모듈성 및 유연성**: 필요한 구성 요소만 가져와 조합하여 맞춤형 애플리케이션 구축이 용이합니다.
  * **다양한 LLM 지원**: OpenAI 모델뿐만 아니라 Hugging Face 등 다양한 LLM과 쉽게 통합할 수 있습니다.
  * **풍부한 생태계**: 문서 로더, 벡터 스토어, 에이전트 도구 등 다양한 기능이 이미 구현되어 있거나 쉽게 확장 가능합니다.
  * **활발한 커뮤니티**: 지속적인 업데이트와 사용자 지원이 활발합니다.
  * **복잡한 작업 단순화**: RAG, 에이전트 등 고급 기능을 비교적 쉽게 구현할 수 있도록 추상화 수준을 제공합니다.

-----
