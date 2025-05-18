LCEL(Langchain Expression Language)
LCEL은 Langchain 구성 요소들을 조합하여 체인(Chain)을 만드는 새롭고 강력하며 선언적인 방식입니다.

-----

# 1\. LCEL (Langchain Expression Language) 이란?

LCEL은 Langchain의 다양한 구성 요소들(예: 프롬프트, 모델, 출력 파서, 리트리버, 사용자 정의 함수 등)을 **파이프(`|`) 연산자**를 사용하여 마치 데이터 흐름처럼 자연스럽게 연결하여 체인을 구성하는 방식입니다.

핵심 아이디어는 \*\*"모든 것이 Runnable이다"\*\*입니다. LCEL에서 체인을 구성하는 각 요소는 `Runnable`이라는 공통 프로토콜(인터페이스)을 따릅니다. 이 `Runnable` 프로토콜은 `invoke`, `batch`, `stream`, `ainvoke` (비동기 invoke) 등 일관된 실행 메서드를 제공하여, 어떤 구성 요소든 동일한 방식으로 호출하고 조합할 수 있게 합니다.

LCEL을 사용하면 다음과 같이 매우 직관적으로 체인을 작성할 수 있습니다:

```python
# 예시: 프롬프트 | 모델 | 출력 파서
chain = prompt_template | llm_model | output_parser
```

-----

# 2\. LCEL 도입 이유

LCEL이 도입된 배경에는 기존 Langchain 방식(예: `LLMChain`, `SequentialChain` 등)의 몇 가지 한계점과 이를 개선하려는 목표가 있습니다:

**기존 방식의 한계점:**

  * **복잡성 및 비일관성**: 각 레거시 체인 클래스마다 사용법이나 인터페이스가 조금씩 다를 수 있어 학습 곡선이 가파르고, 커스텀 체인을 만드는 것이 다소 번거로웠습니다.
  * **스트리밍/비동기/배치 지원의 어려움**: 이러한 고급 기능을 모든 종류의 체인에 일관되게 적용하고 확장하기 어려웠습니다. 예를 들어, 특정 체인에서는 스트리밍이 잘 지원되지만 다른 체인에서는 제한적일 수 있었습니다.
  * **디버깅 및 관찰의 어려움**: 복잡한 체인의 내부 동작을 추적하고 디버깅하는 것이 어려울 수 있었습니다.
  * **유연성 부족**: 작은 기능을 추가하거나 기존 체인의 일부만 수정하는 것이 유연하지 못했습니다.

**LCEL의 도입 목표 및 장점:**

1.  **단순성 및 가독성 향상**: 파이프 연산자(`|`)를 사용하여 체인의 데이터 흐름을 매우 직관적이고 선언적으로 표현할 수 있습니다. 코드가 간결해지고 이해하기 쉬워집니다.
2.  **일관된 인터페이스 (Runnable 프로토콜)**: 모든 LCEL 구성 요소는 `Runnable` 프로토콜을 따르므로 `invoke`, `stream`, `batch`, `ainvoke`, `astream`, `abatch` 등의 표준 메서드를 일관되게 사용할 수 있습니다.
3.  **강력한 스트리밍 지원**: 첫 번째 토큰까지의 시간(Time-To-First-Token, TTFT)을 최소화하고, 부분적인 결과를 실시간으로 스트리밍하는 것이 훨씬 쉬워졌습니다. 이는 사용자 경험 향상에 매우 중요합니다.
4.  **향상된 비동기 지원**: `asyncio`를 활용한 비동기 실행(`ainvoke`, `astream`, `abatch` 등)을 네이티브하고 일관되게 지원합니다.
5.  **병렬 실행 (`RunnableParallel`)**: 여러 Runnable을 동시에 병렬로 실행하고 그 결과를 쉽게 통합할 수 있습니다.
6.  **Fallback 및 재시도 메커니즘**: 특정 Runnable 실행에 실패했을 때 다른 Runnable을 실행하거나(fallback), 자동으로 재시도(retry)하는 기능을 쉽게 추가할 수 있습니다.
7.  **최고 수준의 구성 용이성 (Composability)**: 작은 단위의 Runnable들을 레고 블록처럼 자유롭게 조합하여 매우 복잡한 로직의 체인도 쉽게 구성할 수 있습니다.
8.  **디버깅 및 추적 용이성 (Observability)**: LangSmith와 같은 도구와 긴밀하게 통합되어, 체인의 각 단계별 실행 과정, 입출력, 시간 등을 상세하게 추적하고 디버깅하는 것이 매우 용이해졌습니다.
9.  **입출력 스키마 및 타입 힌팅**: 각 Runnable은 입력 및 출력 스키마를 명시적으로 가질 수 있어, 타입 안정성을 높이고 자동 문서화나 유효성 검사 등에 활용될 수 있습니다.

-----

# 3\. LCEL의 주요 특징 및 개념

  * **Runnable 프로토콜**: LCEL의 모든 구성 요소가 구현하는 핵심 인터페이스입니다. 주요 메서드는 다음과 같습니다:
      * `invoke(input, config=None)`: 단일 입력에 대해 실행하고 단일 출력을 반환합니다.
      * `batch(inputs, config=None)`: 여러 입력에 대해 배치 실행하고 출력 리스트를 반환합니다.
      * `stream(input, config=None)`: 단일 입력에 대해 실행하고 출력 청크(chunk)의 스트림을 반환합니다.
      * `ainvoke`, `abatch`, `astream`: 위 메서드들의 비동기 버전입니다.
  * **파이프 연산자 (`|`)**: 두 `Runnable`을 연결하여 `RunnableSequence`를 만듭니다. 첫 번째 Runnable의 출력이 두 번째 Runnable의 입력으로 전달됩니다.
  * **주요 `Runnable` 구현체**:
      * **Models**: `ChatOpenAI()`, `OpenAI()` 등 LLM 및 챗 모델 자체가 Runnable입니다.
      * **Prompts**: `ChatPromptTemplate.from_template(...)` 등 프롬프트 템플릿도 Runnable입니다. 입력을 받아 포맷팅된 프롬프트를 출력합니다.
      * **OutputParsers**: `StrOutputParser()`, `JsonOutputParser()` 등 출력 파서도 Runnable입니다. LLM의 출력을 받아 원하는 형태로 변환합니다.
      * **Retrievers**: 벡터 스토어의 `as_retriever()`로 생성된 리트리버도 Runnable입니다. 쿼리를 받아 관련 문서를 반환합니다.
      * **`RunnablePassthrough()`**: 입력을 변경하지 않고 그대로 다음 단계로 전달하거나, 현재 컨텍스트에 새로운 키-값 쌍을 추가하는 데 사용됩니다. RAG 체인에서 원본 질문을 유지할 때 유용합니다.
      * **`RunnableParallel({...})`**: 여러 Runnable을 병렬로 실행하고, 각 Runnable의 결과를 지정된 키와 함께 딕셔너리 형태로 묶어서 반환합니다. `dict`를 사용하여 정의합니다.
      * **`RunnableLambda(func)`**: 임의의 Python 함수(또는 lambda 함수)를 Runnable로 쉽게 변환할 수 있습니다.
      * **`RunnableBranch(branches, default)`**: 조건에 따라 다른 Runnable 경로를 선택하여 실행합니다.
      * **`RunnableConfigurableFields(...)` / `configurable_alternatives(...)` / `configurable_fields(...)`**: 실행 시점에 특정 Runnable의 필드(예: LLM의 `temperature`, 프롬프트의 변수)를 동적으로 설정하거나, 여러 Runnable 중 하나를 선택하여 사용할 수 있게 합니다.
  * **LangSmith 통합**: LCEL로 구성된 체인은 실행 시 LangSmith에 자동으로 트레이스가 기록되어 (환경 변수 설정 시) 디버깅과 모니터링이 매우 편리합니다.

-----

# 4\. LCEL 사용 예시 코드

**환경 설정 (필수):**

```python
import os
from langchain_openai import ChatOpenAI # langchain-openai 필요
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

# OpenAI API 키 설정 (실제 키로 대체하거나 환경 변수로 설정)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# LangSmith 설정 (선택 사항이지만 강력 권장, 디버깅 및 추적에 유용)
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "YOUR_LANGSMITH_API_KEY"
# os.environ["LANGCHAIN_PROJECT"] = "My LCEL Project" # 프로젝트 이름 지정

# 모델 초기화
chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
```

## 4.1. 기본적인 체인 구성 (Prompt + LLM + OutputParser)

```python
# 1. 프롬프트 템플릿
prompt_basic = ChatPromptTemplate.from_template(
    "'{topic}'에 대한 농담 하나만 만들어주세요."
)

# 2. 출력 파서
parser_str = StrOutputParser() # LLM의 메시지 객체 출력을 문자열로 변환

# 3. LCEL 체인 구성
chain_basic = prompt_basic | chat_llm | parser_str

# 체인 실행
print("--- 기본 LCEL 체인 실행 ---")
response_basic = chain_basic.invoke({"topic": "프로그래머"})
print(f"결과: {response_basic}")
```

## 4.2. 스트리밍(Streaming) 예시

```python
print("\n--- LCEL 스트리밍 예시 ---")
# 위 chain_basic을 그대로 사용
chunks = []
for chunk in chain_basic.stream({"topic": "고양이"}):
    print(chunk, end="", flush=True)
    chunks.append(chunk)
print("\n스트리밍 완료!")
full_response_stream = "".join(chunks)
# print(f"전체 스트리밍 결과: {full_response_stream}")
```

## 4.3. 병렬 실행 (`RunnableParallel`) 예시

`RunnableParallel`은 여러 작업을 동시에 수행하고 그 결과를 딕셔너리로 모을 때 유용합니다.

```python
# 하나의 주제에 대해 이야기와 시를 동시에 생성
chain_story = ChatPromptTemplate.from_template("'{topic}'에 대한 짧은 이야기를 써주세요.") | chat_llm | parser_str
chain_poem = ChatPromptTemplate.from_template("'{topic}'에 대한 짧은 시를 써주세요.") | chat_llm | parser_str

# RunnableParallel을 사용하여 두 체인을 병렬로 실행할 준비
# 입력은 'topic' 하나를 받아 각 체인에 전달됨
map_chain = RunnableParallel(
    story=chain_story,
    poem=chain_poem
)

print("\n--- RunnableParallel 예시 ---")
response_parallel = map_chain.invoke({"topic": "우주여행"})
print(f"이야기: {response_parallel['story']}")
print(f"시: {response_parallel['poem']}")
```

## 4.4. `RunnablePassthrough` 사용 예시 (RAG 체인에서)

RAG(Retrieval Augmented Generation) 체인에서 검색된 문서(context)와 원본 질문(question)을 모두 다음 프롬프트로 전달해야 할 때 유용합니다.

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 간단한 벡터 스토어 및 리트리버 설정 (실제로는 더 많은 문서 사용)
try:
    documents = [
        Document(page_content="사과는 빨갛고 맛있는 과일입니다."),
        Document(page_content="바나나는 길고 노란색이며 달콤합니다.")
    ]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # RAG 프롬프트
    template_rag = """다음 컨텍스트를 사용하여 질문에 답하세요:
    컨텍스트: {context}
    질문: {question}
    답변:"""
    prompt_rag = ChatPromptTemplate.from_template(template_rag)

    # LCEL을 사용한 RAG 체인
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), # 검색된 문서를 문자열로 변환
            "question": RunnablePassthrough()  # 원본 질문을 그대로 전달
        }
        | prompt_rag
        | chat_llm
        | parser_str
    )

    print("\n--- RunnablePassthrough (RAG) 예시 ---")
    response_rag = rag_chain.invoke("사과는 어떤 과일인가요?")
    print(f"RAG 답변: {response_rag}")

    response_rag_banana = rag_chain.invoke("바나나의 특징은?")
    print(f"RAG 답변 (바나나): {response_rag_banana}")

except ImportError:
    print("\nFAISS 또는 관련 라이브러리가 설치되지 않았습니다. `pip install faiss-cpu langchain-openai` 등을 실행해주세요.")
except Exception as e:
    print(f"\nRAG 예제 중 오류 발생: {e}")
```

## 4.5. `RunnableLambda` 사용 예시

간단한 Python 함수를 체인의 일부로 쉽게 통합할 수 있습니다.

```python
def to_uppercase(input_text: str) -> str:
    return input_text.upper()

def add_greeting(name: str) -> str:
    return f"안녕하세요, {name}님!"

# RunnableLambda를 사용하여 함수를 Runnable로 변환
runnable_upper = RunnableLambda(to_uppercase)
runnable_greet = RunnableLambda(add_greeting)

# 체인 구성
greeting_chain_lambda = runnable_greet | runnable_upper

print("\n--- RunnableLambda 예시 ---")
result_lambda = greeting_chain_lambda.invoke("홍길동") # 입력은 add_greeting 함수의 인자 'name'에 매핑됨
print(f"Lambda 체인 결과: {result_lambda}") # 출력: 안녕하세요, 홍길동님! -> 안녕하세요, 홍길동님! (StrOutputParser가 없으면 그대로)

# 모델과 함께 사용
name_prompt = ChatPromptTemplate.from_template("{person}의 별명을 지어주세요.")
nickname_chain = (
    name_prompt
    | chat_llm
    | parser_str
    | RunnableLambda(lambda nickname: f"별명은 '{nickname.strip()}' 입니다. 이 별명을 대문자로 만들면: ") # 중간 처리
    | runnable_upper # 대문자 변환을 하지 않고, 이전 RunnableLambda의 출력을 받아 새 RunnableLambda가 처리
    | RunnableLambda(lambda final_text: final_text + nickname.upper()) # 이렇게 하면 안됨. nickname 변수 스코프 문제
)

# 더 나은 Lambda 사용 예시 (모델 출력 후 추가 작업)
def format_nickname_response(nickname: str) -> str:
    return f"모델이 생성한 별명: '{nickname.strip()}'. 이것을 대문자로 바꾸면: {nickname.strip().upper()}"

nickname_chain_better = (
    ChatPromptTemplate.from_template("{person}의 재미있는 별명 하나만 알려줘.")
    | chat_llm
    | StrOutputParser()
    | RunnableLambda(format_nickname_response) # 모델 출력을 받아 후처리
)
print(nickname_chain_better.invoke({"person": "개발자"}))
```

## 4.6. Fallback 및 재시도 예시 (개념)

```python
# from langchain_openai import OpenAI # 다른 모델 예시

# # 기본 LLM
# llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
# # 대체 LLM (예: 기본 LLM 호출 실패 시 사용)
# fallback_llm = OpenAI(model_name="text-davinci-001", max_tokens=10) # 오래된 모델 예시

# # .with_fallbacks()를 사용하여 대체 실행 설정
# chain_with_fallback = llm.with_fallbacks([fallback_llm])

# # .with_retry()를 사용하여 재시도 설정
# # (실제로는 오류 유형 등을 지정하여 더 정교하게 설정 가능)
# chain_with_retry = llm.with_retry(stop_after_attempt=3)

# # 사용 예시 (간단히)
# try:
#     # 일부러 오류를 유발하는 상황은 만들기 어려우므로 개념만 소개
#     # response_fallback = chain_with_fallback.invoke("매우 긴 프롬프트 또는 API 제한 초과 시도...")
#     # response_retry = chain_with_retry.invoke("일시적인 네트워크 오류 상황 가정...")
#     pass
# except Exception as e:
#     print(f"Fallback/Retry 예외: {e}")
```
