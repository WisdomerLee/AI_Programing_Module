# 알아야 할 것
1.  **LLMs (Large Language Models)**: 텍스트 문자열을 입력받아 텍스트 문자열을 출력하는 모델입니다.
2.  **Chat Models (챗 모델)**: 채팅 메시지 객체의 리스트를 입력받아 채팅 메시지 객체를 출력하는 모델로, 대화형 인터페이스에 더 적합합니다.
3.  **Text Embedding Models (텍스트 임베딩 모델)**: 텍스트를 의미를 담은 숫자 벡터(임베딩)로 변환하는 모델입니다.

-----

## 1\. LLMs (Large Language Models)

가장 기본적인 언어 모델 인터페이스입니다. 주로 단일 텍스트 입력을 받아 그에 대한 텍스트 응답을 생성합니다.

**주요 특징 및 사용법:**

  * **입력**: 문자열
  * **출력**: 문자열
  * **주요 메서드**:
      * `invoke(input, config=None, **kwargs)`: 모델을 호출하여 결과를 반환합니다. (이전의 `predict` 또는 `__call__`과 유사)
      * `stream(input, config=None, **kwargs)`: 모델을 호출하고 결과 토큰을 실시간 스트림으로 반환합니다.
      * `batch(inputs, config=None, **kwargs)`: 여러 입력에 대해 배치 처리를 합니다.

**주요 파라미터 (OpenAI 모델 기준):**

  * `model_name`: 사용할 모델의 이름 (예: `"gpt-3.5-turbo-instruct"`, `"text-davinci-003"`)
  * `temperature`: 생성 결과의 무작위성 조절 (0.0 \~ 2.0, 보통 0.0 \~ 1.0 사이 사용).
      * 낮을수록 결정론적이고 일관된 결과 생성.
      * 높을수록 다양하고 창의적인 결과 생성.
  * `max_tokens`: 생성할 최대 토큰 수. 응답 길이를 제어합니다.
  * `openai_api_key`: OpenAI API 키 (환경 변수 `OPENAI_API_KEY`로 설정하는 것이 일반적).

**Python 코드 예시:**

```python
import os
from langchain_openai import OpenAI

# API 키 설정 (실제 키로 대체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. 기본 LLM 초기화 (gpt-3.5-turbo-instruct는 비교적 저렴하고 빠릅니다)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.7, max_tokens=100)

# 2. invoke() 메서드를 사용한 단일 예측
prompt = "Langchain을 사용하면 어떤 점이 좋은가요? 간략하게 설명해주세요."
response = llm.invoke(prompt)
print(f"--- invoke() 결과 ---")
print(response)

# 3. temperature 조절에 따른 결과 변화
llm_creative = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=1.0, max_tokens=100) # 더 창의적으로
llm_deterministic = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.1, max_tokens=100) # 더 결정적으로

prompt_story = "옛날 옛날에 용감한 토끼가 살았습니다. 그 다음 이야기는?"

print(f"\n--- Temperature 1.0 (창의적) ---")
print(llm_creative.invoke(prompt_story))

print(f"\n--- Temperature 0.1 (결정적) ---")
print(llm_deterministic.invoke(prompt_story))


# 4. stream() 메서드를 사용한 실시간 응답 처리
print(f"\n--- stream() 결과 ---")
prompt_stream = "파이썬으로 간단한 'Hello World' 코드를 작성해줘."
for chunk in llm.stream(prompt_stream):
    print(chunk, end="", flush=True)
print("\n")

# 5. batch() 메서드를 사용한 여러 입력 처리
prompts = [
    "대한민국의 수도는?",
    "일본의 수도는?",
    "프랑스의 수도는?"
]
batch_responses = llm.batch(prompts)
print(f"\n--- batch() 결과 ---")
for res in batch_responses:
    print(res.strip()) # strip()으로 앞뒤 공백 제거

```

-----

## 2\. Chat Models (챗 모델)

챗 모델은 대화의 맥락을 더 잘 이해하고 처리하도록 설계되었습니다. 입력과 출력이 단순 문자열이 아닌, 역할(system, human, ai)이 지정된 메시지 객체의 리스트 형태입니다.

**주요 특징 및 사용법:**

  * **입력**: `Message` 객체의 리스트 (예: `SystemMessage`, `HumanMessage`, `AIMessage`)
  * **출력**: `AIMessage` (또는 `AIMessageChunk` - 스트리밍 시)
  * **메시지 유형**:
      * `SystemMessage`: AI의 행동 방식이나 역할을 정의하는 메시지 (예: "당신은 친절한 번역가입니다.")
      * `HumanMessage`: 사용자의 입력 메시지.
      * `AIMessage`: AI의 이전 응답 메시지 (대화 기록을 전달할 때 사용).
      * `ToolMessage` / `FunctionMessage`: (고급 기능) 에이전트가 도구를 사용한 결과를 전달할 때 사용.
  * **주요 메서드**: `invoke`, `stream`, `batch` (LLM과 유사)

**주요 파라미터 (OpenAI 챗 모델 기준):**

  * `model_name`: 사용할 챗 모델 이름 (예: `"gpt-4"`, `"gpt-3.5-turbo"`)
  * `temperature`, `max_tokens` 등 LLM과 유사한 파라미터 사용 가능.

**Python 코드 예시:**

```python
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# API 키 설정 (실제 키로 대체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. 기본 Chat Model 초기화
chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=150)

# 2. invoke() 메서드를 사용한 단일 예측 (메시지 리스트 전달)
messages_single_turn = [
    SystemMessage(content="You are a helpful assistant that explains complex concepts simply."),
    HumanMessage(content="양자 컴퓨팅에 대해 어린 아이도 이해할 수 있게 설명해줘.")
]
response_chat = chat.invoke(messages_single_turn)
print(f"--- Chat Model invoke() 결과 ---")
print(f"AI: {response_chat.content}")


# 3. 다중 턴 대화 (이전 대화 내용 포함)
messages_multi_turn = [
    SystemMessage(content="당신은 사용자와 짧은 농담을 주고받는 유쾌한 챗봇입니다."),
    HumanMessage(content="안녕? 농담 하나 해줄래?"),
    AIMessage(content="물론이죠! Q: 세상에서 가장 뜨거운 과일은? A: 천도복숭아! 깔깔!"), # AI의 이전 응답
    HumanMessage(content="하하 재밌네. 다른 것도 하나 더 해줘.")
]
response_chat_multi = chat.invoke(messages_multi_turn)
print(f"\n--- Chat Model 다중 턴 대화 결과 ---")
print(f"AI: {response_chat_multi.content}")


# 4. stream() 메서드를 사용한 실시간 채팅 응답 처리
print(f"\n--- Chat Model stream() 결과 ---")
messages_stream = [
    SystemMessage(content="You are a travel planner. Suggest a 3-day itinerary for Seoul."),
    HumanMessage(content="서울 3일 여행 계획 좀 짜줘. 나는 역사 유적지와 맛집 탐방을 좋아해.")
]
for chunk in chat.stream(messages_stream):
    print(chunk.content, end="", flush=True)
print("\n")


# 5. batch() 메서드를 사용한 여러 메시지 세트 처리
message_sets = [
    [HumanMessage(content="행복이란 무엇일까요?")],
    [
        SystemMessage(content="You are a poet."),
        HumanMessage(content="Write a short poem about the stars.")
    ]
]
batch_chat_responses = chat.batch(message_sets)
print(f"\n--- Chat Model batch() 결과 ---")
for res_msg in batch_chat_responses:
    print(f"AI: {res_msg.content.strip()}")

```

-----

## 3\. Text Embedding Models (텍스트 임베딩 모델)

텍스트를 숫자 벡터로 변환하여, 텍스트 간의 의미적 유사성을 비교하거나 검색, 클러스터링 등에 활용할 수 있게 합니다.

**주요 특징 및 사용법:**

  * **입력**: 단일 텍스트 (쿼리용) 또는 텍스트 리스트 (문서용)
  * **출력**: 숫자 벡터 (임베딩) 또는 숫자 벡터의 리스트
  * **주요 메서드**:
      * `embed_query(text)`: 단일 텍스트에 대한 임베딩을 생성합니다. (주로 사용자 검색어 임베딩 시)
      * `embed_documents(texts)`: 여러 텍스트(문서)에 대한 임베딩 리스트를 생성합니다. (주로 DB에 저장할 문서들 임베딩 시)

**주요 파라미터 (OpenAI 임베딩 모델 기준):**

  * `model_name`: 사용할 임베딩 모델 이름 (예: `"text-embedding-3-small"`, `"text-embedding-ada-002"`)
      * `text-embedding-3-small`은 최신 모델 중 하나로, 성능과 비용 효율성이 좋습니다.
  * `chunk_size`: (필요시) 한 번에 처리할 텍스트 청크 크기.

**Python 코드 예시:**

```python
import os
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity # 유사도 계산을 위해 import
import numpy as np

# API 키 설정 (실제 키로 대체하거나 환경 변수로 설정하세요)
# os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# 1. Text Embedding Model 초기화
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small") # 최신 모델 중 하나

# 2. embed_query() - 단일 텍스트 임베딩
query = "오늘 날씨가 좋네요!"
query_embedding = embeddings_model.embed_query(query)
print(f"--- embed_query() 결과 (처음 5개 차원) ---")
print(query_embedding[:5])
print(f"임베딩 벡터 차원 수: {len(query_embedding)}")


# 3. embed_documents() - 여러 텍스트 임베딩
documents = [
    "강아지는 귀여운 동물입니다.",
    "고양이는 독립적인 동물입니다.",
    "Langchain은 LLM 개발을 돕는 프레임워크입니다."
]
document_embeddings = embeddings_model.embed_documents(documents)
print(f"\n--- embed_documents() 결과 (첫 번째 문서의 처음 5개 차원) ---")
print(document_embeddings[0][:5])
print(f"임베딩된 문서 수: {len(document_embeddings)}")
print(f"각 임베딩 벡터 차원 수: {len(document_embeddings[0])}")


# 4. 임베딩을 활용한 간단한 의미 유사도 비교
text1 = "나는 사과를 좋아한다."
text2 = "나는 바나나를 좋아한다."
text3 = "오늘 점심 뭐 먹지?"

embedding1 = embeddings_model.embed_query(text1)
embedding2 = embeddings_model.embed_query(text2)
embedding3 = embeddings_model.embed_query(text3)

# scikit-learn의 cosine_similarity를 사용하기 위해 numpy 배열로 변환
embedding1_np = np.array(embedding1).reshape(1, -1)
embedding2_np = np.array(embedding2).reshape(1, -1)
embedding3_np = np.array(embedding3).reshape(1, -1)

similarity_1_2 = cosine_similarity(embedding1_np, embedding2_np)[0][0]
similarity_1_3 = cosine_similarity(embedding1_np, embedding3_np)[0][0]

print(f"\n--- 임베딩 유사도 비교 결과 ---")
print(f"'{text1}' vs '{text2}' 유사도: {similarity_1_2:.4f}") # 과일 이야기로 유사도 높음
print(f"'{text1}' vs '{text3}' 유사도: {similarity_1_3:.4f}") # 주제가 달라 유사도 낮음

```

-----

## Model Provider 통합

Langchain은 OpenAI뿐만 아니라 `Hugging Face Hub`, `Azure OpenAI Service`, `Anthropic (Claude)`, `Google Vertex AI` 등 다양한 모델 제공자를 지원합니다. 각 제공자에 맞는 클래스를 임포트하고 필요한 인증 정보를 설정하여 사용할 수 있습니다.

예를 들어, Hugging Face Hub의 오픈 소스 모델을 사용하려면 (일부 모델은 API 키가 필요 없을 수 있습니다):

```python
# from langchain_community.llms import HuggingFaceHub

# # Hugging Face API 토큰 설정 (필요한 경우)
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_HUGGINGFACE_API_TOKEN"

# # 예시: google/flan-t5-xxl 모델 (API 토큰 필요할 수 있음)
# # hf_llm = HuggingFaceHub(
# #     repo_id="google/flan-t5-xxl",
# #     model_kwargs={"temperature": 0.7, "max_length": 100}
# # )

# # try:
# #     hf_response = hf_llm.invoke("프랑스의 수도는 어디인가요?")
# #     print(f"\n--- Hugging Face Hub LLM 결과 ---")
# #     print(hf_response)
# # except Exception as e:
# #     print(f"Hugging Face Hub LLM 호출 중 오류: {e}")
# #     print("HuggingFaceHub API 토큰이 필요하거나, 모델 quota/rate limit 문제일 수 있습니다.")
```

사용 전에 model의 api 사용료 및 key 발급, 사용 rate 등을 자세히 따져보고 사용할 것

-----
