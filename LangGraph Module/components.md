LangGraph는 Langchain 생태계 내에서 순환적이고 상태 기반의 복잡한 LLM 애플리케이션을 구축하기 위한 라이브러리입니다.
그 자체로 거대한 모듈 집합이라기보다는, 특정 목적을 가진 몇 가지 핵심 클래스와 함수들로 구성되어 있습니다.
주요 "모듈" 또는 "구성 요소"는 다음과 같이 정리할 수 있습니다:

-----

# 1\. 그래프 정의 및 실행 (Core Graph Engine)

이 부분이 LangGraph의 가장 핵심적인 엔진 역할을 합니다.

  * **`langgraph.graph.StateGraph`**:
      * **역할**: 상태(State)를 명시적으로 관리하며 순환적인 흐름을 만들 수 있는 그래프를 정의하는 주된 클래스입니다. 그래프의 각 노드는 상태를 입력받아 업데이트된 상태를 반환합니다.
      * **주요 메서드**:
          * `__init__(state_schema: Type[Any])`: 그래프의 상태 스키마(보통 `TypedDict` 또는 Pydantic `BaseModel`)를 인자로 받아 초기화합니다.
          * `add_node(key: str, action: Union[Callable, Runnable])`: 그래프에 노드를 추가합니다. `key`는 노드의 고유 이름이며, `action`은 해당 노드가 수행할 작업(함수 또는 LCEL Runnable)입니다.
          * `add_edge(start_key: str, end_key: str)`: 두 노드 간의 일반적인 (무조건적인) 연결을 추가합니다. `start_key` 노드 완료 후 `end_key` 노드가 실행됩니다.
          * `add_conditional_edges(source_key: str, path: Callable, path_map: Optional[Dict[Any, str]] = None)`: `source_key` 노드의 실행 결과에 따라 다음으로 실행될 노드를 동적으로 결정하는 조건부 엣지를 추가합니다.
              * `path`: `source_key` 노드의 출력을 받아 다음 경로(노드 이름 또는 `END`)를 반환하는 함수.
              * `path_map`: `path` 함수가 반환할 수 있는 각 값에 대해 실제 다음 노드 이름을 매핑한 딕셔너리. (선택 사항, `path` 함수가 직접 노드 이름을 반환할 수도 있음)
          * `set_entry_point(key: str)`: 그래프 실행 시 가장 먼저 실행될 노드를 지정합니다.
          * `set_finish_point(key: str)`: (최신 버전에서는 잘 사용되지 않으며, 조건부 엣지에서 `END`를 사용하는 것이 일반적입니다.) 특정 노드가 완료되면 그래프 실행을 종료하도록 지정합니다.
          * `compile(checkpointer: Optional[BaseCheckpointSaver] = None, interrupt_before: Optional[Sequence[str]] = None, interrupt_after: Optional[Sequence[str]] = None, debug: bool = False)`: 정의된 그래프 구조를 실행 가능한 `CompiledGraph` (LCEL Runnable) 객체로 컴파일합니다.
              * `checkpointer`: 그래프의 상태를 저장하고 복원할 수 있는 체크포인터 객체.
              * `interrupt_before`/`interrupt_after`: 특정 노드 실행 전/후에 실행을 일시 중단할 수 있게 합니다 (Human-in-the-loop 구현 시 유용).
  * **`langgraph.graph.Graph`**:
      * `StateGraph`의 더 기본적인 형태로, 상태를 명시적으로 합치거나 업데이트하는 방식이 다를 수 있습니다. 대부분의 경우 `StateGraph`가 더 사용하기 편리합니다.
  * **`langgraph.graph.MessageGraph`**:
      * **역할**: 채팅 애플리케이션과 같이 상태가 메시지 리스트(`Sequence[BaseMessage]`)로 구성되는 경우에 특화된 그래프입니다. 새로운 메시지가 기존 메시지 리스트에 추가되는 방식으로 상태가 자연스럽게 업데이트됩니다 (내부적으로 `operator.add` 사용).
      * 에이전트 실행기(Agent Executor)를 LangGraph로 직접 구축할 때 매우 유용합니다.
  * **`END` (from `langgraph.graph import END`)**:
      * 조건부 엣지에서 그래프 실행을 종료시키고 싶을 때 `path` 함수가 반환하는 특별한 값입니다.

-----

# 2\. 상태 정의 (State Definition)

LangGraph는 명시적인 상태 스키마를 사용합니다. 이는 Python의 타입을 활용하여 정의됩니다.

  * **`typing.TypedDict`**:
      * 간단한 딕셔너리 형태의 상태를 정의할 때 주로 사용됩니다. 각 키와 해당 값의 타입을 명시합니다.
      * 상태의 특정 필드를 업데이트할 때는 해당 키에 새로운 값을 할당하는 딕셔너리를 노드에서 반환합니다.
  * **Pydantic `BaseModel`**:
      * 더 복잡한 데이터 구조나 유효성 검사가 필요한 경우 Pydantic 모델을 사용하여 상태를 정의할 수 있습니다.
  * **상태 업데이트 방식**:
      * `StateGraph`에서는 노드가 상태의 일부 필드만 업데이트하는 딕셔너리를 반환하면, LangGraph가 이를 기존 상태에 병합(merge)합니다.
      * `operator.add`와 `Annotated`를 사용하여 리스트 형태의 상태 필드에 요소를 누적 추가할 수 있습니다 (예: `MessageGraph`의 메시지, 에이전트의 중간 단계 등).
    <!-- end list -->
    ```python
    from typing import TypedDict, Annotated, Sequence
    import operator
    from langchain_core.messages import BaseMessage

    class MyState(TypedDict):
        input_query: str
        intermediate_results: Annotated[Sequence[str], operator.add] # 결과 누적
        messages: Annotated[Sequence[BaseMessage], operator.add] # 메시지 누적
        final_answer: str | None
    ```

-----

# 3\. 체크포인터 (Checkpointers for Persistence and Resilience)

그래프의 실행 상태를 저장하고 복원하여, 긴 작업이 중단되었을 때 이어가거나, 병렬적인 여러 대화 세션을 관리할 수 있게 합니다.

  * **`langgraph.checkpoint.base.BaseCheckpointSaver`**:
      * 모든 체크포인터가 구현해야 하는 추상 기본 클래스입니다.
  * **구현체**:
      * `langgraph.checkpoint.memory.MemorySaver`: 상태를 인메모리에 저장합니다. 간단한 테스트나 단일 세션에 적합하지만, 애플리케이션 재시작 시 상태가 소실됩니다.
      * `langgraph.checkpoint.sqlite.SqliteSaver`: 상태를 SQLite 데이터베이스에 저장합니다. 로컬 파일 기반의 지속성이 필요할 때 유용합니다.
          * `SqliteSaver.from_conn_string("my_checkpoint_db.sqlite")`
      * `langgraph.checkpoint.aiosqlite.AsyncSqliteSaver`: SQLite를 비동기적으로 사용하는 버전입니다.
      * **클라우드 기반 체크포인터 (주로 비동기 지원)**:
          * `langgraph.checkpoint. τότεgres.AsyncPostgresSaver` (PostgreSQL)
          * `langgraph.checkpoint.aredis.AsyncRedisSaver` (Redis)
          * (향후 더 많은 백엔드 지원 가능성 있음)
  * **사용법**:
      * `graph.compile(checkpointer=my_checkpointer)`와 같이 컴파일 시점에 체크포인터를 전달합니다.
      * 컴파일된 그래프를 `invoke`, `stream` 등으로 호출할 때 `configurable={"thread_id": "some_unique_session_id"}`와 같이 `thread_id`를 제공하여 각 실행/대화 세션의 상태를 구분하고 관리합니다.

-----

# 4\. 사전 구축된 유틸리티 및 통합 (Prebuilt Utilities and Integrations)

LangGraph는 주로 개발자가 직접 그래프를 구성하는 유연성을 제공하는 데 초점을 맞추지만, 특정 패턴을 쉽게 구현하기 위한 몇 가지 유틸리티나 통합 지점이 있습니다.

  * **LCEL과의 통합**:
      * LangGraph의 노드는 LCEL `Runnable` 객체가 될 수 있습니다. 즉, 복잡한 LCEL 체인을 하나의 노드로 만들어 LangGraph 워크플로우에 통합할 수 있습니다.
      * `compile()`된 LangGraph 자체도 LCEL `Runnable`이므로, 다른 LCEL 체인과 결합될 수 있습니다.
  * **LangSmith 통합**:
      * LangGraph로 구축된 애플리케이션은 LangSmith와 긴밀하게 통합됩니다. 환경 변수(`LANGCHAIN_TRACING_V2="true"` 등)를 설정하면, 그래프의 각 노드 실행, 상태 변화, 입출력 등이 LangSmith에 상세하게 기록되어 디버깅, 모니터링, 성능 분석에 매우 유용합니다.
  * **Agent Executor 패턴 구현**:
      * LangGraph는 기존의 `AgentExecutor`가 수행하던 역할을 개발자가 직접 그래프로 더욱 세밀하게 제어하며 구축할 수 있는 강력한 방법을 제공합니다. 이를 위한 명시적인 "사전 구축된 에이전트 그래프"가 있는 것은 아니지만, `MessageGraph`와 조건부 엣지를 활용하여 ReAct, OpenAI Functions/Tools Agent 등의 패턴을 직접 구현하는 것이 일반적입니다. Langchain 문서나 예제에서 이러한 패턴을 LangGraph로 구현하는 방법을 찾아볼 수 있습니다.

-----

**주요 모듈 요약**:

  * **그래프 정의 및 실행**: `StateGraph`, `Graph`, `MessageGraph`, `END` (in `langgraph.graph`)
  * **상태 정의**: Python의 `TypedDict`, Pydantic `BaseModel`, `Annotated`와 `operator.add` (in `typing`, `operator`)
  * **지속성 및 상태 복원**: `BaseCheckpointSaver` 및 그 구현체들 (e.g., `MemorySaver`, `SqliteSaver` in `langgraph.checkpoint.*`)

LangGraph는 이러한 핵심 구성 요소들을 조합하여, 개발자가 LLM 애플리케이션의 상태와 제어 흐름을 매우 정교하게 설계하고 구현할 수 있도록 지원합니다.
