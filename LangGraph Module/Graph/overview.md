LangGraph는 주로 상태(State)를 중심으로 그래프를 구성하며, `StateGraph`와 `MessageGraph`가 가장 흔하게 사용됩니다.

---

# 1. `StateGraph`

`langgraph.graph.StateGraph`는 LangGraph에서 가장 일반적으로 사용되는 핵심 클래스로, 명시적인 상태(State)를 기반으로 순환적이고 복잡한 워크플로우를 구축할 수 있게 해줍니다.

**특징:**

* **상태 기반 (Stateful)**: 그래프 전체의 상태 스키마(일반적으로 `TypedDict` 또는 Pydantic `BaseModel` 사용)를 정의하고, 각 노드는 이 상태를 입력으로 받아 업데이트된 상태의 일부 또는 전체를 반환합니다. 반환된 부분 상태는 기존 상태에 자동으로 병합됩니다.
* **순환 허용 (Cyclical)**: 노드 간의 엣지를 통해 복잡한 루프나 조건부 흐름을 쉽게 구성할 수 있습니다.
* **LCEL Runnable로 컴파일**: `compile()` 메서드를 통해 Langchain Expression Language (LCEL)의 `Runnable` 프로토콜을 따르는 실행 가능한 객체로 변환되어, Langchain의 다른 부분과 원활하게 통합됩니다.
* **유연한 상태 업데이트**: 노드는 전체 상태 객체를 반환할 필요 없이, 변경하고자 하는 상태 필드만 포함된 딕셔너리를 반환하여 상태를 업데이트할 수 있습니다. `Annotated[List, operator.add]`와 같이 특정 필드에 값이 누적되도록 정의할 수도 있습니다.

**주요 메서드 및 역할:**

* **`__init__(self, state_schema: Type[Any])`**
    * **역할**: `StateGraph` 객체를 초기화합니다.
    * **인자**:
        * `state_schema`: 그래프 전체에서 사용될 상태의 타입을 정의합니다. (예: `class MyState(TypedDict): ...`)

* **`add_node(self, key: str, action: Union[Callable, Runnable])`**
    * **역할**: 그래프에 작업 단위인 노드(node)를 추가합니다.
    * **인자**:
        * `key` (str): 노드의 고유한 이름 (문자열). 이 이름을 사용하여 엣지를 연결합니다.
        * `action` (Callable | Runnable): 해당 노드가 실행할 실제 작업입니다. Python 함수(상태를 입력받고, 업데이트할 상태 부분을 담은 딕셔너리 반환) 또는 LCEL `Runnable` 객체가 될 수 있습니다.

* **`add_edge(self, start_key: str, end_key: str)`**
    * **역할**: 두 노드 간의 일반적인(무조건적인) 연결, 즉 엣지(edge)를 추가합니다. `start_key`로 지정된 노드가 실행 완료된 후에는 항상 `end_key`로 지정된 노드가 실행됩니다.
    * **인자**:
        * `start_key` (str): 시작 노드의 이름.
        * `end_key` (str): 다음으로 실행될 노드의 이름.

* **`add_conditional_edges(self, source_key: str, path: Callable[..., Union[str, list[str]]], path_map: Optional[dict[Any, str]] = None)`**
    * **역할**: `source_key` 노드의 실행 결과에 따라 다음으로 실행될 노드를 동적으로 결정하는 조건부 엣지를 추가합니다. 이는 복잡한 분기 로직을 구현하는 데 핵심적입니다.
    * **인자**:
        * `source_key` (str): 조건부 분기가 시작되는 노드의 이름입니다. 이 노드의 실행 결과가 `path` 함수로 전달됩니다.
        * `path` (Callable): `source_key` 노드의 실행 결과(보통 상태 객체)를 입력으로 받아, 다음으로 이동할 경로의 이름(문자열) 또는 여러 경로의 이름 리스트를 반환하는 함수입니다. `langgraph.graph.END`를 반환하면 해당 경로는 종료됩니다.
        * `path_map` (Optional[dict[Any, str]]): `path` 함수가 반환한 값(키)과 실제 다음 노드의 이름(값)을 매핑하는 딕셔너리입니다. 만약 `path` 함수가 직접 다음 노드의 이름을 반환한다면 `path_map`은 필요 없을 수 있습니다.

* **`set_entry_point(self, key: str)`**
    * **역할**: 그래프가 실행될 때 가장 먼저 시작될 진입 노드(entry point)를 지정합니다.
    * **인자**:
        * `key` (str): 시작 노드의 이름.

* **`set_finish_point(self, key: str)`**
    * **역할**: (최신 LangGraph 버전에서는 `END`를 사용하는 조건부 엣지가 더 일반적입니다.) 특정 노드가 실행 완료되면 그래프 전체의 실행을 종료하도록 지정할 수 있습니다.
    * **인자**:
        * `key` (str): 종료 노드의 이름.

* **`compile(self, checkpointer: Optional[BaseCheckpointSaver] = None, interrupt_before: Optional[Sequence[str]] = None, interrupt_after: Optional[Sequence[str]] = None, debug: bool = False)`**
    * **역할**: 지금까지 정의된 노드와 엣지 구성을 바탕으로 실제로 실행 가능한 `CompiledGraph` 객체를 생성합니다. 이 객체는 LCEL `Runnable` 인터페이스를 따릅니다.
    * **인자**:
        * `checkpointer` (Optional[BaseCheckpointSaver]): 그래프의 실행 상태를 저장하고 복원할 수 있는 체크포인터 객체 (예: `MemorySaver`, `SqliteSaver`). 이를 통해 긴 작업의 중단/재개, 병렬 세션 관리 등이 가능해집니다.
        * `interrupt_before` / `interrupt_after` (Optional[Sequence[str]]): 지정된 노드들의 실행 전 또는 후에 그래프 실행을 일시 중단(interrupt)할 수 있는 지점을 설정합니다. Human-in-the-loop과 같은 시나리오에 유용합니다.
        * `debug` (bool): 디버그 모드를 활성화합니다 (기본값: False).

---

# 2. `MessageGraph`

`langgraph.graph.MessageGraph`는 채팅 애플리케이션과 같이 대화 메시지의 연속을 주요 상태로 다루는 경우에 특화된 그래프 클래스입니다.

**특징:**

* **채팅 최적화**: 상태가 기본적으로 `Sequence[BaseMessage]` (Langchain의 메시지 객체 리스트)로 가정됩니다.
* **메시지 자동 누적**: 노드가 새로운 메시지(들)를 반환하면, 이 메시지들이 기존 상태의 메시지 리스트의 끝에 자동으로 추가됩니다. 이는 `StateGraph`에서 `Annotated[Sequence[BaseMessage], operator.add]`를 사용하는 것과 유사한 효과를 내부적으로 처리해줍니다.
* **간결한 에이전트 구현**: 메시지 기반의 대화형 에이전트(예: Langchain의 Agent Executor 패턴을 LangGraph로 직접 구현)를 만들 때 코드를 더 간결하게 작성할 수 있습니다.

**주요 메서드 및 역할:**

* **`__init__(self)`**:
    * **역할**: `MessageGraph` 객체를 초기화합니다. 별도의 상태 스키마를 전달받지 않으며, 상태는 메시지 리스트로 고정됩니다.
* **`add_node`, `add_edge`, `add_conditional_edges`, `set_entry_point`, `compile`**:
    * **역할**: `StateGraph`의 해당 메서드들과 거의 동일한 역할을 수행합니다. 다만, 모든 노드는 메시지 리스트를 현재 상태로 입력받고, 업데이트할 메시지(또는 메시지 리스트)를 반환하는 방식으로 작동합니다.

---

# 3. `Graph` (기본 그래프)

`langgraph.graph.Graph`는 `StateGraph`보다 더 저수준의 기본적인 그래프 클래스입니다.

**특징:**

* **유연하지만 더 많은 수동 설정 필요**: 상태를 어떻게 합칠지(reduce)에 대한 커스텀 로직(`reducer` 함수)을 직접 정의해야 할 수 있습니다.
* **특수한 경우 사용**: 대부분의 경우, 상태 관리가 더 편리한 `StateGraph`나 채팅에 특화된 `MessageGraph`를 사용하는 것이 권장됩니다. `Graph`는 매우 특수한 상태 병합 로직이 필요하거나 LangGraph의 내부 작동 방식을 깊이 이해하고 제어하고 싶을 때 고려될 수 있습니다.

**주요 메서드**: `StateGraph`와 유사한 구조를 가지지만, 상태 처리 방식에서 차이가 있을 수 있으며, `reducer`와 같은 추가적인 설정이 필요할 수 있습니다.

---

## 특별한 상수

* **`END` (from `langgraph.graph import END`)**:
    * **역할**: `add_conditional_edges`의 `path` 함수가 반환할 수 있는 특별한 문자열 상수입니다. `path` 함수가 특정 조건에서 `END`를 반환하면, 해당 경로의 그래프 실행은 거기서 종료됩니다.

---

`compile()`을 통해 생성된 `CompiledGraph` 객체는 `invoke`, `stream`, `batch` 등 표준적인 LCEL 메서드를 통해 실행됩니다.
