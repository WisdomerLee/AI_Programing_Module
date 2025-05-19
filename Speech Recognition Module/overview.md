주로 사용되는 모듈은 **`SpeechRecognition`** 라이브러리입니다.
이 라이브러리는 다양한 음성 인식 엔진 및 API를 편리하게 사용할 수 있도록 도와주는 래퍼(wrapper)입니다.

-----

# `SpeechRecognition` 라이브러리

`SpeechRecognition` 라이브러리는 다음과 같은 여러 엔진과 API를 지원합니다:

1.  **CMU Sphinx (오프라인)**: 인터넷 연결 없이 사용 가능하지만, 설정이 다소 복잡하고 한국어 지원이 제한적일 수 있습니다.
2.  **Google Web Speech API (온라인)**: `recognize_google()` 함수를 통해 사용. 별도의 API 키 없이 사용 가능하나, 일일 사용량 제한이 있을 수 있습니다.
3.  **Google Cloud Speech-to-Text (온라인)**: `recognize_google_cloud()` 함수를 통해 사용. API 키 필요. 매우 높은 인식률과 다양한 기능을 제공합니다. (유료)
4.  **Microsoft Azure Speech (온라인)**: API 키 필요.
5.  **Wit.ai (온라인)**: API 키 필요.
6.  **Houndify (온라인)**: API 키 필요.
7.  **IBM Speech to Text (온라인)**: API 키 필요.

-----

#  음성 인식의 일반적인 동작 원리

음성 인식 시스템은 대략 다음과 같은 단계를 거쳐 소리를 텍스트로 변환합니다:

1.  **오디오 입력**: 마이크나 오디오 파일로부터 음성 신호를 받습니다. 이 아날로그 신호는 디지털 데이터로 변환됩니다 (샘플링 및 양자화).
2.  **전처리 (Preprocessing)**: 입력된 오디오에서 잡음(noise)을 제거하고, 음성 구간을 탐지하며, 신호를 정규화하는 등의 작업을 수행합니다.
3.  **특징 추출 (Feature Extraction)**: 오디오 신호에서 음성의 특징을 나타내는 중요한 파라미터들을 추출합니다. 가장 널리 사용되는 방법 중 하나는 MFCC (Mel-Frequency Cepstral Coefficients)입니다. 이는 사람이 소리를 인지하는 방식을 모방하여 음성 특징을 추출합니다.
4.  **음향 모델 (Acoustic Model, AM)**: 추출된 특징을 바탕으로 가장 가능성이 높은 최소 소리 단위(음소, phoneme) 또는 단어의 일부(subword unit)를 예측합니다. 이 모델은 대량의 음성 데이터와 해당 전사 데이터를 사용하여 훈련됩니다. (예: HMM - Hidden Markov Model, DNN - Deep Neural Network)
5.  **언어 모델 (Language Model, LM)**: 음향 모델에서 나온 음소/단어 후보들이 실제로 의미 있는 단어나 문장을 구성할 확률을 계산합니다. 예를 들어, "아이스크림"은 "아이 스크림"보다 확률이 높게 평가됩니다. (예: N-gram 모델, RNN 기반 언어 모델)
6.  **디코딩 (Decoding)**: 음향 모델과 언어 모델을 결합하여, 입력된 음성 특징에 대해 가장 가능성이 높은 단어 시퀀스(문장)를 찾아냅니다.

`SpeechRecognition` 라이브러리는 주로 3\~6번 과정을 외부 API에 의존하거나, CMU Sphinx의 경우 자체적으로 수행합니다. 사용자는 1번(오디오 입력 제공)과 라이브러리 함수 호출만 신경 쓰면 됩니다.

-----

#  `SpeechRecognition` 활용 예시 코드

먼저 필요한 라이브러리를 설치해야 합니다.

```bash
pip install SpeechRecognition PyAudio
```

  * `SpeechRecognition`: 핵심 라이브러리
  * `PyAudio`: 마이크 입력을 처리하기 위해 필요합니다. (Windows에서는 `pipwin install pyaudio` 또는 미리 컴파일된 .whl 파일 설치가 필요할 수 있습니다. Linux에서는 `portaudio` 관련 패키지 설치가 선행되어야 할 수 있습니다.)

## 1\. 마이크 입력을 텍스트로 변환 (Google Web Speech API 사용)

```python
import speech_recognition as sr

# Recognizer 객체 생성
r = sr.Recognizer()

# 마이크를 오디오 소스로 사용
with sr.Microphone() as source:
    print("말씀해주세요...")
    # 마이크로부터 오디오 데이터를 읽음 (주변 소음 레벨에 맞춰 자동 조정)
    # r.adjust_for_ambient_noise(source) # 필요한 경우 사용
    try:
        audio = r.listen(source, timeout=5, phrase_time_limit=5) # 5초 동안 듣거나, 5초 동안 말이 없으면 종료
        print("음성 인식 중...")
        # Google Web Speech API를 사용하여 한국어로 인식
        text = r.recognize_google(audio, language='ko-KR')
        print(f"인식된 텍스트: {text}")
    except sr.WaitTimeoutError:
        print("시간 초과: 지정된 시간 동안 음성이 감지되지 않았습니다.")
    except sr.UnknownValueError:
        print("Google Web Speech API가 오디오를 이해할 수 없습니다.")
    except sr.RequestError as e:
        print(f"Google Web Speech API 서비스 요청에 실패했습니다; {e}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")

```

**동작 원리 (위 코드 기준):**

1.  `sr.Recognizer()`: 음성 인식을 위한 객체를 초기화합니다.
2.  `sr.Microphone()`: 시스템의 기본 마이크를 오디오 입력 소스로 설정합니다.
3.  `r.listen(source)`: 마이크로부터 음성 입력을 받습니다. `listen`은 음성이 시작될 때까지 기다렸다가, 음성이 끝나면 해당 오디오 데이터를 반환합니다. `timeout`은 음성 입력을 기다리는 최대 시간, `phrase_time_limit`은 녹음할 구문의 최대 길이입니다.
4.  `r.recognize_google(audio, language='ko-KR')`: 캡처된 `audio` 데이터를 Google Web Speech API로 전송하여 텍스트로 변환합니다. `language='ko-KR'`로 한국어 인식을 지정합니다.
5.  예외 처리:
      * `sr.WaitTimeoutError`: 정해진 시간 동안 아무 말도 하지 않으면 발생합니다.
      * `sr.UnknownValueError`: API가 음성을 이해할 수 없을 때 (예: 너무 시끄럽거나 발음이 불분명할 때) 발생합니다.
      * `sr.RequestError`: API 서버에 연결할 수 없거나 API 자체에 문제가 있을 때 발생합니다.

## 2\. 오디오 파일(.wav)을 텍스트로 변환 (Google Web Speech API 사용)

먼저, 예시로 사용할 `audio.wav` 파일이 있다고 가정합니다.

```python
import speech_recognition as sr

# Recognizer 객체 생성
r = sr.Recognizer()

# 오디오 파일 경로
AUDIO_FILE = "audio.wav" # 여기에 실제 wav 파일 경로를 입력하세요.

# 오디오 파일을 오디오 소스로 사용
with sr.AudioFile(AUDIO_FILE) as source:
    print(f"'{AUDIO_FILE}' 파일에서 오디오 읽는 중...")
    try:
        audio = r.record(source)  # 파일 전체를 읽음
        # audio = r.record(source, duration=10) # 처음 10초만 읽음
        # audio = r.record(source, offset=5, duration=10) # 5초 지점부터 10초 동안 읽음

        print("음성 인식 중...")
        # Google Web Speech API를 사용하여 한국어로 인식
        text = r.recognize_google(audio, language='ko-KR')
        print(f"인식된 텍스트: {text}")
    except sr.UnknownValueError:
        print("Google Web Speech API가 오디오를 이해할 수 없습니다.")
    except sr.RequestError as e:
        print(f"Google Web Speech API 서비스 요청에 실패했습니다; {e}")
    except FileNotFoundError:
        print(f"오디오 파일을 찾을 수 없습니다: {AUDIO_FILE}")
    except Exception as e:
        print(f"알 수 없는 오류 발생: {e}")
```

**동작 원리 (위 코드 기준):**

1.  `sr.AudioFile(AUDIO_FILE)`: 지정된 경로의 오디오 파일을 입력 소스로 엽니다. `.wav` 형식이 가장 일반적이며, 다른 형식을 사용하려면 `ffmpeg` 같은 추가 도구가 필요할 수 있습니다.
2.  `r.record(source)`: 오디오 파일의 내용을 `audio` 데이터 객체로 읽어들입니다. `duration`이나 `offset` 파라미터를 사용하여 파일의 특정 부분만 읽을 수도 있습니다.
3.  이후 과정은 마이크 입력 예시와 동일하게 `r.recognize_google()`을 통해 API로 전송하고 결과를 받습니다.

-----

## 추가 팁

  * **오디오 품질**: 깨끗한 오디오 입력은 인식률을 크게 향상시킵니다. 주변 소음이 적은 환경에서, 좋은 품질의 마이크를 사용하는 것이 좋습니다.
  * **API 키**: Google Cloud Speech-to-Text, Azure 등 더 강력한 API를 사용하려면 해당 서비스에 가입하고 API 키를 발급받아 코드에 설정해야 합니다. `SpeechRecognition` 라이브러리는 이를 위한 파라미터들을 제공합니다.
  * **오프라인 인식 (CMU Sphinx)**:
      * 설치가 다소 복잡하고, 한국어 모델을 구하고 설정하는 것이 어려울 수 있습니다.
      * 인식률은 온라인 API에 비해 떨어질 수 있지만, 인터넷 연결이 필요 없고 무료라는 장점이 있습니다.
      * `r.recognize_sphinx(audio, language='ko-KR')`와 같이 사용할 수 있지만, 한국어 언어팩 및 음향 모델 설치가 선행되어야 합니다.
