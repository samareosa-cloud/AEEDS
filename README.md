# AEEDS

Advanced Electrical and Electronic Design Laboratory and Software Practice

전기전자심화설계 및 소프트웨어실습에서 진행한 실습 코드를 정리한 Repository입니다.

---

## Development Environment

- C++
- Visual Studio
- OpenCV 3.4.5
- x64
- VC15

---

## OpenCV Setup

본 Repository의 프로젝트는 **OpenCV 3.4.5**를 사용합니다.

### 1. OpenCV 설치

OpenCV 3.4.5를 설치한 후 `build` 폴더의 위치를 확인합니다.

예시:

```text
C:\opencv\build
```

개인 PC의 설치 위치에 따라 경로는 달라질 수 있습니다.

---

### 2. `OPENCV_DIR` 환경 변수 설정

Windows의 **환경 변수**에서 다음 사용자 변수를 추가합니다.

```text
변수 이름: OPENCV_DIR
변수 값: C:\path\to\opencv\build
```

예시:

```text
OPENCV_DIR=C:\opencv\build
```

---

### 3. OpenCV DLL 경로 설정

프로그램 실행 시 OpenCV DLL을 찾을 수 있도록 Windows의 `Path` 환경 변수에 다음 경로를 추가합니다.

```text
%OPENCV_DIR%\x64\vc15\bin
```

해당 경로에는 다음과 같은 OpenCV DLL 파일이 위치합니다.

```text
opencv_world345.dll
opencv_world345d.dll
```

환경 변수를 수정한 후에는 Visual Studio를 다시 실행합니다.

---

### 4. Visual Studio Property Sheet 설정

Repository의 `OpenCV_x64.props` 파일을 사용하여 OpenCV 설정을 적용할 수 있습니다.

`OpenCV_x64.props`는 다음 항목을 자동으로 설정합니다.

- OpenCV Include Directory
- OpenCV Library Directory
- Debug Library
- Release Library

사용되는 라이브러리는 다음과 같습니다.

```text
Debug   : opencv_world345d.lib
Release : opencv_world345.lib
```

Visual Studio에서 다음 순서로 Property Sheet를 추가합니다.

```text
View
→ Other Windows
→ Property Manager
→ Debug | x64 또는 Release | x64
→ Add Existing Property Sheet
→ OpenCV_x64.props 선택
```

Debug와 Release를 모두 사용하는 경우 각각 `OpenCV_x64.props`를 연결합니다.

---

## OpenCV Directory Configuration

`OpenCV_x64.props`에서는 개인 PC의 절대 경로 대신 `OPENCV_DIR` 환경 변수를 사용합니다.

### Include Directory

```text
$(OPENCV_DIR)\include
```

### Library Directory

```text
$(OPENCV_DIR)\x64\vc15\lib
```

### DLL Directory

```text
%OPENCV_DIR%\x64\vc15\bin
```

따라서 OpenCV 설치 위치가 다른 PC에서도 `OPENCV_DIR` 환경 변수만 변경하면 동일한 Property Sheet를 사용할 수 있습니다.

---

## Repository Structure

```text
AEEDS/
│
├── .gitignore
├── OpenCV_x64.props
├── README.md
│
├── W2_colortopink/
├── W2_imageresizing/
├── W2_imagerotation/
│
└── ...
```

각 실습은 주차 및 실습 내용을 기준으로 구분하여 정리합니다.

시험 대비 정리(W2~W6 개념·핵심 코드·예상 문제·모의고사)는 [`StudyGuide/`](StudyGuide/README.md) 폴더에 있습니다.

---

## macOS (VS Code + OpenCV 5) 에서 실행하기

모든 코드는 Windows(Visual Studio + OpenCV 3.4.5)와 macOS(VS Code + Homebrew OpenCV 5)에서 모두 빌드됩니다.

1. `brew install opencv pkg-config`
2. VS Code 에서 **`AEEDS` 폴더 자체**를 연다 (`.vscode` 설정을 읽어야 함)
3. 실습 폴더 안의 아무 `.cpp` 나 열고 **`Cmd + Shift + B`** → 그 폴더의 `.cpp` 를 전부 같이 빌드하고 실행
   (W5 처럼 `main.cpp` + `cornerDetector.cpp` / `utils.cpp` 로 나뉜 폴더도 똑같이)

- 결과 창 닫기: 창을 클릭한 뒤 아무 키 / 멈추면 터미널에서 `Ctrl + C`
- 웹캠 권한: 시스템 설정 → 개인정보 보호 및 보안 → 카메라 → Visual Studio Code 허용 후 VS Code 재시작
- 아이폰(연속성 카메라)이 잡히면 `VideoCapture capture(0)` 을 `capture(1)` 로

| 폴더 | 내용 | 강의 |
|---|---|---|
| `W5_VideoCapture` | 웹캠 기본 구조 | LN5 |
| `W5_HarrisVideo` | 웹캠 + Harris (`cornerDetector.cpp` 분리) + 평균 처리 속도 | LN5 |
| `W5_ORB_Image` | ORB 매칭 (두 영상, 슬라이드 main + `utils.cpp`) | LN5 |
| `W5_ORB_Video` | ORB 매칭 (참조 영상/첫 프레임 ↔ 웹캠) | LN5 |
| `W6_OpticalFlow` | Farneback dense optical flow (슬라이드 코드) | LN6 |
| `W6_LucasKanade` | Lucas-Kanade 직접 구현 (웹캠) | LN6 |

> 웹캠 코드의 시간 측정은 Windows 전용 `QueryPerformanceCounter` 대신 `getTickCount()` 를 사용합니다. 시험 답안은 슬라이드 표기를 따르세요.

---

## Notes

Visual Studio에서 자동으로 생성되는 빌드 및 사용자 설정 파일은 `.gitignore`를 통해 Repository에서 제외합니다.

예:

```text
.vs/
Debug/
Release/
x64/
x86/
*.obj
*.pdb
*.ilk
*.user
```

OpenCV의 `.dll`, `.lib` 파일은 Repository에 직접 포함하지 않습니다.
