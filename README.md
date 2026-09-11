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
