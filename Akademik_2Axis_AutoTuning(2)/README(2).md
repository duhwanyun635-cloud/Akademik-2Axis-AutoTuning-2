# Akademik 2-Axis AutoTuning (2)

이 저장소는 기존 `Akademik-2Axis-Stabilizer` 및 기존 AutoTuning 앱을 수정하지 않고 별도로 배포하기 위한 두 번째 프로젝트입니다.

## 포함 파일

- `app(2).py`: 가상 파도 시뮬레이션, Akademik 전체 INS 항해 데이터 자동 다운로드/재생, Hs/Tp 재구성 자료 연결, FF/Kp/Kd/상보필터 alpha 자동 튜닝
- `requirements.txt`: Streamlit 및 Python 패키지 의존성
- `.gitignore`: 대용량 데이터와 캐시가 GitHub에 올라가지 않도록 제외

## Streamlit Community Cloud 설정

| 항목 | 값 |
|---|---|
| Repository | `duhwanyun635-cloud/Akademik-2Axis-AutoTuning-2` |
| Branch | `main` |
| Main file path | `app(2).py` |
| App URL 예시 | `akademik-2axis-autotuning-2` |

`requirements.txt`는 Streamlit이 자동으로 찾도록 저장소 최상위에 그대로 두세요.

## 실행

```bash
pip install -r requirements.txt
streamlit run "app(2).py"
```

## 실제 데이터 자동 준비

앱에서 `Akademik 전체 항해 데이터`를 선택한 뒤 `전체 항해 데이터 자동 다운로드 및 등록 시작`을 누르면 ACE INS ZIP 및 재구성 Hs/Tp CSV를 공개 Zenodo 자료에서 내려받고 일별 항해 파일을 시간순으로 인덱싱합니다. 일별 CSV를 하나씩 수동 업로드할 필요가 없습니다.

대용량 원본 파일은 GitHub에 올리지 않습니다. 실행환경의 사용자 캐시 폴더에 다운로드합니다. Streamlit Cloud 실행환경이 재생성되면 다시 다운로드할 수 있습니다.

## 자동 튜닝

자동 튜닝 대상은 FF, Kp, Kd, complementary-filter alpha입니다. 서보 시간상수, 최대 각속도, 회전 한계, 가상 센서 오차는 고정조건으로 평가합니다. 모델 최적값은 실물 장치의 검증 완료값이 아닙니다. MPU-9250 등 실제 IMU와 실제 서보의 측정 데이터를 이용해 별도 검증해야 합니다.

## 데이터/모델의 한계

- ACE INS 자료는 1 Hz이며 제공기관이 원자료를 추가 보정하거나 품질검사하지 않았다고 설명합니다. 50 Hz 재생은 선형 보간이며 기록되지 않은 고주파 운동을 복원하지 않습니다.
- Hs/Tp는 Heave 기록과 RAO로 재구성한 해상상태 추정값이지, 같은 시각 부표가 직접 측정한 파랑값이 아닙니다.
- 제어 최적값은 현재 수학적 센서/서보 모델과 고정조건에 종속된 추천 후보입니다. 실제 하드웨어에 적용하기 전에 안전한 낮은 이득부터 실험 검증이 필요합니다.

## 데이터 출처

- ACE INS: https://zenodo.org/records/3733228 (CC BY 4.0)
- Reconstructed sea state Hs/Tp: https://zenodo.org/records/4541564 (CC BY 4.0)
