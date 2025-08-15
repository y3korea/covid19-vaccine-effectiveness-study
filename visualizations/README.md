# COVID-19 백신 효과성 시각화 가이드

## 📊 개요
이 디렉토리는 COVID-19 백신 효과성 메타분석 연구의 고품질 시각화 자료를 포함합니다.

## 🖼️ 이미지 사양

### 해상도
- **기본 DPI**: 600 (저널 출판 품질)
- **대체 DPI**: 300 (웹/프레젠테이션용)
- **벡터 형식**: SVG (무한 확대 가능)

### 파일 형식

| 형식 | 용도 | 특징 |
|------|------|------|
| **PNG** | 웹, 프레젠테이션 | 무손실 압축, 투명 배경 지원 |
| **JPG** | 이메일, 빠른 공유 | 손실 압축, 작은 파일 크기 |
| **PDF** | 저널 투고, 인쇄 | 벡터 그래픽, 편집 가능 |
| **SVG** | 편집, 확대/축소 | 벡터 형식, 무한 확대 |
| **TIFF** | 아카이브, 출판사 요구 | 무압축, 최고 품질 |

## 📁 파일 구조

```
visualizations/
├── generate_high_resolution_images.py   # 고해상도 이미지 생성 스크립트
├── comprehensive_visualizations.py      # 종합 시각화 스크립트
├── create_interactive_dashboard.html    # 인터랙티브 대시보드
├── run_visualization.sh                 # 실행 스크립트
├── README.md                           # 이 파일
└── high_res_images/                    # 생성된 고해상도 이미지
    ├── forest_plot.png                 # Forest plot (600 DPI)
    ├── forest_plot.jpg                 # Forest plot (JPEG)
    ├── forest_plot.pdf                 # Forest plot (PDF)
    ├── forest_plot.svg                 # Forest plot (SVG)
    ├── forest_plot.tiff                # Forest plot (TIFF)
    ├── funnel_plot.*                   # Funnel plot (모든 형식)
    ├── variant_heatmap.*               # 변이 효과성 히트맵
    ├── waning_immunity.*               # 면역 감소 그래프
    ├── age_effectiveness.*             # 연령별 효과성
    ├── summary_infographic.*           # 요약 인포그래픽
    └── image_generation_report.txt     # 생성 리포트
```

## 🚀 실행 방법

### 방법 1: Shell 스크립트 사용 (권장)
```bash
chmod +x run_visualization.sh
./run_visualization.sh
```

### 방법 2: Python 직접 실행
```bash
# 패키지 설치
pip install pandas numpy matplotlib seaborn scipy plotly

# 고해상도 이미지 생성
python generate_high_resolution_images.py
```

### 방법 3: 인터랙티브 대시보드
```bash
# 브라우저에서 직접 열기
open create_interactive_dashboard.html
```

## 📊 생성되는 시각화 목록

### 1. Forest Plot
- 35개 연구의 백신 효과성 메타분석
- 백신 종류별 색상 구분
- 95% 신뢰구간 표시
- 가중치 비례 마커 크기

### 2. Funnel Plot
- 출판 편향 평가
- Trim-and-fill 조정 전후 비교
- Egger's test 결과

### 3. Variant Heatmap
- 9개 변이 × 8개 백신 매트릭스
- 색상 코드 효과성 표시
- 20-100% 범위

### 4. Waning Immunity
- 시간에 따른 효과 감소
- 백신 종류별 비교
- 임상 결과별 분석

### 5. Age Effectiveness
- 7개 연령 그룹 분석
- 감염 예방 vs 중증 예방
- 백신 종류별 비교

### 6. Summary Infographic
- 핵심 지표 요약
- 전체 효과성: 82.3%
- 8.4M 참여자
- 지역별 분포

## 🎨 색상 팔레트

```python
colors = {
    'mRNA': '#2E86AB',        # 파란색
    'Vector': '#A23B72',      # 보라색
    'Inactivated': '#F18F01', # 주황색
    'Overall': '#C73E1D'      # 빨간색
}
```

## 📝 인용 방법

```bibtex
@article{choi2025covid,
  title={COVID-19 Vaccine Effectiveness: A Systematic Review and Meta-Analysis},
  author={Choi, Wansuk},
  journal={Medical Research Repository},
  year={2025},
  month={January},
  doi={10.xxxxx/xxxxx}
}
```

## 🔧 문제 해결

### 메모리 부족 오류
```python
# matplotlib 백엔드 변경
import matplotlib
matplotlib.use('Agg')
```

### 폰트 관련 오류
```bash
# 폰트 캐시 재생성
rm -rf ~/.matplotlib/fontlist-*.json
python -c "import matplotlib.pyplot as plt"
```

### DPI 조정
```python
# 파일 크기가 너무 큰 경우
viz = HighResolutionVisualizations()
viz.dpi = 300  # 600에서 300으로 조정
```

## 📞 연락처
- 작성자: Wansuk Choi
- 이메일: y3korea@gmail.com
- GitHub: @y3korea

## 📄 라이선스
MIT License - 자유롭게 사용, 수정, 배포 가능

---
*마지막 업데이트: 2025-01-15*