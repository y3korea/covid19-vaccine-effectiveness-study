#!/bin/bash

# COVID-19 Vaccine Effectiveness - High Resolution Visualization Generator
# Author: Wansuk Choi
# Date: 2025-01-15

echo "========================================"
echo "COVID-19 백신 효과성 시각화 생성기"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되지 않았습니다."
    echo "설치: brew install python3"
    exit 1
fi

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "🔧 가상환경 생성 중..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 가상환경 활성화..."
source venv/bin/activate

# Install required packages
echo "📦 필요한 패키지 설치 중..."
pip install --quiet --upgrade pip
pip install --quiet pandas numpy matplotlib seaborn scipy plotly

echo ""
echo "🎨 고해상도 이미지 생성 시작..."
echo "   해상도: 600 DPI (출판 품질)"
echo "   형식: PNG, JPG, PDF, SVG, TIFF"
echo ""

# Run the high-resolution visualization script
python3 generate_high_resolution_images.py

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 모든 시각화가 성공적으로 생성되었습니다!"
    echo ""
    echo "📁 생성된 파일 위치:"
    echo "   $(pwd)/high_res_images/"
    echo ""
    echo "📊 생성된 이미지 목록:"
    ls -lh high_res_images/*.png 2>/dev/null | awk '{print "   - "$NF": "$5}'
    echo ""
    echo "📈 총 파일 크기:"
    du -sh high_res_images/
    echo ""
    echo "💡 팁: PDF 파일은 저널 투고용, PNG는 프레젠테이션용으로 사용하세요."
else
    echo ""
    echo "❌ 시각화 생성 중 오류가 발생했습니다."
    echo "   로그를 확인해주세요."
fi

# Deactivate virtual environment
deactivate

echo ""
echo "========================================"
echo "작업 완료"
echo "========================================"