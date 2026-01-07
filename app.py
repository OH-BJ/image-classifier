import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

# 1. 페이지 설정
st.set_page_config(
    page_title="AI 이미지 분류기",
    page_icon="✨",
    layout="wide"
)

# 2. 제목
st.title("AI 만능 이미지 분류기")
st.markdown("""
여러 장의 이미지를 한 번에 분석하거나, 카메라로 찍어서 바로 확인해보세요!  
(Model: Google ViT-Base / ImageNet-1k)
""")

# 3. 모델 로딩 (캐싱)
@st.cache_resource
def load_model():
    classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    return classifier

with st.spinner("AI 모델을 불러오는 중입니다..."):
    classifier = load_model()

# 공통 분석 함수
def analyze_image(image_obj):
    # 2단 컬럼 레이아웃
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(image_obj, caption="Input Image", width=350)
    
    with col2:
        # 모델 추론 (빠른 처리를 위해 버튼 없이 바로 실행하도록 변경)
        results = classifier(image_obj)
        top_result = results[0]
        label = top_result['label']
        score = top_result['score']
        
        # 이모지 매핑
        emoji = "🤖"
        if "dog" in label or "retriever" in label or "terrier" in label:
            emoji = "🐶"
        elif "cat" in label or "tabby" in label:
            emoji = "🐱"
        elif "car" in label or "vehicle" in label:
            emoji = "🚗"
        elif "coffee" in label or "cup" in label:
            emoji = "☕"
        elif "food" in label or "burger" in label or "pizza" in label:
            emoji = "🍔"
        
        st.success(f"{emoji} **[{label}]** ({score*100:.1f}%)")
        
        # 차트 시각화
        df = pd.DataFrame(results)
        df['score'] = df['score'] * 100 
        
        st.bar_chart(
            df.set_index('label')['score'],
            color=["#FF4B4B"],
            height=200 # 차트 높이 조절
        )

# 4. 탭 구성
tab1, tab2 = st.tabs(["📁 파일 업로드 (여러 장 가능)", "📸 카메라 촬영"])

# 탭 1: 파일 업로드
with tab1:
    uploaded_files = st.file_uploader(
        "이미지 파일을 선택하세요 (여러 개 선택 가능)", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True 
    )
    
    if uploaded_files:
        st.write(f"총 {len(uploaded_files)}장의 이미지를 분석합니다.")
        
        # 버튼 하나로 일괄 분석 시작
        if st.button("전체 분석 시작", type="primary"):
            # 반복문(for)으로 파일 하나하나 꺼내서 분석
            for file in uploaded_files:
                st.divider() # 구분선
                image = Image.open(file)
                analyze_image(image) # 위에서 만든 함수 호출

# 탭 2: 카메라 촬영
with tab2:
    camera_file = st.camera_input("직접 사진을 찍어보세요!")
    if camera_file:
        st.divider()
        image = Image.open(camera_file)
        analyze_image(image) # 함수 재사용