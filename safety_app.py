import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time  # 최초 애니메이션 트리거용 라이브러리

# 1. 웹 페이지 레이아웃 및 테마 스타일 설정
st.set_page_config(
    page_title="스마트 팩토리 설비 안전 진단 대시보드",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (파이썬 3.14 호환 및 폰트 스타일 최적화)
st.markdown("""
    <style>
    .main-title { font-size:40px; font-weight:bold; color:#1E3A8A; margin-bottom:5px; }
    .sub-title { font-size:18px; color:#4B5563; margin-bottom:20px; }
    .card-title { font-size:20px; font-weight:600; color:#1F2937; margin-bottom:10px; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# 2. 헤더 구역 디자인
st.markdown('<p class="main-title">⚙️ AI 기반 기계 설비 안전 예측 시스템</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">"AI로 널리 산업 현장을 안전하게 이롭게 하다" | SMOTE 기반 고정밀 분류 모델 연동</p>', unsafe_allow_html=True)
st.divider()

# 3. AI 모델 및 스케일러 파일 불러오기
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('predictive_maintenance_model.pkl')
        scaler = joblib.load('factory_scaler.pkl')
        return model, scaler
    except:
        return None, None

model, scaler = load_resources()

if model is None or scaler is None:
    st.error("⚠️ 'predictive_maintenance_model.pkl' 또는 'factory_scaler.pkl' 파일이 소스코드와 같은 폴더에 있는지 확인해 주세요.")
    st.stop()

# 4. 사이드바 입력 폼 디자인 (편의성 개선: 가이드 툴팁 장착)
st.sidebar.markdown("### 🔌 실시간 센서 제어판")
st.sidebar.write("현재 가동 중인 설비의 센서 값을 조절하세요.")

air_temp = st.sidebar.slider("🌡️ 공기 온도 [K]", min_value=290.0, max_value=350.0, value=300.0, step=0.1, help="공장 내부의 대기 온도입니다.")
proc_temp = st.sidebar.slider("🔥 공정 온도 [K]", min_value=290.0, max_value=350.0, value=310.0, step=0.1, help="가동 중인 설비 부품의 자체 온도입니다.")
rpm = st.sidebar.number_input("🔄 회전 속도 [RPM]", min_value=1000, max_value=3000, value=1500, step=50, help="모터의 분당 회전 속도입니다.")
torque = st.sidebar.slider("⚡ 토크 부하 [Nm]", min_value=3.0, max_value=80.0, value=35.0, step=0.5, help="기계가 받고 있는 회전 부하입니다.")
tool_wear = st.sidebar.slider("⏳ 공구 마모 시간 [min]", min_value=0, max_value=250, value=10, step=1, help="현재 부품을 교체 없이 사용한 시간입니다.")

# 5. 기계공학 파생변수 실시간 연산
temp_diff = proc_temp - air_temp
mechanical_power = torque * (rpm * 2 * np.pi / 60)
wear_torque_ratio = torque * tool_wear

# 입력 데이터를 데이터프레임으로 바인딩
input_df = pd.DataFrame([{
    '공기온도': air_temp, '공정온도': proc_temp, '회전속도': rpm, '토크': torque, '공구마모시간': tool_wear,
    '온도차이': temp_diff, '기계동력': mechanical_power, '마모대비토크': wear_torque_ratio
}])

# 6. 메인 화면 상단 지표(Metric) 레이아웃 배치
m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="🔺 실시간 내외부 온도 차이", value=f"{temp_diff:.1f} K", delta=f"{'과열주의' if temp_diff > 11 else '정상'}")
with m_col2:
    st.metric(label="⚙️ 연산된 기계 동력(Power)", value=f"{mechanical_power/1000:.2f} kW")
with m_col3:
    st.metric(label="📊 누적 마모 부하량", value=f"{wear_torque_ratio:.1f}")

st.write("")

# 하단 대시보드 분할 [좌측 4 : 우측 5]
layout_col1, layout_col2 = st.columns([4, 5])

with layout_col1:
    st.markdown('<p class="card-title">📝 센서 수치 분석 데이터 테이블</p>', unsafe_allow_html=True)
    display_df = input_df.T.rename(columns={0: "현재 모니터링 값"})
    st.dataframe(display_df, use_container_width=True, height=315)

with layout_col2:
    st.markdown('<p class="card-title">🚨 AI 실시간 위험도 진단 결과</p>', unsafe_allow_html=True)
    
    # AI 스케일링 변환 및 실시간 추론 진행
    features = ['공기온도', '공정온도', '회전속도', '토크', '공구마모시간', '온도차이', '기계동력', '마모대비토크']
    input_scaled = scaler.transform(input_df[features])
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100  # 위험 확률 (%)

    # --- 7. [최초 로딩 애니메이션 제어 트리거] ---
    if "first_load" not in st.session_state:
        st.session_state.first_load = True
        start_val = 0.0
    else:
        start_val = prob

    if prob < 50:
        bar_color = '#111827'  # 안전 (다크 차콜)
    elif prob < 80:
        bar_color = '#1E3A8A'  # 주의 (딥 블루)
    else:
        bar_color = '#7F1D1D'  # 위험 (딥 레드)

    # 선명한 신호등 3색 베이스 구성
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = start_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        number = {'suffix': "%", 'font': {'size': 26, 'weight': 'bold', 'color': '#1F2937'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1.5, 'tickcolor': "#4B5563"},
            'bar': {'color': bar_color, 'thickness': 0.55},
            'bgcolor': "#F3F4F6",
            'borderwidth': 1,
            'bordercolor': "#D1D5DB",
            'steps': [
                {'range': [0, 50], 'color': '#10B981'},   # 선명한 초록 (안전)
                {'range': [50, 80], 'color': '#F59E0B'},  # 선명한 노랑 (주의)
                {'range': [80, 100], 'color': '#EF4444'}  # 선명한 빨강 (위험)
            ],
        }
    ))
    
    fig.update_layout(
        height=220, 
        margin=dict(l=30, r=30, t=20, b=20),
        transition={'duration': 700, 'easing': 'cubic-in-out'}
    )
    
    # 중복 키 에러 방지 처리 완료된 플레이스홀더 렌더링
    gauge_placeholder = st.empty()
    gauge_placeholder.plotly_chart(fig, use_container_width=True, key="factory_base_gauge")

    if st.session_state.first_load:
        time.sleep(0.1)
        fig.update_traces(value=prob)
        gauge_placeholder.plotly_chart(fig, use_container_width=True, key="factory_active_gauge")
        st.session_state.first_load = False
    # ----------------------------------------------------

    # ====================================================================
    # 🎛️ 8. AI 원인 상세 진단 및 현장 조치 매뉴얼 매핑 (2, 3번 기능)
    # ====================================================================
    st.write("")
    
    # [정상 / 안전 판정]
    if prediction == 0 and prob < 50:
        st.success(f"🟢 **설비 상태: [ 정상 / 안전 ]** \n\n현재 기계가 매우 안정적으로 작동하고 있습니다. (위험 확률: {prob:.1f}%)")
        
    # [주의 및 위험 판정 시 다이내믹 원인 추적 분석 블록 발동]
    else:
        fault_reasons = []
        action_steps = []
        
        # 도메인 지식 기반 규칙 매핑
        if temp_diff > 11.0:
            fault_reasons.append("• **[발열 이상]** 내외부 온도 차이(ΔT)가 과도하게 발생 중입니다. (베어링 윤활 부족 및 내부 마찰 의심)")
            action_steps.append("1. 냉각수 주입 라인 및 냉각 팬 가동 상태를 점검하십시오.")
            action_steps.append("2. 설비 주요 구동부에 그리스(윤활유)를 도포하십시오.")
            
        if torque > 55.0:
            fault_reasons.append("• **[과토크 부하]** 토크 수치가 임계치(55 Nm)를 초과했습니다. (소재 결착, 가공 부하 급증 또는 기어박스 이상 의심)")
            action_steps.append("1. 가공 중인 원자재의 공급 속도(Feed Rate)를 하향 조절하십시오.")
            action_steps.append("2. 척(Chuck)이나 스핀들에 이물질이 끼었는지 육안 점검하십시오.")
            
        if tool_wear > 180:
            fault_reasons.append("• **[공구 마모 임박]** 누적 마모 시간이 유효 한계 수치(180분)를 넘어섰습니다. (칩 배출 불량 및 정밀도 저하 발생 중)")
            action_steps.append("1. 현재 공정 단계가 종료되는 즉시 부품(Insert Tip 등)을 교체하십시오.")
            action_steps.append("2. 절삭유 분사 압력을 높여 마찰열을 일시적으로 낮추십시오.")
            
        if rpm > 2500 and torque > 45.0:
            fault_reasons.append("• **[과부하 밸런스 붕괴]** 고속 회전 중에 고토크가 동시에 걸려 기계 동력이 한계치에 도달했습니다.")
            action_steps.append("1. 주축 회전 속도(RPM)를 제어판에서 15% 감속 조치하십시오.")

        # 복합 원인 예외 처리
        if not fault_reasons:
            fault_reasons.append("• **[복합 패턴 고장]** 단일 센서 임계치는 정상 범위이나, 복합 센서 수치 조합이 고장 전조 패턴과 일치합니다.")
            action_steps.append("1. 설비 운전 모드를 '수동(Manual)'으로 전환하고 정밀 정비 진단을 대기하십시오.")

        # 상태별 상단 카드 노출
        if prob < 80:
            st.warning(f"🟡 **설비 상태: [ 주의 요구 ]** 누적 부하로 인해 주의가 필요합니다. (위험 확률: {prob:.1f}%)")
        else:
            st.error(f"🔴 **설비 상태: [ 위험 / 고장 임박 ]** 심각한 이상 징후가 감지되었습니다. 즉시 가동 중단을 고려하십시오! (위험 확률: {prob:.1f}%)")
            
        # 상세 진단서 리포트 출력 구역 (접이식 아코디언 컴포넌트)
        with st.expander("🔍 AI 정밀 원인 진단 및 현장 조치 매뉴얼 보기", expanded=True):
            st.markdown("#### 📊 AI가 분석한 주요 이상 원인")
            for reason in fault_reasons:
                st.write(reason)
                
            st.markdown("---")
            
            st.markdown("#### 🛠 ... 현장 작업자 실시간 대응 지침 (SOP)")
            for step in action_steps:
                st.write(step)
                
            st.markdown("<p style='font-size:12px; color:#9CA3AF; margin-top:10px;'>* 본 지침은 정밀 정비 전 설비 파손을 막기 위한 AI 추천 조치 매뉴얼입니다.</p>", unsafe_allow_html=True)
