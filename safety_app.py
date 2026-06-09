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

# 4. 사이드바 입력 폼 디자인
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

# 7. 하단 대시보드 좌우 분할 비율 조정 [좌측 5 : 우측 5 반반 황금 분배]
layout_col1, layout_col2 = st.columns([5, 5])

with layout_col1:
    st.markdown('<p class="card-title">📝 센서 수치 분석 데이터 테이블</p>', unsafe_allow_html=True)
    display_df = input_df.T.rename(columns={0: "현재 모니터링 값"})
    # 테이블 높이를 정밀하게 확장하여 우측 레이아웃과의 대칭 균형을 맞춥니다.
    st.dataframe(display_df, use_container_width=True, height=450)

with layout_col2:
    st.markdown('<p class="card-title">🚨 AI 실시간 위험도 진단 결과</p>', unsafe_allow_html=True)
    
    # AI 추론 진행
    features = ['공기온도', '공정온도', '회전속도', '토크', '공구마모시간', '온도차이', '기계동력', '마모대비토크']
    input_scaled = scaler.transform(input_df[features])
    
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1] * 100

    # 최초 애니메이션 제어 트리거
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

    # 3색 신호등 게이지
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
                {'range': [0, 50], 'color': '#10B981'},   
                {'range': [50, 80], 'color': '#F59E0B'},  
                {'range': [80, 100], 'color': '#EF4444'}  
            ],
        }
    ))
    
    # 한쪽으로 치우쳐 보이지 않도록 게이지 크기 조절 및 상하 여백 정렬
    fig.update_layout(
        height=200, 
        margin=dict(l=40, r=40, t=10, b=10)
    )
    
    gauge_placeholder = st.empty()
    gauge_placeholder.plotly_chart(fig, use_container_width=True, key="factory_base_gauge")

    if st.session_state.first_load:
        time.sleep(0.1)
        fig.update_traces(value=prob)
        gauge_placeholder.plotly_chart(fig, use_container_width=True, key="factory_active_gauge")
        st.session_state.first_load = False

    # ====================================================================
    # 🎛️ 8. AI 원인 상세 진단 및 현장 조치 매뉴얼 (2, 3번) - 위치 밸런스 조정
    # ====================================================================
    fault_reasons = []
    action_steps = []

    # 🟡 [주의 상태] 위험 확률 50% ~ 80% 미만
    if 50 <= prob < 80:
        st.warning(f"🟡 **설비 상태: [ 주의 요구 ]** 누적 부하로 인해 주의가 필요합니다. (위험 확률: {prob:.1f}%)")
        if temp_diff > 9.5:
            fault_reasons.append("• **[미세 발열 발생]** 부품 간의 마찰열이 조금씩 축적되고 있습니다. (ΔT: {temp_diff:.1f} K)")
            action_steps.append("- 공장 공조 장치를 확인하고 설비 주변 환기 상태를 점검해 주세요.")
        if torque > 45.0:
            fault_reasons.append("• **[토크 부하 상승]** 기계 구동부에 평소보다 다소 높은 저항 부하가 걸리고 있습니다.")
            action_steps.append("- 주입되는 원자재의 속도 또는 공급 밸런스가 치우쳐져 있는지 점검해 주세요.")
        if tool_wear > 120:
            fault_reasons.append("• **[공구 노후화 진행]** 현재 공구의 마모 시간이 절반 이상 경과하였습니다. ({tool_wear}분 가동)")
            action_steps.append("- 다음 정기 점검 교체 대상 리스트에 본 설비를 등록해 주세요.")
        if not fault_reasons:
            fault_reasons.append("• **[설비 열화 전조 현상]** 센서들의 개별 수치는 정상이나, 복합적인 경미한 열화 수치가 시작되었습니다.")
            action_steps.append("- 회전 속도(RPM)를 현재 수치보다 5~10% 줄여 운전하는 것을 권장합니다.")

    # 🔴 [위험 상태] 위험 확률 80% 이상
    elif prob >= 80:
        st.error(f"🔴 **설비 상태: [ 위험 / 고장 임박 ]** 심각한 이상 징후가 감지되었습니다. 즉시 조치가 필요합니다! (위험 확률: {prob:.1f}%)")
        if temp_diff > 11.0:
            fault_reasons.append("• 🚨 **[임계 발열 초과]** 온도 차이가 한계를 넘었습니다. 베어링 마찰 손상 또는 윤활 부족 유력.")
            action_steps.append("1. 즉시 냉각 팬의 정상 작동 여부를 확인하고 가동을 잠시 중단하십시오.")
            action_steps.append("2. 주요 회전 부위에 긴급 윤활 그리스(Grease) 주입공정을 지시하십시오.")
        if torque > 55.0:
            fault_reasons.append("• 🚨 **[과토크 락 경고]** 기계가 견딜 수 있는 토크 한계선을 침범했습니다. 이물질 결착 의심.")
            action_steps.append("1. 메인 가공 장치의 원자재 투입 라인 스핀들 속도를 비상 감속하십시오.")
            action_steps.append("2. 가공 피드 라인 내부의 물리적 칩(Chip) 결착 여부를 육안 검사하십시오.")
        if tool_wear > 180:
            fault_reasons.append("• 🚨 **[공구 마모 한계 도달]** 공구 날의 유효 수명이 다해 파손 위험이 극도로 높습니다.")
            action_steps.append("1. 현재 작업 사이클이 끝나는 즉시 장비를 멈추고 새 공구 부품으로 정비하십시오.")
        if not fault_reasons:
            fault_reasons.append("• 🚨 **[복합 임계치 고장 패턴]** 특정 단일 수치보다는 가동 파워 대비 급격한 부하 밸런스 붕괴 현상입니다.")
            action_steps.append("1. 설비 가동 모드를 안전 모드(Manual)로 강제 전환 후 유지보수 팀에 전파하십시오.")

    # 🟢 [정상 상태] 위험 확률 50% 미만
    else:
        st.success(f"🟢 **설비 상태: [ 정상 / 안전 ]** \n\n현재 기계가 매우 안정적으로 작동하고 있습니다. (위험 확률: {prob:.1f}%)")
        # 정상 상태일 때도 우측 영역이 텅 비어 보이지 않도록 부드러운 가이드를 채워 균형을 맞춥니다.
        fault_reasons.append("• **[상태 진단]** 모든 센서의 실시간 인입 수치가 정상 가이드라인 범주 내에 있습니다.")
        action_steps.append("- 특이사항 없음: 현재 설정된 작업 부하(SOP)를 그대로 유지하며 모니터링하십시오.")

    # 👁️ 상세 분석 창 배치 (항상 노출하여 테이블과 높이를 맞춤)
    with st.expander("🔍 AI 정밀 원인 진단 및 현장 조치 매뉴얼", expanded=True):
        st.markdown("#### 📊 AI가 분석한 주요 이상 원인")
        for reason in fault_reasons:
            st.write(reason)
            
        st.markdown("---")
        
        st.markdown("#### 🛠️ 현장 작업자 실시간 대응 지침 (SOP)")
        for step in action_steps:
            st.write(step)
