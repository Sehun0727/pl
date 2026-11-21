import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

st.set_page_config(page_title="극지식물 생육 차이 분석", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)


def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed
    return df


def make_bar_chart(df, value_col, title, y_label):
    if df is None or value_col not in df.columns or "개체번호" not in df.columns:
        return None
    summary = df.groupby("개체번호")[value_col].mean().reset_index()
    summary = summary.sort_values(value_col, ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=summary["개체번호"], y=summary[value_col], name=title, marker_color="lightblue"))
    fig.update_layout(title=title, xaxis_title="개체번호", yaxis_title=y_label, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"))
    return fig


st.sidebar.title("🌱 극지식물 생육 분석")
env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 CSV", type="csv")

env_df = None
growth_df = None
if env_file is not None:
    with st.spinner("환경 데이터를 읽고 있습니다..."):
        env_df = read_csv(env_file)
        env_df = std_cols(env_df)
if growth_file is not None:
    with st.spinner("생육 데이터를 읽고 있습니다..."):
        growth_df = read_csv(growth_file)
        growth_df = std_cols(growth_df)

school_name = "우리 학교"
if env_file is not None:
    school_name = env_file.name.split("_")[0] if "_" in env_file.name else env_file.name.rsplit(".", 1)[0]
elif growth_file is not None:
    school_name = growth_file.name.rsplit(".", 1)[0]
st.sidebar.success(f"학교: {school_name}")

tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        if env_file is None and growth_file is None:
            st.info("왼쪽 사이드바에서 환경 CSV와 생육 CSV 파일을 업로드해 주세요.")
        else:
            st.subheader("연구 배경 및 목적")
            st.write("본 대시보드는 우리 학교 극지식물의 환경 조건과 개체별 생육 결과를 분석하여, 환경 조건이 생장에 미치는 영향을 시각적으로 파악합니다.")
            
            st.subheader("데이터 요약")
            rows = []
            if env_df is not None and len(env_df) > 0:
                rows.append(["환경 측정 횟수", len(env_df)])
                items = []
                if "temperature" in env_df.columns: items.append("온도")
                if "humidity" in env_df.columns: items.append("습도")
                if "ph" in env_df.columns: items.append("pH")
                if "ec" in env_df.columns: items.append("EC")
                if items:
                    rows.append(["환경 측정 항목", ", ".join(items)])
            if growth_df is not None and len(growth_df) > 0:
                rows.append(["개체 수", len(growth_df)])
                items = []
                if "잎 수" in growth_df.columns: items.append("잎 수")
                if "지상부 길이" in growth_df.columns: items.append("지상부 길이")
                if "지하부 길이" in growth_df.columns: items.append("지하부 길이")
                if "지상부 질량" in growth_df.columns: items.append("지상부 질량")
                if "지하부 질량" in growth_df.columns: items.append("지하부 질량")
                if "생중량" in growth_df.columns: items.append("생중량")
                if items:
                    rows.append(["생육 측정 항목", ", ".join(items)])
            if rows:
                st.dataframe(pd.DataFrame(rows, columns=["항목", "값"]), width="stretch")
            else:
                st.info("업로드된 파일에서 요약할 수 있는 정보가 없습니다.")
            
            st.subheader("주요 지표")
            col1, col2, col3, col4 = st.columns(4)
            if env_df is not None and "temperature" in env_df.columns:
                temp = env_df["temperature"].dropna()
                col1.metric("평균 온도", f"{temp.mean():.1f}°C" if len(temp) > 0 else "데이터 없음")
            else:
                col1.metric("평균 온도", "온도 데이터 없음")
            if env_df is not None and "humidity" in env_df.columns:
                hum = env_df["humidity"].dropna()
                col2.metric("평균 습도", f"{hum.mean():.1f}%" if len(hum) > 0 else "데이터 없음")
            else:
                col2.metric("평균 습도", "습도 데이터 없음")
            if growth_df is not None and "생중량" in growth_df.columns:
                bio = growth_df["생중량"].dropna()
                col3.metric("평균 생중량", f"{bio.mean():.2f} g" if len(bio) > 0 else "데이터 없음")
            else:
                col3.metric("평균 생중량", "생중량 데이터 없음")
            if growth_df is not None:
                col4.metric("총 개체수", len(growth_df))
            else:
                col4.metric("총 개체수", "데이터 없음")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        st.subheader("환경 요약")
        rows = []
        for col in ["temperature", "humidity", "ph", "ec"]:
            if col in env_df.columns:
                vals = env_df[col].dropna()
                if len(vals) > 0:
                    rows.append([col, f"평균 {vals.mean():.1f}", f"최소 {vals.min():.1f}", f"최대 {vals.max():.1f}"])
        if rows:
            st.dataframe(pd.DataFrame(rows, columns=["항목", "평균", "최소", "최대"]), width="stretch")
        else:
            st.info("환경 요약 데이터를 표시할 수 없습니다.")
        
        st.subheader("날짜별 평균 시계열")
        if env_df is not None and "time" in env_df.columns:
            daily = env_df.groupby(env_df["time"].dt.date).mean(numeric_only=True).reset_index()
            daily = daily.rename(columns={"time": "date"})
            fig = go.Figure()
            if "temperature" in daily.columns:
                fig.add_trace(go.Scatter(x=daily["date"], y=daily["temperature"], mode="lines", name="온도"))
            if "humidity" in daily.columns:
                fig.add_trace(go.Scatter(x=daily["date"], y=daily["humidity"], mode="lines", name="습도"))
            if "ph" in daily.columns:
                fig.add_trace(go.Scatter(x=daily["date"], y=daily["ph"], mode="lines", name="pH"))
            if "ec" in daily.columns:
                fig.add_trace(go.Scatter(x=daily["date"], y=daily["ec"], mode="lines", name="EC"))
            if len(fig.data) > 0:
                fig.update_layout(title="일별 평균 환경 지표", xaxis_title="날짜", yaxis_title="값")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("시계열을 그릴 수 없습니다.")
        else:
            st.info("시간 열이 없어 시계열을 그릴 수 없습니다.")
        
        with st.expander("환경 데이터 원본 테이블"):
            if env_df is not None and len(env_df) > 0:
                st.dataframe(env_df, width="stretch")
            else:
                st.info("표시할 환경 데이터가 없습니다.")
            if env_file is not None:
                csv = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("환경 CSV 다운로드", csv, file_name=env_file.name.replace(".csv", "_원본.csv"), mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        st.subheader("핵심 결과")
        if growth_df is not None and "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
            bio_df = growth_df[["개체번호", "생중량"]].dropna()
            if len(bio_df) > 0:
                mean_bio = bio_df["생중량"].mean()
                max_row = bio_df.loc[bio_df["생중량"].idxmax()]
                min_row = bio_df.loc[bio_df["생중량"].idxmin()]
                c1, c2, c3 = st.columns(3)
                c1.metric("평균 생중량", f"{mean_bio:.2f} g")
                c2.metric("최대 개체", f"{max_row['개체번호']} ({max_row['생중량']:.2f} g)")
                c3.metric("최소 개체", f"{min_row['개체번호']} ({min_row['생중량']:.2f} g)")
        
        st.subheader("개체별 생육 비교")
        if growth_df is not None:
            col1, col2 = st.columns(2)
            for col, title, label in [("생중량", "평균 생중량", "평균 생중량 (g)"), ("잎 수", "평균 잎 수", "평균 잎 수 (장)")]:
                fig = make_bar_chart(growth_df, col, title, label)
                if fig is not None:
                    if col == "생중량":
                        col1.plotly_chart(fig, width="stretch")
                    else:
                        col2.plotly_chart(fig, width="stretch")
            col3, col4 = st.columns(2)
            for col, title, label in [("지상부 길이", "평균 지상부 길이", "평균 지상부 길이 (mm)"), ("지하부 길이", "평균 지하부 길이", "평균 지하부 길이 (mm)")]:
                fig = make_bar_chart(growth_df, col, title, label)
                if fig is not None:
                    if col == "지상부 길이":
                        col3.plotly_chart(fig, width="stretch")
                    else:
                        col4.plotly_chart(fig, width="stretch")
        else:
            st.info("생육 데이터를 업로드해 주세요.")
        
        st.subheader("생중량 분포")
        if growth_df is not None and "생중량" in growth_df.columns:
            bio_vals = growth_df["생중량"].dropna()
            if len(bio_vals) >= 2:
                fig = go.Figure()
                fig.add_trace(go.Box(y=bio_vals, name="생중량", marker_color="lightblue"))
                fig.update_layout(title="생중량 분포")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("생중량 분포를 그릴 수 없습니다.")
        else:
            st.info("생중량 데이터가 없습니다.")
        
        st.subheader("상관관계 분석")
        if growth_df is not None and "잎 수" in growth_df.columns and "지상부 길이" in growth_df.columns and "생중량" in growth_df.columns:
            merged1 = growth_df[["잎 수", "생중량"]].dropna()
            merged2 = growth_df[["지상부 길이", "생중량"]].dropna()
            if len(merged1) >= 3 and len(merged2) >= 3:
                fig = make_subplots(rows=1, cols=2, subplot_titles=("잎 수 vs 생중량", "지상부 길이 vs 생중량"))
                x1 = merged1["잎 수"]
                y1 = merged1["생중량"]
                fig.add_trace(go.Scatter(x=x1, y=y1, mode="markers", name="데이터", marker=dict(color="lightblue")), row=1, col=1)
                z1 = np.polyfit(x1, y1, 1)
                p1 = np.poly1d(z1)
                fig.add_trace(go.Scatter(x=x1, y=p1(x1), mode="lines", name="추세선", line=dict(dash="dash", color="red")), row=1, col=1)
                x2 = merged2["지상부 길이"]
                y2 = merged2["생중량"]
                fig.add_trace(go.Scatter(x=x2, y=y2, mode="markers", name="데이터", marker=dict(color="lightgreen")), row=1, col=2)
                z2 = np.polyfit(x2, y2, 1)
                p2 = np.poly1d(z2)
                fig.add_trace(go.Scatter(x=x2, y=p2(x2), mode="lines", name="추세선", line=dict(dash="dash", color="red")), row=1, col=2)
                fig.update_layout(title="상관관계 분석", height=500)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("상관관계를 계산할 데이터가 부족합니다.")
        else:
            st.info("상관관계 분석에 필요한 열이 없습니다.")
        
        st.subheader("생중량 상위 25% vs 하위 25%")
        if growth_df is not None and "생중량" in growth_df.columns and "개체번호" in growth_df.columns:
            bio_df2 = growth_df[["개체번호", "생중량"]].dropna()
            if len(bio_df2) >= 4:
                bio_df2 = bio_df2.sort_values("생중량", ascending=False)
                n = len(bio_df2)
                top_n = max(1, n // 4)
                bottom_n = max(1, n // 4)
                top_mean = bio_df2.iloc[:top_n]["생중량"].mean()
                bottom_mean = bio_df2.iloc[-bottom_n:]["생중량"].mean()
                fig = go.Figure()
                fig.add_trace(go.Bar(x=["상위 25%", "하위 25%"], y=[top_mean, bottom_mean], marker_color=["lightgreen", "salmon"]))
                fig.update_layout(title="상위 25% vs 하위 25% 평균 생중량 비교", xaxis_title="그룹", yaxis_title="평균 생중량 (g)")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("비교할 데이터가 충분하지 않습니다.")
        
        with st.expander("생육 데이터 원본 테이블"):
            if growth_df is not None and len(growth_df) > 0:
                st.dataframe(growth_df, width="stretch")
            else:
                st.info("표시할 생육 데이터가 없습니다.")
            if growth_file is not None:
                st.download_button("생육 CSV 다운로드", growth_df.to_csv(index=False).encode("utf-8-sig"), file_name=growth_file.name.replace(".csv", "_원본.csv"), mime="text/csv")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
