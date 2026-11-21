import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------- 페이지 설정 ----------
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

# ---------- 한글 폰트 CSS ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- 캐시 데이터 로딩 함수 ----------
@st.cache_data
def load_file(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


def read_csv(file):
    return load_file(file)


# ---------- 열 이름 표준화 ----------
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

    # 숫자 열 정리
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c]
                .astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num

    # 생중량 없으면 지상부+지하부 질량으로 계산
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]

    # 시간 파싱
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(
                parts[0], format="%Y%m%d"
            ) + pd.to_timedelta(parts[1].astype(int), unit="h")
        df["time"] = parsed

    return df


# ---------- 학교명 추출 ----------
def school_name(filename):
    if filename is None or filename == "":
        return "업로드한 파일"
    base = os.path.splitext(filename)[0]
    # 첫 번째 '_' 또는 '-' 앞까지가 학교명이라고 가정
    name = base.split("_")[0].split("-")[0]
    return name


# ---------- 그래프 폰트 레이아웃 ----------
def korean_layout(fig, title=None):
    fig.update_layout(
        font=dict(
            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
        ),
        title=dict(
            text=title, font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif")
        )
        if title
        else {},
    )
    return fig


# ---------- 하향선(plotly express 대안)으로 추세선 추가 ----------
def add_trendline(fig, x, y, color="rgba(255, 0, 0, 0.8)"):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return
    slope, intercept = np.polyfit(df["x"], df["y"], 1)
    x_line = np.array([df["x"].min(), df["x"].max()])
    y_line = slope * x_line + intercept
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            line=dict(color=color, dash="dash", width=2),
            name="추세선",
            hoverinfo="skip",
        )
    )


# ---------- 상관계수 계산 ----------
def corr_text(x, y):
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df) < 2:
        return "데이터 부족"
    r = df["x"].corr(df["y"])
    return f"상관계수 r = {r:.2f}"


# ---------- 메인 ----------
def main():
    st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")
    st.markdown(
        "CSV 두 개(환경 데이터·생육 결과 데이터)를 업로드하면, 학교마다 다른 열 이름도 자동으로 맞춰서 분석합니다."
    )

    # ===== 사이드바 =====
    st.sidebar.header("📂 데이터 업로드")
    env_file = st.sidebar.file_uploader("환경 CSV", type="csv")
    growth_file = st.sidebar.file_uploader("생육 결과 CSV", type="csv")

    env_df = pd.DataFrame()
    growth_df = pd.DataFrame()
    env_loaded = env_file is not None
    growth_loaded = growth_file is not None

    if env_loaded:
        env_df = read_csv(env_file)
        if env_df.empty:
            st.sidebar.warning("⚠️ 환경 CSV를 읽지 못했습니다.")
            env_loaded = False
        else:
            env_df = std_cols(env_df)

    if growth_loaded:
        growth_df = read_csv(growth_file)
        if growth_df.empty:
            st.sidebar.warning("⚠️ 생육 결과 CSV를 읽지 못했습니다.")
            growth_loaded = False
        else:
            growth_df = std_cols(growth_df)

    school_env = school_name(env_file.name if env_loaded else None)
    school_growth = school_name(growth_file.name if growth_loaded else None)
    school = school_env if env_loaded else (school_growth if growth_loaded else "업로드한 파일")

    st.sidebar.info(f"🏫 분석 중인 학교: **{school}**")

    st.sidebar.header("📏 생육 측정 항목")
    measure_options = []
    if growth_loaded:
        for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]:
            if c in growth_df.columns:
                measure_options.append(c)
    if not measure_options:
        measure_options = []
    selected_measures = st.sidebar.multiselect(
        "비교할 항목 선택",
        options=measure_options,
        default=measure_options[:1] if measure_options else [],
    )
    if not selected_measures:
        selected_measures = measure_options

    # ===== 탭 =====
    tab1, tab2, tab3 = st.tabs(
        ["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"]
    )

    # ---------- Tab 1: 실험 개요 ----------
    with tab1:
        try:
            st.header("📖 실험 개요")

            # 연구 배경 및 목적
            st.subheader("연구 배경 및 목적")
            st.markdown(
                """
- 극지식물(툰드라·高山 식물)은 낮은 온도, 강한 바람, 짧은 생육기 같은 척박한 환경에 적응한 식물입니다.
- 같은 종이라도 개체마다 크기·잎 수·생체량이 다르게 자라는데, 이 차이를 정량적으로 비교하면
  **어떤 개체가 환경에 잘 적응했는지**, **개체 간 차이를 만드는 요인은 무엇인지** 알 수 있습니다.
- 이 대시보드는 여러분이 측정한 극지식물 개체별 생육 데이터를
  **평균·표준편차·변동계수(CV)·상관관계** 같은 기초 통계로 보여 줍니다.

> **기초 통계 용어**
> - **평균**: 값들을 모두 더해 개수로 나눈 ‘대표값’.
> - **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지를 나타내는 값. 클수록 개체별 차이가 큽니다.
> - **변동계수(CV)**: 표준편차를 평균으로 나눈 값(백분율). 단위가 다른 항목끼리도 흩어진 정도를 비교할 수 있습니다.
> - **상관계수(r)**: 두 변수가 함께 어떻게 변하는지 나타내는 값. -1~1 사이이며, 1에 가까울수록 함께 증가, -1에 가까울수록 함께 감소합니다.
"""
            )

            # 데이터 요약 표
            st.subheader("데이터 요약 표")
            summary_rows = []

            # 개체 수
            n_growth = len(growth_df) if growth_loaded else 0
            if n_growth > 0:
                summary_rows.append(
                    {"항목": "총 개체 수", "값": f"{n_growth}개체"}
                )
            else:
                summary_rows.append({"항목": "총 개체 수", "값": "데이터 없음"})

            # 측정 기간
            if env_loaded and "time" in env_df.columns and env_df["time"].notna().any():
                t_min = env_df["time"].min()
                t_max = env_df["time"].max()
                summary_rows.append(
                    {
                        "항목": "환경 측정 기간",
                        "값": f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}",
                    }
                )
            else:
                summary_rows.append({"항목": "환경 측정 기간", "값": "데이터 없음"})

            # 환경 측정 항목
            env_measures = []
            for c in ["temperature", "humidity", "ph", "ec"]:
                if c in env_df.columns:
                    env_measures.append(c)
            summary_rows.append(
                {
                    "항목": "환경 측정 항목",
                    "값": ", ".join(env_measures) if env_measures else "없음",
                }
            )

            # 생육 측정 항목
            growth_measures = []
            if growth_loaded:
                for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량", "지상부 질량", "지하부 질량"]:
                    if c in growth_df.columns:
                        growth_measures.append(c)
            summary_rows.append(
                {
                    "항목": "생육 측정 항목",
                    "값": ", ".join(growth_measures) if growth_measures else "없음",
                }
            )

            if summary_rows:
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("📌 아직 데이터가 없습니다. 사이드바에서 CSV를 업로드해 주세요.")

            # 주요 지표 카드 4개
            st.subheader("주요 지표 카드")
            c1, c2, c3, c4 = st.columns(4)

            # 총 개체수
            with c1:
                if n_growth > 0:
                    st.metric("총 개체수", f"{n_growth}개체")
                else:
                    st.metric("총 개체수", "데이터 없음")

            # 평균 온도
            with c2:
                if env_loaded and "temperature" in env_df.columns:
                    t_vals = env_df["temperature"].dropna()
                    if len(t_vals) >= 2:
                        st.metric(
                            "평균 온도",
                            f"{t_vals.mean():.1f}℃",
                        )
                    elif len(t_vals) == 1:
                        st.metric("평균 온도", f"{t_vals.iloc[0]:.1f}℃")
                    else:
                        st.metric("평균 온도", "데이터 없음")
                else:
                    st.metric("평균 온도", "데이터 없음")

            # 평균 습도
            with c3:
                if env_loaded and "humidity" in env_df.columns:
                    h_vals = env_df["humidity"].dropna()
                    if len(h_vals) >= 2:
                        st.metric(
                            "평균 습도",
                            f"{h_vals.mean():.1f}%",
                        )
                    elif len(h_vals) == 1:
                        st.metric("평균 습도", f"{h_vals.iloc[0]:.1f}%")
                    else:
                        st.metric("평균 습도", "데이터 없음")
                else:
                    st.metric("평균 습도", "데이터 없음")

            # 평균 생중량
            with c4:
                if growth_loaded and "생중량" in growth_df.columns:
                    g_vals = growth_df["생중량"].dropna()
                    if len(g_vals) >= 2:
                        st.metric(
                            "평균 생중량",
                            f"{g_vals.mean():.2f}g",
                        )
                    elif len(g_vals) == 1:
                        st.metric("평균 생중량", f"{g_vals.iloc[0]:.2f}g")
                    else:
                        st.metric("평균 생중량", "데이터 없음")
                else:
                    st.metric("평균 생중량", "데이터 없음")

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ---------- Tab 2: 환경 데이터 ----------
    with tab2:
        try:
            if not env_loaded:
                st.info("📌 환경 CSV를 사이드바에서 업로드해 주세요. 업로드하면 이 탭의 내용이 표시됩니다.")
            else:
                st.header("🌡️ 환경 데이터")

                # 환경 요약 표
                st.subheader("환경 요약 (평균·최소·최대)")
                env_summary_rows = []
                for c in ["temperature", "humidity", "ph", "ec"]:
                    if c in env_df.columns:
                        vals = env_df[c].dropna()
                        if len(vals) >= 2:
                            label_kr = {
                                "temperature": "온도(℃)",
                                "humidity": "습도(%)",
                                "ph": "pH",
                                "ec": "EC",
                            }.get(c, c)
                            env_summary_rows.append(
                                {
                                    "측정 항목": label_kr,
                                    "평균": f"{vals.mean():.2f}",
                                    "최소": f"{vals.min():.2f}",
                                    "최대": f"{vals.max():.2f}",
                                }
                            )
                        else:
                            label_kr = {
                                "temperature": "온도(℃)",
                                "humidity": "습도(%)",
                                "ph": "pH",
                                "ec": "EC",
                            }.get(c, c)
                            env_summary_rows.append(
                                {
                                    "측정 항목": label_kr,
                                    "평균": "데이터 부족",
                                    "최소": "데이터 부족",
                                    "최대": "데이터 부족",
                                }
                            )
                if env_summary_rows:
                    st.dataframe(
                        pd.DataFrame(env_summary_rows),
                        width="stretch",
                        hide_index=True,
                    )
                else:
                    st.info("📌 환경 데이터에서 온도·습도·pH·EC 열을 찾지 못했어요.")

                # 날짜별 시계열
                if "time" in env_df.columns and env_df["time"].notna().any():
                    st.subheader("날짜별 시계열 (일별 평균)")

                    env_df_day = (
                        env_df.dropna(subset=["time"])
                        .copy()
                        .assign(date=lambda d: d["time"].dt.floor("D"))
                    )

                    env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df_day.columns]
                    if env_cols:
                        day_agg = (
                            env_df_day.groupby("date")[env_cols]
                            .mean()
                            .reset_index()
                            .sort_values("date")
                        )

                        c_env1, c_env2 = st.columns(2)
                        with c_env1:
                            if "temperature" in day_agg.columns:
                                fig_temp = px.line(
                                    day_agg,
                                    x="date",
                                    y="temperature",
                                    markers=True,
                                    title="일별 평균 온도 변화",
                                )
                                fig_temp.add_trace(
                                    go.Scatter(
                                        x=day_agg["date"],
                                        y=day_agg["temperature"],
                                        mode="markers",
                                        marker=dict(size=6),
                                        showlegend=False,
                                    )
                                )
                                korean_layout(fig_temp)
                                fig_temp.update_layout(
                                    xaxis_title="날짜",
                                    yaxis_title="온도(℃)",
                                )
                                st.plotly_chart(fig_temp, width="stretch")
                            else:
                                st.info("📌 온도 열이 없어 온도 그래프를 표시할 수 없어요.")

                        with c_env2:
                            if "humidity" in day_agg.columns:
                                fig_hum = px.line(
                                    day_agg,
                                    x="date",
                                    y="humidity",
                                    markers=True,
                                    title="일별 평균 습도 변화",
                                )
                                fig_hum.add_trace(
                                    go.Scatter(
                                        x=day_agg["date"],
                                        y=day_agg["humidity"],
                                        mode="markers",
                                        marker=dict(size=6),
                                        showlegend=False,
                                    )
                                )
                                korean_layout(fig_hum)
                                fig_hum.update_layout(
                                    xaxis_title="날짜",
                                    yaxis_title="습도(%)",
                                )
                                st.plotly_chart(fig_hum, width="stretch")
                            else:
                                st.info("📌 습도 열이 없어 습도 그래프를 표시할 수 없어요.")

                        # pH·EC 변화 (열이 있을 때만)
                        env_leftover = [c for c in ["ph", "ec"] if c in day_agg.columns]
                        if len(env_leftover) >= 2:
                            c_env3, c_env4 = st.columns(2)
                            with c_env3:
                                if "ph" in day_agg.columns:
                                    fig_ph = px.line(
                                        day_agg,
                                        x="date",
                                        y="ph",
                                        markers=True,
                                        title="일별 평균 pH 변화",
                                    )
                                    fig_ph.add_trace(
                                        go.Scatter(
                                            x=day_agg["date"],
                                            y=day_agg["ph"],
                                            mode="markers",
                                            marker=dict(size=6),
                                            showlegend=False,
                                        )
                                    )
                                    korean_layout(fig_ph)
                                    fig_ph.update_layout(
                                        xaxis_title="날짜",
                                        yaxis_title="pH",
                                    )
                                    st.plotly_chart(fig_ph, width="stretch")
                                else:
                                    st.info("📌 pH 열이 없어 pH 그래프를 표시할 수 없어요.")
                            with c_env4:
                                if "ec" in day_agg.columns:
                                    fig_ec = px.line(
                                        day_agg,
                                        x="date",
                                        y="ec",
                                        markers=True,
                                        title="일별 평균 EC 변화",
                                    )
                                    fig_ec.add_trace(
                                        go.Scatter(
                                            x=day_agg["date"],
                                            y=day_agg["ec"],
                                            mode="markers",
                                            marker=dict(size=6),
                                            showlegend=False,
                                        )
                                    )
                                    korean_layout(fig_ec)
                                    fig_ec.update_layout(
                                        xaxis_title="날짜",
                                        yaxis_title="EC",
                                    )
                                    st.plotly_chart(fig_ec, width="stretch")
                                else:
                                    st.info("📌 EC 열이 없어 EC 그래프를 표시할 수 없어요.")
                        elif len(env_leftover) == 1:
                            c_env3 = st.columns(1)[0]
                            if "ph" in day_agg.columns:
                                fig_ph = px.line(
                                    day_agg,
                                    x="date",
                                    y="ph",
                                    markers=True,
                                    title="일별 평균 pH 변화",
                                )
                                fig_ph.add_trace(
                                    go.Scatter(
                                        x=day_agg["date"],
                                        y=day_agg["ph"],
                                        mode="markers",
                                        marker=dict(size=6),
                                        showlegend=False,
                                    )
                                )
                                korean_layout(fig_ph)
                                fig_ph.update_layout(
                                    xaxis_title="날짜",
                                    yaxis_title="pH",
                                )
                                st.plotly_chart(fig_ph, width="stretch")
                            elif "ec" in day_agg.columns:
                                fig_ec = px.line(
                                    day_agg,
                                    x="date",
                                    y="ec",
                                    markers=True,
                                    title="일별 평균 EC 변화",
                                )
                                fig_ec.add_trace(
                                    go.Scatter(
                                        x=day_agg["date"],
                                        y=day_agg["ec"],
                                        mode="markers",
                                        marker=dict(size=6),
                                        showlegend=False,
                                    )
                                )
                                korean_layout(fig_ec)
                                fig_ec.update_layout(
                                    xaxis_title="날짜",
                                    yaxis_title="EC",
                                )
                                st.plotly_chart(fig_ec, width="stretch")
                    else:
                        st.info("📌 환경 데이터에서 시계열로 그릴 측정 항목이 없어요.")
                else:
                    st.info("📌 환경 데이터에 날짜/시간 열이 없거나 형식이 달라 시계열을 그릴 수 없어요.")

                # expander: 환경 데이터 원본 + CSV 다운로드
                with st.expander("📄 환경 데이터 원본 테이블 + CSV 다운로드"):
                    if env_loaded:
                        st.dataframe(env_df, width="stretch")
                        csv_env = env_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="환경 CSV 다운로드 (utf-8-sig)",
                            data=csv_env,
                            file_name=f"{school}_환경데이터_다운로드.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

    # ---------- Tab 3: 생육 결과 ----------
    with tab3:
        try:
            if not growth_loaded:
                st.info("📌 생육 결과 CSV를 사이드바에서 업로드해 주세요. 업로드하면 이 탭의 내용이 표시됩니다.")
            else:
                st.header("📊 생육 결과")

                # ----- 핵심 결과 카드 -----
                st.subheader("🥇 핵심 결과 카드")
                if "생중량" in growth_df.columns:
                    g_vals = growth_df["생중량"].dropna()
                    if len(g_vals) >= 1:
                        mean_g = g_vals.mean()
                        max_g = g_vals.max()
                        min_g = g_vals.min()
                        idx_max = g_vals.idxmax()
                        idx_min = g_vals.idxmin()
                        label_max = growth_df.loc[idx_max, "개체번호"] if "개체번호" in growth_df.columns else f"개체 {idx_max}"
                        label_min = growth_df.loc[idx_min, "개체번호"] if "개체번호" in growth_df.columns else f"개체 {idx_min}"
                        kc1, kc2, kc3, kc4 = st.columns(4)
                        with kc1:
                            st.metric("평균 생중량", f"{mean_g:.2f}g")
                        with kc2:
                            st.metric("최대 생중량", f"{max_g:.2f}g", delta=f"({label_max})")
                        with kc3:
                            st.metric("최소 생중량", f"{min_g:.2f}g", delta=f"({label_min})")
                        with kc4:
                            if len(g_vals) >= 2:
                                cv = (g_vals.std(ddof=1) / mean_g) * 100
                                st.metric("변동계수(CV)", f"{cv:.1f}%")
                            else:
                                st.metric("변동계수(CV)", "데이터 부족")
                    else:
                        st.info("📌 생중량 값이 하나도 없어 핵심 결과 카드를 표시할 수 없어요.")
                else:
                    st.info("📌 생육 데이터에 '생중량' 열이 없어 핵심 결과 카드를 표시할 수 없어요.")

                # ----- 개체별 생육 비교 (막대 그래프) -----
                st.subheader("개체별 생육 비교 (막대 그래프)")
                if not selected_measures:
                    st.info("📌 비교할 측정 항목을 사이드바에서 선택해 주세요.")
                else:
                    # 생중량 있으면 맨 앞으로
                    measure_cols = []
                    if "생중량" in growth_df.columns and "생중량" in selected_measures:
                        measure_cols.insert(0, "생중량")
                    for m in selected_measures:
                        if m != "생중량" and m in growth_df.columns:
                            measure_cols.append(m)

                    # 상위·하위 25% 계산 (생중량 기준)
                    top_bottom_info = {}
                    if "생중량" in growth_df.columns:
                        g_vals = growth_df["생중량"].dropna()
                        if len(g_vals) >= 4:
                            q75 = g_vals.quantile(0.75)
                            q25 = g_vals.quantile(0.25)
                            top_ids = growth_df.loc[
                                growth_df["생중량"] >= q75, "개체번호"
                            ].tolist() if "개체번호" in growth_df.columns else []
                            bottom_ids = growth_df.loc[
                                growth_df["생중량"] <= q25, "개체번호"
                            ].tolist() if "개체번호" in growth_df.columns else []
                            top_bottom_info = {
                                "q75": q75,
                                "q25": q25,
                                "top_ids": top_ids,
                                "bottom_ids": bottom_ids,
                            }

                    n_measures = len(measure_cols)
                    if n_measures == 0:
                        st.info("📌 표시할 측정 항목이 없어요.")
                    else:
                        rows_needed = (n_measures + 1) // 2
                        for r in range(rows_needed):
                            cols = st.columns(2)
                            for ci in range(2):
                                idx = r * 2 + ci
                                if idx >= n_measures:
                                    break
                                m = measure_cols[idx]
                                label_kr = {
                                    "잎 수": "평균 잎 수",
                                    "지상부 길이": "평균 지상부 길이",
                                    "지하부 길이": "평균 지하부 길이",
                                    "생중량": "평균 생중량",
                                }.get(m, m)
                                title_txt = f"개체별 {label_kr}"

                                vals = growth_df[m].dropna()
                                if len(vals) == 0:
                                    cols[ci].info(f"📌 {label_kr} 값이 없어 그래프를 표시할 수 없어요.")
                                    continue

                                fig_bar = go.Figure()
                                ids = growth_df["개체번호"] if "개체번호" in growth_df.columns else growth_df.index
                                fig_bar.add_trace(
                                    go.Bar(
                                        x=ids,
                                        y=growth_df[m],
                                        name=label_kr,
                                        marker=dict(
                                            color=[
                                                "#FFB347"
                                                if (
                                                    "생중량" in growth_df.columns
                                                    and m == "생중량"
                                                    and len(top_bottom_info) > 0
                                                    and id_ in top_bottom_info["top_ids"]
                                                )
                                                else "#87CEEB"
                                                if (
                                                    "생중량" in growth_df.columns
                                                    and m == "생중량"
                                                    and len(top_bottom_info) > 0
                                                    and id_ in top_bottom_info["bottom_ids"]
                                                )
                                                else "#A9A9A9"
                                                for id_ in ids
                                            ],
                                            line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
                                        ),
                                        hovertemplate="%{x}<br>" + label_kr + ": %{y:.2f}<extra></extra>",
                                    )
                                )
                                korean_layout(fig_bar, title=title_txt)
                                fig_bar.update_layout(
                                    xaxis_title="개체",
                                    yaxis_title=label_kr,
                                    xaxis=dict(tickangle=45),
                                    bargap=0.2,
                                    height=380,
                                )
                                cols[ci].plotly_chart(fig_bar, width="stretch")

                                # 범례 설명
                                if m == "생중량" and len(top_bottom_info) > 0:
                                    with cols[ci]:
                                        st.caption(
                                            f"🟠 주황색 막대 = 생중량 상위 25% (Q3 이상, {len(top_bottom_info['top_ids'])}개체) · 🔵 파란색 막대 = 생중량 하위 25% (Q1 이하, {len(top_bottom_info['bottom_ids'])}개체)"
                                        )

                # ----- 생중량 분포 (박스플롯/바이올린) -----
                st.subheader("생중량 분포 (박스플롯)")
                if "생중량" in growth_df.columns:
                    g_vals = growth_df["생중량"].dropna()
                    if len(g_vals) >= 2:
                        fig_box = go.Figure()
                        fig_box.add_trace(
                            go.Box(
                                y=growth_df["생중량"],
                                name="생중량",
                                fillcolor="rgba(255, 179, 71, 0.5)",
                                line=dict(color="rgba(255, 140, 0, 0.8)"),
                                boxpoints="all",
                                jitter=0.3,
                                pointpos=-1.8,
                                marker=dict(
                                    color=[
                                        "#FF6B6B" if (
                                            len(top_bottom_info) > 0
                                            and id_ in top_bottom_info["top_ids"]
                                        )
                                        else "#4ECDC4" if (
                                            len(top_bottom_info) > 0
                                            and id_ in top_bottom_info["bottom_ids"]
                                        )
                                        else "#95A5A6"
                                        for id_ in (growth_df["개체번호"] if "개체번호" in growth_df.columns else growth_df.index)
                                    ],
                                    size=6,
                                ),
                                hovertemplate="개체: %{x}<br>생중량: %{y:.2f}g<extra></extra>",
                            )
                        )
                        korean_layout(fig_box, title="생중량 분포 (박스플롯)")
                        fig_box.update_layout(
                            yaxis_title="생중량(g)",
                            height=380,
                        )
                        st.plotly_chart(fig_box, width="stretch")

                        with st.expander("📌 박스플롯 읽는 법"):
                            st.markdown(
                                """
- **박스(Box)**: 데이터의 중간 50%가 들어 있는 구간(Q1 ~ Q3).
- **박스 안의 선**: 중앙값(median).
- **수염(Whisker)**: 이상치를 제외한 데이터의 일반적인 범위.
- **점(Dots)**: 이상치 후보 또는 개별 관측값.
- **주황색 점**: 생중량 상위 25% 개체, **민트색 점**: 생중량 하위 25% 개체.

> 개체별 생중량 차이가 클수록 박스가 길어지고, 수염 바깥의 점이 많아집니다.
"""
                            )
                    else:
                        st.info("📌 생중량 값이 2개 미만이라 분포를 표시할 수 없어요.")
                else:
                    st.info("📌 생육 데이터에 '생중량' 열이 없어 분포 그래프를 표시할 수 없어요.")

                # ----- 상관관계 분석 (산점도 2개 + 추세선) -----
                st.subheader("상관관계 분석 (산점도 + 추세선)")
                corr_pairs = []
                if "생중량" in growth_df.columns:
                    for m in ["잎 수", "지상부 길이", "지하부 길이"]:
                        if m in growth_df.columns:
                            corr_pairs.append((m, "생중량"))

                if not corr_pairs:
                    st.info("📌 생중량과 비교할 측정 항목(잎 수·지상부 길이·지하부 길이)이 없어요.")
                else:
                    n_corr = len(corr_pairs)
                    for cc in range((n_corr + 1) // 2):
                        cols_corr = st.columns(2)
                        for ci in range(2):
                            idx = cc * 2 + ci
                            if idx >= n_corr:
                                break
                            x_col, y_col = corr_pairs[idx]
                            x_vals = growth_df[x_col].dropna()
                            y_vals = growth_df.loc[x_vals.index, y_col].dropna() if x_vals.index.isin(growth_df.index) else growth_df[y_col].dropna()

                            # 재정렬
                            df_corr = growth_df[[x_col, y_col]].dropna()
                            if len(df_corr) < 2:
                                cols_corr[ci].info(f"📌 {x_col}와 {y_col}의 비교 데이터가 부족해요.")
                                continue

                            fig_scatter = px.scatter(
                                df_corr,
                                x=x_col,
                                y=y_col,
                                title=f"{x_col} vs {y_col}",
                                labels={x_col: x_col, y_col: y_col},
                                hover_name=growth_df.loc[df_corr.index, "개체번호"] if "개체번호" in growth_df.columns else None,
                            )
                            add_trendline(fig_scatter, df_corr[x_col], df_corr[y_col])
                            korean_layout(fig_scatter)
                            fig_scatter.update_layout(
                                xaxis_title=x_col,
                                yaxis_title=y_col,
                                height=380,
                            )
                            cols_corr[ci].plotly_chart(fig_scatter, width="stretch")

                            with cols_corr[ci]:
                                r_text = corr_text(df_corr[x_col], df_corr[y_col])
                                st.caption(f"📐 {r_text}")
                                if x_col == "잎 수":
                                    st.caption("잎 수가 많을수록 생중량이 함께 증가하는 경향이 있는지 확인합니다.")
                                elif x_col == "지상부 길이":
                                    st.caption("지상부 길이가 길수록 생중량이 함께 증가하는 경향이 있는지 확인합니다.")
                                elif x_col == "지하부 길이":
                                    st.caption("지하부 길이가 길수록 생중량이 함께 증가하는 경향이 있는지 확인합니다.")

                # ----- expander: 생육 데이터 원본 + CSV 다운로드 -----
                with st.expander("📄 생육 데이터 원본 + CSV 다운로드"):
                    if growth_loaded:
                        st.dataframe(growth_df, width="stretch")
                        csv_growth = growth_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            label="생육 결과 CSV 다운로드 (utf-8-sig)",
                            data=csv_growth,
                            file_name=f"{school}_생육결과데이터_다운로드.csv",
                            mime="text/csv",
                        )

        except Exception as e:
            st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


if __name__ == "__main__":
    main()
