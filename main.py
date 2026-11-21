import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# ─────────────────────────────────────────────
# 데이터 로딩 (utf-8-sig → cp949 폴백)
# ─────────────────────────────────────────────
@st.cache_data
def load_csv(file):
    """bytes → 두 인코딩 시도 → pd.DataFrame"""
    raw = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


def std_cols(df):
    """열 이름을 표준 집합으로 통일 + 숫자 강제 변환"""
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
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
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
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


def 실수된_열(df, 후보):
    """df에 실제로 존재하는 후보 열만 골라 리스트로 반환"""
    return [c for c in 후보 if c in df.columns]


def 요약_데이터프레임(df, label):
    """데이터 요약 표용 2열 DataFrame 생성"""
    rows = []
    rows.append(["학교명", label])
    if "time" in df.columns and pd.api.types.is_datetime64_any_dtype(df["time"]):
        t = df["time"].dropna()
        if len(t) > 0:
            rows.append(["측정 기간", f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}"])
    if "개체번호" in df.columns:
        rows.append(["개체 수", str(df["개체번호"].nunique())])
    env_cols = ["temperature", "humidity", "ph", "ec"]
    growth_cols = ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]
    모든_열 = env_cols + growth_cols
    존재하는_열 = 실수된_열(df, 모든_열)
    if 존재하는_열:
        rows.append(["측정 항목", ", ".join(존재하는_열)])
    return pd.DataFrame(rows, columns=["항목", "값"])


# ─────────────────────────────────────────────
# Streamlit 앱
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지 식물 생육 대시보드", layout="wide")

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

st.title("❄️ 송천고에서는 극지 식물이 어떻게 자랄까?")

# ── 사이드바 ───────────────────────────────
env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("📊 생육 결과 데이터 CSV", type="csv")
선택_열 = st.sidebar.multiselect(
    "생육 측정 항목 선택",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"],
    default=None,
)
if not 선택_열:
    선택_열 = ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량", "생중량"]

학교명 = "우리 학교"
if env_file is not None:
    학교명 = str(env_file.name).split("_")[0]
if growth_file is not None:
    이름 = str(growth_file.name).split("_")[0]
    if 이름 and 이름 != "uploaded_file":
        학교명 = 이름

# ── 데이터 로딩 ────────────────────────────
with st.spinner("CSV를 읽고 있어요…"):
    env_df = load_csv(env_file) if env_file is not None else pd.DataFrame()
    growth_df = load_csv(growth_file) if growth_file is not None else pd.DataFrame()
    if not env_df.empty:
        env_df = std_cols(env_df)
    if not growth_df.empty:
        growth_df = std_cols(growth_df)

tabs = st.tabs(["📋 연구 배경 및 목적", "🌡️ 환경 데이터", "📊 생육 결과"])


# ══════════════════════════════════════════════
# Tab 1 — 연구 배경 및 목적
# ══════════════════════════════════════════════
with tabs[0]:
    try:
        st.markdown("### 연구 배경 및 목적")
        st.info(
            "송천고 재배 환경에서 극지 식물이 어떻게 생육하는지 파악하고, "
            "개체별 생육 차이와 부위 간 생장 관계를 살펴보기 위한 실험입니다."
        )
        st.markdown("#### 📋 데이터 요약")

        학교_df = 요약_데이터프레임(env_df, 학교명)
        성장_df = 요약_데이터프레임(growth_df, 학교명)
        전체 = pd.concat([학교_df, 성장_df], ignore_index=True)
        전체 = 전체.drop_duplicates().reset_index(drop=True)
        if 전체.empty or (전체.shape[0] == 1 and 전체.iloc[0, 0] == "학교명"):
            st.info("아직 CSV를 업로드하지 않았어요. 사이드바에서 파일을 올려주세요.")
        else:
            st.dataframe(전체, width="stretch", hide_index=True)

        # 주요 지표 카드 4개
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if "개체번호" in growth_df.columns:
                st.metric("총 개체 수", str(growth_df["개체번호"].nunique()))
            elif "개체번호" in env_df.columns:
                st.metric("총 개체 수", str(env_df["개체번호"].nunique()))
            else:
                st.metric("총 개체 수", "—")

        if "temperature" in env_df.columns:
            t_clean = env_df["temperature"].dropna()
            if len(t_clean) >= 2:
                with col2:
                    st.metric("🌡️ 평균 온도", f"{t_clean.mean():.1f} ℃")
        if "humidity" in env_df.columns:
            h_clean = env_df["humidity"].dropna()
            if len(h_clean) >= 2:
                with col3:
                    st.metric("💧 평균 습도", f"{h_clean.mean():.1f} %")
        if "생중량" in growth_df.columns:
            g_clean = growth_df["생중량"].dropna()
            if len(g_clean) >= 2:
                with col4:
                    st.metric("⚖️ 평균 생중량", f"{g_clean.mean():.1f} g")
        elif "생중량" in env_df.columns:
            g_clean = env_df["생중량"].dropna()
            if len(g_clean) >= 2:
                with col4:
                    st.metric("⚖️ 평균 생중량", f"{g_clean.mean():.1f} g")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# Tab 2 — 환경 데이터
# ══════════════════════════════════════════════
with tabs[1]:
    try:
        if env_df.empty:
            st.info("사이드바에서 환경 데이터 CSV를 올려주세요.")
        else:
            st.markdown(f"### 🌡️ {학교명} 환경 데이터")

            # 환경 요약 표
            env_cands = ["temperature", "humidity", "ph", "ec"]
            env_cols = 실수된_열(env_df, env_cands)
            if env_cols:
                s = env_df[env_cols].dropna()
                if len(s) >= 1:
                    summ = pd.DataFrame({
                        "항목": env_cols,
                        "평균": [round(s[c].mean(), 2) for c in env_cols],
                        "최소": [round(s[c].min(), 2) for c in env_cols],
                        "최대": [round(s[c].max(), 2) for c in env_cols],
                    })
                    st.dataframe(summ, width="stretch", hide_index=True)

            # 일별 시계열 (온도+습도 서브플롯)
            if "time" in env_df.columns and len(env_df) >= 2:
                idx = env_df.set_index("time").dropna(how="all").index
                if pd.api.types.is_datetime64_any_dtype(idx):
                    daily = env_df.set_index("time").resample("D").mean().dropna(how="all")

                    temp_ax = "temperature" if "temperature" in daily.columns else None
                    hum_ax = "humidity" if "humidity" in daily.columns else None
                    if temp_ax or hum_ax:
                        nrows = 2 if (temp_ax and hum_ax) else 1
                        fig = make_subplots(
                            rows=nrows, cols=1,
                            subplot_titles=(
                                ("온도 변화", "습도 변화")
                                if (temp_ax and hum_ax)
                                else ("온도 변화",)
                                if temp_ax
                                else ("습도 변화",)
                            ),
                            vertical_spacing=0.12,
                        )
                        r = 1
                        if temp_ax:
                            fig.add_trace(
                                go.Scatter(
                                    x=daily.index, y=daily[temp_ax],
                                    mode="lines", name="온도",
                                    line=dict(color="#e74c3c", width=2),
                                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>온도: %{y:.1f} ℃<extra></extra>",
                                ),
                                row=r, col=1,
                            )
                            r += 1
                        if hum_ax:
                            fig.add_trace(
                                go.Scatter(
                                    x=daily.index, y=daily[hum_ax],
                                    mode="lines", name="습도",
                                    line=dict(color="#3498db", width=2),
                                    hovertemplate="날짜: %{x|%Y-%m-%d}<br>습도: %{y:.1f} %<extra></extra>",
                                ),
                                row=r, col=1,
                            )
                        fig.update_layout(
                            height=280 * nrows + 40,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            showlegend=True,
                            margin=dict(l=40, r=20, t=40, b=40),
                            hovermode="x unified",
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        for ax, title in zip(
                            (temp_ax, hum_ax),
                            (("온도 변화", "습도 변화") if (temp_ax and hum_ax) else
                             (("온도 변화",) if temp_ax else ("습도 변화",))),
                        ):
                            pass
                        st.plotly_chart(fig, width="stretch")

            # pH·EC 변화 (열이 있을 때만)
            ph_ax = "ph" if "ph" in env_df.columns else None
            ec_ax = "ec" if "ec" in env_df.columns else None
            if ph_ax or ec_ax:
                sub_cols = [c for c in (ph_ax, ec_ax) if c is not None]
                if "time" in env_df.columns:
                    fig2 = make_subplots(
                        rows=len(sub_cols), cols=1,
                        subplot_titles=[f"{'pH' if c=='ph' else 'EC'} 변화" for c in sub_cols],
                        vertical_spacing=0.12,
                    )
                    colors = {"ph": "#27ae60", "ec": "#8e44ad"}
                    r = 1
                    for c in sub_cols:
                        d2 = env_df.dropna(subset=[c])
                        fig2.add_trace(
                            go.Scatter(
                                x=d2["time"], y=d2[c],
                                mode="lines", name=c.upper(),
                                line=dict(color=colors.get(c, "#333"), width=2),
                                hovertemplate=(
                                    "날짜: %{x|%Y-%m-%d %H:%M}<br>"
                                    + ("pH: %{y:.2f}" if c == "ph" else "EC: %{y:.2f} mS/cm")
                                    + "<extra></extra>"
                                ),
                            ),
                            row=r, col=1,
                        )
                        r += 1
                    fig2.update_layout(
                        height=200 * len(sub_cols) + 40,
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        showlegend=True,
                        margin=dict(l=40, r=20, t=40, b=40),
                        hovermode="x unified",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig2, width="stretch")

            # 원본 테이블 + CSV 다운로드
            with st.expander("📄 환경 데이터 원본 표 + CSV 다운로드"):
                st.dataframe(env_df, width="stretch", hide_index=True)
                csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ 환경 데이터 CSV 다운로드 (utf-8-sig)",
                    csv_bytes,
                    file_name=f"{학교명}_환경데이터.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ══════════════════════════════════════════════
# Tab 3 — 생육 결과
# ══════════════════════════════════════════════
with tabs[2]:
    try:
        if growth_df.empty:
            st.info("사이드바에서 생육 결과 데이터 CSV를 올려주세요.")
        else:
            st.markdown(f"### 📊 {학교명} 생육 결과")

            g = growth_df.copy()
            basic_cols = 실수된_열(g, ["잎 수", "지상부 길이", "지하부 길이",
                                       "지상부 질량", "지하부 질량", "생중량"])

            # 핵심 결과 카드
            col_a, col_b, col_c = st.columns(3)
            if "생중량" in g.columns:
                s = g["생중량"].dropna()
                if len(s) >= 2:
                    avg_g = s.mean()
                    with col_a:
                        st.metric("⚖️ 평균 생중량", f"{avg_g:.1f} g")
                    with col_b:
                        idx_max = s.idxmax()
                        val_max = s.max()
                        nm = g.loc[idx_max, "개체번호"]
                        st.metric("🏆 최대 생중량", f"{val_max:.1f} g")
                        st.caption(f"개체: {nm}")
                    with col_c:
                        idx_min = s.idxmin()
                        val_min = s.min()
                        nm2 = g.loc[idx_min, "개체번호"]
                        st.metric("🔻 최소 생중량", f"{val_min:.1f} g")
                        st.caption(f"개체: {nm2}")

            # 개체별 비교 (선택 열)
            st.markdown("#### 평균 생육 지표 비교")
            if basic_cols:
                clean = g.dropna(subset=basic_cols)
                if len(clean) >= 1:
                    avg_d = pd.DataFrame({
                        "측정 항목": basic_cols,
                        "평균값": [round(clean[c].mean(), 1) for c in basic_cols],
                    })
                    st.dataframe(avg_d, width="stretch", hide_index=True)

            # 생중량 분포 (박스플롯)
            if "생중량" in g.columns and len(g) >= 3:
                gc = g.dropna(subset=["생중량"])
                if len(gc) >= 3:
                    클래식 = gc["개체번호"].astype(str).tolist()
                    vals = gc["생중량"].tolist()
                    fig3 = go.Figure()
                    fig3.add_trace(go.Box(
                        y=vals,
                        name="생중량",
                        boxmean=True,
                        fillcolor="rgba(52,152,219,0.55)",
                        line=dict(color="#2c3e50", width=1.5),
                        hovertemplate="개체: %{x}<br>생중량: %{y:.1f} g<extra></extra>",
                        x=클래식,
                    ))
                    overall_mean = np.mean(vals)
                    fig3.add_hline(
                        y=overall_mean,
                        line_dash="dot",
                        line_color="coral",
                        annotation_text=f"전체 평균 {overall_mean:.1f} g",
                        annotation_position="top right",
                    )
                    fig3.update_layout(
                        title="개체별 생중량 분포 (박스플롯)",
                        yaxis_title="생중량 (g)",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=40, r=20, t=50, b=40),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                    )
                    st.plotly_chart(fig3, width="stretch")

            # 피어슨 상관 + 추세선 산점도 + 히트맵
            corr_cols = list(dict.fromkeys(["생중량"] + 선택_열))
            corr_cols = 실수된_열(g, corr_cols)
            if len(corr_cols) >= 3:
                clean_c = g.dropna(subset=corr_cols)
                if len(clean_c) >= 2:
                    st.markdown("#### 🔗 생육 지표 간 관계 (상관계수 · 추세선)")
                    for col in corr_cols:
                        if col == "생중량":
                            continue
                        pair = g.dropna(subset=[col, "생중량"])
                        if len(pair) < 2:
                            st.info(f"{col} vs 생중량: 데이터가 부족해 분석할 수 없어요.")
                            continue
                        xv = pair[col].values
                        yv = pair["생중량"].values
                        r_val, p_val = stats.pearsonr(xv, yv)
                        r2 = r_val ** 2
                        b, a = np.polyfit(xv, yv, 1)
                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(
                            x=xv, y=yv,
                            mode="markers",
                            marker=dict(
                                color="#e74c3c",
                                size=8,
                                line=dict(color="white", width=1),
                            ),
                            hovertemplate=(
                                f"{col}: %{{x:.1f}}<br>"
                                f"생중량: %{{y:.1f}} g<extra></extra>"
                            ),
                            name="개체",
                        ))
                        xs = np.linspace(xv.min(), xv.max(), 100)
                        fig_sc.add_trace(go.Scatter(
                            x=xs,
                            y=b * xs + a,
                            mode="lines",
                            line=dict(color="#2c3e50", width=2, dash="dash"),
                            name=f"추세선 (r={r_val:.2f})",
                        ))
                        fig_sc.add_annotation(
                            x=1, y=1, xref="paper", yref="paper",
                            text=f"Pearson r = {r_val:.2f}  (R² = {r2:.2f})",
                            showarrow=False,
                            font=dict(size=13, color="#2c3e50"),
                            bgcolor="rgba(255,255,255,0.85)",
                            borderpad=4,
                        )
                        fig_sc.update_layout(
                            title=f"{col} vs 생중량",
                            xaxis_title=col,
                            yaxis_title="생중량 (g)",
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                            margin=dict(l=50, r=20, t=50, b=50),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=380,
                        )
                        st.plotly_chart(fig_sc, width="stretch")

                    # 히트맵
                    cm = clean_c[corr_cols].corr().round(2)
                    fig_hm = go.Figure(data=go.Heatmap(
                        z=cm.values,
                        x=corr_cols,
                        y=corr_cols,
                        colorscale="RdBu_r",
                        zmid=0,
                        text=cm.values,
                        texttemplate="%{text}",
                        textfont={"size": 11, "color": "#2c3e50"},
                        hovertemplate="%{y} · %{x} = %{z:.2f}<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        title="생육 지표 간 피어슨 상관계수 히트맵",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        margin=dict(l=60, r=30, t=50, b=60),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=420,
                    )
                    st.plotly_chart(fig_hm, width="stretch")

            # 무거운 개체 vs 가벼운 개체 비교
            if "생중량" in g.columns and len(g) >= 4:
                gc2 = g.dropna(subset=["생중량"])
                if len(gc2) >= 4:
                    mean_g = gc2["생중량"].mean()
                    heavy = gc2[gc2["생중량"] >= mean_g].dropna(how="all")
                    light = gc2[gc2["생중량"] < mean_g].dropna(how="all")
                    comp_cols = 실수된_열(g, ["잎 수", "지상부 길이", "지하부 길이",
                                               "지상부 질량", "지하부 질량"])
                    if comp_cols and len(heavy) >= 1 and len(light) >= 1:
                        st.markdown("#### 🏋️ 무거운 개체 vs 가벼운 개체 특징 비교")
                        rows = []
                        for c in comp_cols:
                            hv = heavy[c].dropna()
                            lv = light[c].dropna()
                            if len(hv) >= 1 and len(lv) >= 1:
                                rows.append({
                                    "측정 항목": c,
                                    "무거운 개체 평균": round(hv.mean(), 1),
                                    "가벼운 개체 평균": round(lv.mean(), 1),
                                    "차이": round(hv.mean() - lv.mean(), 1),
                                })
                        if rows:
                            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            # 원본 테이블 + CSV 다운로드
            with st.expander("📄 생육 결과 원본 표 + CSV 다운로드"):
                st.dataframe(growth_df, width="stretch", hide_index=True)
                csv_bytes2 = growth_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️ 생육 결과 CSV 다운로드 (utf-8-sig)",
                    csv_bytes2,
                    file_name=f"{학교명}_생육결과.csv",
                    mime="text/csv",
                )
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
