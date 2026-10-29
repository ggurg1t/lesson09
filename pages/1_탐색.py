import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="뇌졸중 예측 실습실 - 탐색", page_icon="🔎")

DATA_URL = "https://raw.githubusercontent.com/ggurg1t/data/refs/heads/main/stroke.csv"

GROUP_ORDER = ["뇌졸중 겪지 않음", "뇌졸중 겪음"]
GROUP_COLORS = {"뇌졸중 겪지 않음": "#4C78A8", "뇌졸중 겪음": "#E45756"}


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def stroke_rate_bar(data: pd.DataFrame, column: str, title: str):
    """어떤 열의 값(0/1)별 뇌졸중 비율을 막대그래프로 만든다."""
    rate = (
        data.groupby(column)["stroke"]
        .agg(비율=lambda s: s.mean() * 100, 사람수="size")
        .reset_index()
    )
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate,
        x="구분",
        y="비율",
        text=rate["비율"].map(lambda v: f"{v:.2f}%"),
        hover_data={"사람수": True, "비율": ":.2f"},
        title=title,
        category_orders={"구분": ["없음", "있음"]},
        labels={"비율": "뇌졸중 비율 (%)", "구분": ""},
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, max(rate["비율"].max() * 1.25, 1)])
    return fig


st.title("🔎 탐색")
st.caption("같은 데이터를 여러 방향에서 살펴봅니다.")

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

df = df.copy()
df["그룹"] = df["stroke"].map({0: "뇌졸중 겪지 않음", 1: "뇌졸중 겪음"})

# ---------------------------------------------------------------
# 1. 나이와 평균 혈당의 분포
# ---------------------------------------------------------------
st.header("1. 나이와 평균 혈당의 분포")
left, right = st.columns(2)
with left:
    fig = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)
with right:
    fig = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당"},
    )
    fig.update_layout(yaxis_title="사람 수", bargap=0.05)
    st.plotly_chart(fig)

# ---------------------------------------------------------------
# 2. 뇌졸중 여부에 따른 비교
# ---------------------------------------------------------------
st.header("2. 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")
left, right = st.columns(2)
with left:
    fig = px.box(
        df,
        x="그룹",
        y="age",
        color="그룹",
        title="나이",
        category_orders={"그룹": GROUP_ORDER},
        color_discrete_map=GROUP_COLORS,
        labels={"age": "나이", "그룹": ""},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)
with right:
    fig = px.box(
        df,
        x="그룹",
        y="avg_glucose_level",
        color="그룹",
        title="평균 혈당",
        category_orders={"그룹": GROUP_ORDER},
        color_discrete_map=GROUP_COLORS,
        labels={"avg_glucose_level": "평균 혈당", "그룹": ""},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

mean_table = (
    df.groupby("그룹")[["age", "avg_glucose_level"]]
    .mean()
    .reindex(GROUP_ORDER)
    .rename(columns={"age": "나이 평균", "avg_glucose_level": "평균 혈당의 평균"})
    .round(2)
    .reset_index()
    .rename(columns={"그룹": "구분"})
)
st.markdown("**두 그룹의 평균값**")
st.dataframe(mean_table, hide_index=True)

# ---------------------------------------------------------------
# 3. 고혈압, 심장병과 뇌졸중 비율
# ---------------------------------------------------------------
st.header("3. 고혈압·심장병 여부에 따른 뇌졸중 비율")
left, right = st.columns(2)
with left:
    st.plotly_chart(stroke_rate_bar(df, "hypertension", "고혈압 여부별 뇌졸중 비율"))
with right:
    st.plotly_chart(stroke_rate_bar(df, "heart_disease", "심장병 여부별 뇌졸중 비율"))

# ---------------------------------------------------------------
# 4. bmi가 빈 사람들
# ---------------------------------------------------------------
st.header("4. 체질량지수(bmi)가 비어 있는 사람들")
bmi_missing = df[df["bmi"].isna()]


def summarize(label: str, part: pd.DataFrame) -> dict:
    n = len(part)
    n_stroke = int((part["stroke"] == 1).sum())
    return {
        "구분": label,
        "사람 수": n,
        "뇌졸중 사람 수": n_stroke,
        "뇌졸중 비율": f"{n_stroke / n * 100:.2f}%" if n else "-",
    }


compare = pd.DataFrame(
    [summarize("bmi가 빈 사람들", bmi_missing), summarize("전체", df)]
)
st.dataframe(compare, hide_index=True)

# ---------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ---------------------------------------------------------------
st.header("5. 흡연 상태별 사람 수")
smoking = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
smoking["흡연 상태"] = smoking["흡연 상태"].fillna("(빈 값)")
st.dataframe(smoking, hide_index=True)
