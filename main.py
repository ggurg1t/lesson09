import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# 기본 설정: 브라우저 탭 제목과 아이콘
# ---------------------------------------------------------------
st.set_page_config(page_title="뇌졸중 예측 실습실", page_icon="🧠")

DATA_URL = "https://raw.githubusercontent.com/ggurg1t/data/refs/heads/main/stroke.csv"
SAVE_DIR = Path(__file__).parent / "내_기록"
SAVE_FILE = SAVE_DIR / "기록.json"

COL_NAME = "열 이름"
COL_MEANING = "우리말 뜻"
COL_KIND = "값의 종류"
COL_NULL = "빈 값 개수"


# ---------------------------------------------------------------
# 저장 / 불러오기
# ---------------------------------------------------------------
def load_notes() -> dict:
    """내_기록 폴더의 json 파일에서 이전에 적은 내용을 불러온다."""
    try:
        with open(SAVE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        meanings = data.get("meanings", {})
        source = data.get("source", "")
        if not isinstance(meanings, dict) or not isinstance(source, str):
            raise ValueError
        return {"meanings": meanings, "source": source}
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
        return {"meanings": {}, "source": ""}


def save_notes(meanings: dict, source: str) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"meanings": meanings, "source": source},
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def describe_kind(series: pd.Series) -> str:
    """열에 들어 있는 값이 어떤 종류인지 짧게 설명한다."""
    values = series.dropna()
    n_unique = values.nunique()
    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 2:
            shown = ", ".join(str(v) for v in sorted(values.unique()))
            return f"숫자 · {n_unique}가지 ({shown})"
        if n_unique == len(values):
            return f"숫자 · 번호 (모두 다른 값)"
        return f"숫자 · {values.min():g} ~ {values.max():g}"
    shown = ", ".join(str(v) for v in values.unique()[:8])
    return f"글자 · {n_unique}가지 ({shown})"


# ---------------------------------------------------------------
# 화면 시작
# ---------------------------------------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.caption("뇌졸중 데이터를 살펴보며 예측 모델을 만들어 보는 실습 공간입니다.")

st.markdown(
    """
    <style>
    .big-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
        padding: 1.2rem 0.5rem;
        text-align: center;
        background: rgba(128, 128, 128, 0.07);
    }
    .big-card .label {
        font-size: 1rem;
        opacity: 0.75;
    }
    .big-card .value {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1.3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# 새로고침하면 파일에서 다시 불러온다 (세션당 한 번만)
if "notes_loaded" not in st.session_state:
    notes = load_notes()
    st.session_state["initial_meanings"] = notes["meanings"]
    st.session_state["source_text"] = notes["source"]
    st.session_state["notes_loaded"] = True

# ---------------------------------------------------------------
# 1. 큰 숫자 카드 네 개
# ---------------------------------------------------------------
n_rows = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / n_rows * 100 if n_rows else 0

cards = [
    ("전체 사람 수", f"{n_rows:,}명"),
    ("열 개수", f"{n_cols}개"),
    ("stroke가 1인 사람 수", f"{n_stroke:,}명"),
    ("stroke가 1인 비율", f"{ratio:.2f}%"),
]
for col, (label, value) in zip(st.columns(4), cards):
    col.markdown(
        f'<div class="big-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------
# 2. 열 설명 표 (우리말 뜻은 직접 입력)
# ---------------------------------------------------------------
st.subheader("열 설명 표")
st.caption("‘우리말 뜻’ 칸을 눌러 교재를 보고 직접 적어 넣으세요. 적은 내용은 자동으로 저장됩니다.")

base_table = pd.DataFrame(
    {
        COL_NAME: list(df.columns),
        COL_MEANING: [
            st.session_state["initial_meanings"].get(c, "") for c in df.columns
        ],
        COL_KIND: [describe_kind(df[c]) for c in df.columns],
        COL_NULL: [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    base_table,
    key="column_table",
    hide_index=True,
    disabled=[COL_NAME, COL_KIND, COL_NULL],
    column_config={
        COL_MEANING: st.column_config.TextColumn(COL_MEANING, help="교재를 보고 직접 적어 주세요."),
    },
)

# ---------------------------------------------------------------
# 3. 처음 다섯 줄
# ---------------------------------------------------------------
st.subheader("데이터 처음 다섯 줄")
st.dataframe(df.head(5), hide_index=True)

# ---------------------------------------------------------------
# 4. 데이터 출처
# ---------------------------------------------------------------
st.subheader("데이터 출처")
source_text = st.text_area(
    "출처를 적어 주세요 (교재에 있는 글을 옮겨 적으면 됩니다).",
    key="source_text",
    height=120,
)

# ---------------------------------------------------------------
# 저장: 바뀐 내용이 있을 때만 파일에 쓴다
# ---------------------------------------------------------------
current_meanings = {
    row[COL_NAME]: (row[COL_MEANING] or "") for _, row in edited.iterrows()
}
saved = load_notes()
saved_meanings = {c: saved["meanings"].get(c, "") for c in df.columns}

if current_meanings != saved_meanings or source_text != saved["source"]:
    try:
        save_notes(current_meanings, source_text)
        st.toast("내_기록 폴더에 저장했어요.", icon="💾")
    except OSError as e:
        st.warning(f"저장하지 못했습니다: {e}")
