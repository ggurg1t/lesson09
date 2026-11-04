import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="뇌졸중 예측 실습실 - 채점표", page_icon="📝")

DATA_URL = "https://raw.githubusercontent.com/ggurg1t/data/refs/heads/main/stroke.csv"

# '분류 모델' 페이지에서 처음 골라져 있던 입력 속성과 같다 (체질량지수 제외)
FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

POSITIVE = 1  # stroke가 1이면 뇌졸중(양성), 0이면 아님
LIST_LIMIT = 20
NOT_AVAILABLE = "계산할 수 없음"

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"
BASE_NAME = "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)"


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def confusion_counts(y_true, y_pred) -> dict:
    """뇌졸중을 양성으로 두고 TP·FN·FP·TN 사람 수를 센다."""
    (tp, fn), (fp, tn) = confusion_matrix(y_true, y_pred, labels=[POSITIVE, 0])
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)}


def metrics_from(c: dict) -> dict:
    """혼동행렬의 네 칸에서 정확도·재현율·정밀도·F1을 구한다. (비율은 0~1)"""
    total = c["TP"] + c["FN"] + c["FP"] + c["TN"]
    actual_pos = c["TP"] + c["FN"]  # 재현율의 분모: 실제 뇌졸중인 사람 수
    pred_pos = c["TP"] + c["FP"]  # 정밀도의 분모: 뇌졸중이라 예측한 사람 수

    accuracy = (c["TP"] + c["TN"]) / total if total else None
    recall = c["TP"] / actual_pos if actual_pos else None
    precision = c["TP"] / pred_pos if pred_pos else None  # 예측한 사람이 0명이면 계산할 수 없음

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 정밀도와 재현율이 모두 0이면 F1도 0으로 둔다
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"정확도": accuracy, "재현율": recall, "정밀도": precision, "F1": f1}


def pct(value) -> str:
    return NOT_AVAILABLE if value is None else f"{value * 100:.2f}%"


def confusion_table_html(c: dict) -> str:
    """행은 실제, 열은 예측인 2×2 표 (합계 포함)를 HTML로 만든다."""
    row_pos = c["TP"] + c["FN"]
    row_neg = c["FP"] + c["TN"]
    col_pos = c["TP"] + c["FP"]
    col_neg = c["FN"] + c["TN"]
    total = row_pos + row_neg

    def cell(abbr: str, count: int, kind: str) -> str:
        return (
            f'<td class="cm-cell {kind}"><div class="cm-abbr">{abbr}</div>'
            f'<div class="cm-count">{count:,}명</div></td>'
        )

    def total_cell(count: int) -> str:
        return f'<td class="cm-total">{count:,}명</td>'

    rows = [
        '<table class="cm">',
        '<tr><th class="cm-corner">실제 \\ 예측</th><th>예측: 뇌졸중</th><th>예측: 아님</th><th class="cm-total-h">합계</th></tr>',
        "<tr><th>실제: 뇌졸중</th>"
        + cell("TP", c["TP"], "good")
        + cell("FN", c["FN"], "bad")
        + total_cell(row_pos)
        + "</tr>",
        "<tr><th>실제: 아님</th>"
        + cell("FP", c["FP"], "bad")
        + cell("TN", c["TN"], "good")
        + total_cell(row_neg)
        + "</tr>",
        '<tr><th class="cm-total-h">합계</th>'
        + total_cell(col_pos)
        + total_cell(col_neg)
        + total_cell(total)
        + "</tr>",
        "</table>",
    ]
    return "".join(rows)


def person_list(part: pd.DataFrame) -> pd.DataFrame:
    """번호·나이·평균 혈당·고혈압 여부·심장병 여부 목록을 나이가 많은 순으로 정렬한다."""
    out = part.sort_values(["age", "id"], ascending=[False, True])
    return pd.DataFrame(
        {
            "번호": out["id"].to_numpy(),
            "나이": out["age"].to_numpy(),
            "평균 혈당": out["avg_glucose_level"].round(2).to_numpy(),
            "고혈압 여부": out["hypertension"].map({1: "있음", 0: "없음"}).to_numpy(),
            "심장병 여부": out["heart_disease"].map({1: "있음", 0: "없음"}).to_numpy(),
        }
    )


st.markdown(
    """
    <style>
    table.cm { border-collapse: collapse; width: 100%; text-align: center; }
    table.cm th, table.cm td {
        border: 1px solid rgba(128, 128, 128, 0.4);
        padding: 0.5rem 0.4rem;
    }
    table.cm th { background: rgba(128, 128, 128, 0.12); font-weight: 600; font-size: 0.9rem; }
    table.cm .cm-corner { font-size: 0.8rem; opacity: 0.8; }
    table.cm .cm-abbr { font-size: 0.85rem; font-weight: 700; opacity: 0.8; }
    table.cm .cm-count { font-size: 1.6rem; font-weight: 700; line-height: 1.3; }
    table.cm .good { background: rgba(31, 119, 180, 0.15); }
    table.cm .bad { background: rgba(214, 39, 40, 0.15); }
    table.cm .cm-total { background: rgba(128, 128, 128, 0.07); font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📝 채점표")
st.caption("테스트 데이터로 모델의 답을 채점합니다. 뇌졸중(stroke = 1)을 양성으로 둡니다.")

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------------------------------------------------------------
# 1. '분류 모델' 페이지와 똑같이 나누고 학습하기
#    번호 순으로 정렬한 뒤 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정
# ---------------------------------------------------------------
data = raw.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3
train = data[~is_test].copy()
test = data[is_test].copy()

X_train, y_train = train[FEATURES], train["stroke"]
X_test, y_test = test[FEATURES], test["stroke"]

st.caption(
    "입력 속성: " + ", ".join(LABELS[f] for f in FEATURES)
    + f" · 훈련용 {len(train):,}명 · 테스트용 {len(test):,}명"
)

logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logit.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=0)
tree.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

models = {LOGIT_NAME: logit, TREE_NAME: tree, BASE_NAME: baseline}
predictions = {name: model.predict(X_test) for name, model in models.items()}
counts = {name: confusion_counts(y_test, pred) for name, pred in predictions.items()}

n_test = len(test)
n_test_pos = int((y_test == POSITIVE).sum())

# ---------------------------------------------------------------
# 2. 혼동행렬
# ---------------------------------------------------------------
st.header("혼동행렬")
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_test_pos:,}명입니다.")
st.caption("행은 실제, 열은 예측입니다. TP·FN·FP·TN은 뇌졸중을 양성으로 둔 약자입니다.")

left, right = st.columns(2)
for col, name in zip((left, right), (LOGIT_NAME, TREE_NAME)):
    with col:
        st.markdown(f"**{name}**")
        st.markdown(confusion_table_html(counts[name]), unsafe_allow_html=True)

st.caption(
    "TP: 뇌졸중인데 뇌졸중이라고 예측 · FN: 뇌졸중인데 아니라고 예측(놓친 환자) · "
    "FP: 아닌데 뇌졸중이라고 예측(헛짚은 사람) · TN: 아닌데 아니라고 예측"
)

# ---------------------------------------------------------------
# 3. 정확도·재현율·정밀도·F1
# ---------------------------------------------------------------
st.header("네 가지 점수")
st.write(f"테스트용 {n_test:,}명 가운데 뇌졸중인 사람은 {n_test_pos:,}명입니다.")

rows = []
for name in (LOGIT_NAME, TREE_NAME, BASE_NAME):
    m = metrics_from(counts[name])
    rows.append(
        {
            "모델": name,
            "정확도": pct(m["정확도"]),
            "재현율": pct(m["재현율"]),
            "정밀도": pct(m["정밀도"]),
            "F1": pct(m["F1"]),
        }
    )
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.caption(
    "정확도 = (TP + TN) ÷ 전체 사람 수 · "
    "재현율 = TP ÷ 실제 뇌졸중인 사람 수(TP + FN) · "
    "정밀도 = TP ÷ 뇌졸중이라 예측한 사람 수(TP + FP) · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없습니다. "
    "정밀도와 재현율이 모두 0이면 F1은 0으로 적었습니다."
)

# ---------------------------------------------------------------
# 4. 틀린 사람 목록
# ---------------------------------------------------------------
st.header("틀린 사람 목록")
chosen_name = st.selectbox("어느 모델의 목록을 볼까요?", [LOGIT_NAME, TREE_NAME])
chosen_pred = predictions[chosen_name]

missed = test[(y_test.to_numpy() == POSITIVE) & (chosen_pred != POSITIVE)]  # 놓친 환자 (FN)
wrong_guess = test[(y_test.to_numpy() != POSITIVE) & (chosen_pred == POSITIVE)]  # 헛짚은 사람 (FP)

left, right = st.columns(2)
for col, title, part in (
    (left, "놓친 환자 (실제 뇌졸중인데 아니라고 예측)", missed),
    (right, "헛짚은 사람 (아닌데 뇌졸중이라고 예측)", wrong_guess),
):
    with col:
        st.subheader(f"{title} — 전체 {len(part):,}명")
        listing = person_list(part)
        if len(listing) > LIST_LIMIT:
            st.caption(f"나이가 많은 순서로 앞 {LIST_LIMIT}명만 보여 줍니다.")
            listing = listing.head(LIST_LIMIT)
        else:
            st.caption("나이가 많은 순서입니다.")
        if listing.empty:
            st.write("해당하는 사람이 없습니다.")
        else:
            st.dataframe(listing, hide_index=True)

c = counts[chosen_name]
if len(missed) == c["FN"] and len(wrong_guess) == c["FP"]:
    st.write(
        f"두 목록의 전체 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong_guess):,}명)는 "
        f"{chosen_name}의 혼동행렬 FN({c['FN']:,}) · FP({c['FP']:,}) 칸과 같습니다."
    )
else:
    st.warning(
        f"두 목록의 사람 수(놓친 환자 {len(missed):,}명, 헛짚은 사람 {len(wrong_guess):,}명)가 "
        f"혼동행렬 FN({c['FN']:,}) · FP({c['FP']:,}) 칸과 다릅니다."
    )
