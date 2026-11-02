import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="뇌졸중 예측 실습실 - 분류 모델", page_icon="🧪")

DATA_URL = "https://raw.githubusercontent.com/ggurg1t/data/refs/heads/main/stroke.csv"

FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]
LABELS = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}  # 0 또는 1 값을 갖는 속성

POSITIVE = 1  # stroke가 1이면 뇌졸중(양성), 0이면 아님
ANSWER_NAME = {0: "아님", 1: "뇌졸중"}

BLUE = "#1f77b4"
RED = "#d62728"
LIGHT_BLUE = "#cfe0f3"
LIGHT_RED = "#f6d0cf"
NODE_GRAY = "#f2f2f2"

GRID_N = 200


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt_num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def accuracy_card(title: str, train_acc: float, test_acc: float) -> str:
    return f"""
    <div class="acc-card">
      <div class="acc-title">{title}</div>
      <div class="acc-big">{test_acc * 100:.2f}%</div>
      <div class="acc-big-label">테스트 데이터 정확도</div>
      <div class="acc-small-row">
        <div><span>훈련 데이터</span><br><b>{train_acc * 100:.2f}%</b></div>
        <div><span>테스트 데이터</span><br><b>{test_acc * 100:.2f}%</b></div>
      </div>
    </div>
    """


def axis_range(feature: str, series: pd.Series) -> tuple:
    if feature in BINARY:
        return -0.5, 1.5
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


def linspace(lo: float, hi: float, n: int) -> list:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def build_dot(tree: DecisionTreeClassifier, features: list) -> str:
    """훈련된 의사결정트리를 DOT 문자열로 바꾼다."""
    t = tree.tree_
    lines = [
        "digraph Tree {",
        'node [shape=box, style="rounded,filled", fontname="sans-serif"];',
        'edge [fontname="sans-serif"];',
    ]
    for i in range(t.node_count):
        counts = t.value[i][0]
        n = int(t.n_node_samples[i])
        k = int(round(counts[1] / counts.sum() * n))  # 훈련용 중 실제 뇌졸중 사람 수
        rate = k / n * 100
        stats = f"훈련용 {n:,}명\\n뇌졸중 {k:,}명 ({rate:.1f}%)"
        left, right = int(t.children_left[i]), int(t.children_right[i])
        if left == -1:  # 더 묻지 않고 답을 내는 마디
            answer = int(tree.classes_[int(counts.argmax())])
            color = LIGHT_RED if answer == POSITIVE else LIGHT_BLUE
            label = f"답: {ANSWER_NAME[answer]}\\n{stats}"
            lines.append(f'n{i} [label="{label}", fillcolor="{color}"];')
        else:
            feat = features[int(t.feature[i])]
            if feat in BINARY:
                question = f"{LABELS[feat]}이(가) 있나요?"
                yes_child, no_child = right, left
            else:
                question = f"{LABELS[feat]} ≤ {fmt_num(float(t.threshold[i]))} ?"
                yes_child, no_child = left, right
            label = f"{question}\\n{stats}"
            lines.append(f'n{i} [label="{label}", fillcolor="{NODE_GRAY}"];')
            first, second = (left, right)
            for child in (first, second):
                text = "예" if child == yes_child else "아니요"
                lines.append(f'n{i} -> n{child} [label="{text}"];')
    lines.append("}")
    return "\n".join(lines)


st.markdown(
    """
    <style>
    .acc-card {
        border: 1px solid rgba(128, 128, 128, 0.35);
        border-radius: 12px;
        padding: 1rem 0.6rem;
        text-align: center;
        background: rgba(128, 128, 128, 0.07);
        height: 100%;
    }
    .acc-title { font-size: 0.9rem; font-weight: 600; min-height: 3.6em; }
    .acc-big { font-size: 2.8rem; font-weight: 700; line-height: 1.2; }
    .acc-big-label { font-size: 0.85rem; opacity: 0.7; margin-bottom: 0.6rem; }
    .acc-small-row { display: flex; justify-content: space-around; font-size: 0.8rem; }
    .acc-small-row span { opacity: 0.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 분류 모델")
st.caption("stroke가 1이면 뇌졸중, 0이면 아님입니다. 뇌졸중을 양성으로 둡니다.")

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------------------------------------------------------------
# 1. 입력 속성 고르기
# ---------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=FEATURES,
    default=DEFAULT_FEATURES,
    format_func=LABELS.get,
)
selected = [f for f in FEATURES if f in chosen]  # 항상 같은 순서로 맞춘다

if len(selected) < 2:
    st.info("속성을 두 개 이상 골라 주세요. 그림을 그리려면 최소 두 개가 필요합니다.")
    st.stop()

# ---------------------------------------------------------------
# 2. 훈련용 / 테스트용 나누기: 번호 순으로 정렬해 열 명씩 묶고, 앞 세 명이 테스트용
# ---------------------------------------------------------------
data = raw.sort_values("id").reset_index(drop=True)
is_test = (data.index % 10) < 3
train = data[~is_test].copy()
test = data[is_test].copy()

bmi_median = None
if "bmi" in selected:  # 체질량지수를 고른 경우에만, 훈련용의 중앙값으로 채운다
    bmi_median = float(train["bmi"].median())
    train["bmi"] = train["bmi"].fillna(bmi_median)
    test["bmi"] = test["bmi"].fillna(bmi_median)

X_train, y_train = train[selected], train["stroke"]
X_test, y_test = test[selected], test["stroke"]

st.caption(f"훈련용 {len(train):,}명 · 테스트용 {len(test):,}명 (열 명씩 묶어 앞 세 명을 테스트용으로 고정)")
if bmi_median is not None:
    st.caption(f"체질량지수는 비어 있는 사람이 있어 훈련용의 중앙값 {bmi_median:.1f}로 채웠습니다.")

# ---------------------------------------------------------------
# 3. 모델 만들기 (크기 맞추기와 학습은 훈련용으로만)
# ---------------------------------------------------------------
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logit.fit(X_train, y_train)

tree = DecisionTreeClassifier(max_depth=3, min_samples_split=5, random_state=0)
tree.fit(X_train, y_train)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

results = [
    ("로지스틱 회귀(확률로 답하는 모델)", logit),
    ("의사결정트리(질문으로 답하는 모델)", tree),
    ("기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)", baseline),
]

st.header("정확도")
for col, (name, model) in zip(st.columns(3), results):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    col.markdown(accuracy_card(name, train_acc, test_acc), unsafe_allow_html=True)

st.caption(
    "로지스틱 회귀는 훈련용으로만 크기 맞추기(StandardScaler)를 한 뒤 학습합니다. "
    "의사결정트리는 질문의 기준을 원래 값으로 읽을 수 있도록 크기 맞추기 없이 학습합니다(나누는 칸은 같습니다). "
    "의사결정트리는 질문을 최대 3번(max_depth=3)까지만 하고, 사람이 5명 미만인 마디는 더 나누지 않으며"
    "(min_samples_split=5), 난수는 고정(random_state=0)했습니다. "
    "뇌졸중인 사람이 드물기 때문에, 기준 모델의 정확도와 견주어 보세요."
)

# ---------------------------------------------------------------
# 4. 두 속성으로 그리는 산점도
# ---------------------------------------------------------------
st.header("테스트 데이터 산점도와 모델이 가르는 경계")

c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", selected, format_func=LABELS.get)
y_feat = c2.selectbox(
    "세로축", [f for f in selected if f != x_feat], format_func=LABELS.get
)

x_lo, x_hi = axis_range(x_feat, X_test[x_feat])
y_lo, y_hi = axis_range(y_feat, X_test[y_feat])
xs = linspace(x_lo, x_hi, GRID_N)
ys = linspace(y_lo, y_hi, GRID_N)

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다
fixed = {f: float(X_test[f].median()) for f in selected if f not in (x_feat, y_feat)}

grid = pd.MultiIndex.from_product([xs, ys], names=[x_feat, y_feat]).to_frame(index=False)
for f, value in fixed.items():
    grid[f] = value
grid = grid[selected]

pos_index = list(logit.classes_).index(POSITIVE)
grid["p"] = logit.predict_proba(grid[selected])[:, pos_index]  # 뇌졸중일 확률
grid["tree"] = tree.predict(grid[selected])

p_table = grid.pivot(index=y_feat, columns=x_feat, values="p")
tree_table = grid.pivot(index=y_feat, columns=x_feat, values="tree")
p_min, p_max = float(grid["p"].min()), float(grid["p"].max())
line_visible = p_min < 0.5 < p_max

fig = go.Figure()

# 의사결정트리가 나눈 칸 (옅은 색)
fig.add_trace(
    go.Heatmap(
        x=list(tree_table.columns),
        y=list(tree_table.index),
        z=tree_table.values.tolist(),
        zmin=0,
        zmax=1,
        colorscale=[[0, LIGHT_BLUE], [1, LIGHT_RED]],
        showscale=False,
        hoverinfo="skip",
        showlegend=False,
    )
)

# 로지스틱 회귀가 0.5로 가르는 선
if line_visible:
    fig.add_trace(
        go.Contour(
            x=list(p_table.columns),
            y=list(p_table.index),
            z=p_table.values.tolist(),
            contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
            line=dict(color="black", width=3),
            colorscale=[[0, "black"], [1, "black"]],
            showscale=False,
            hoverinfo="skip",
            name="로지스틱 회귀의 0.5 경계선",
            showlegend=True,
        )
    )

# 범례용: 의사결정트리 칸 색
for color, name in [
    (LIGHT_BLUE, "의사결정트리: '아님'이라고 답하는 칸"),
    (LIGHT_RED, "의사결정트리: '뇌졸중'이라고 답하는 칸"),
]:
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=14, color=color, symbol="square", line=dict(width=1, color="#999")),
            name=name,
        )
    )

# 테스트 데이터 점 (0/1 속성은 겹치지 않게 살짝 흩어 찍는다)
plot_df = test.copy()
plot_df["_x"] = plot_df[x_feat].astype(float)
plot_df["_y"] = plot_df[y_feat].astype(float)
jittered = False
if x_feat in BINARY:
    plot_df["_x"] += ((plot_df["id"] * 0.6180339887) % 1 - 0.5) * 0.16
    jittered = True
if y_feat in BINARY:
    plot_df["_y"] += ((plot_df["id"] * 0.7548776662) % 1 - 0.5) * 0.16
    jittered = True

for value, name, color in [(0, "아님(실제)", BLUE), (1, "뇌졸중(실제)", RED)]:
    part = plot_df[plot_df["stroke"] == value]
    fig.add_trace(
        go.Scatter(
            x=part["_x"],
            y=part["_y"],
            mode="markers",
            name=name,
            marker=dict(size=6, color=color, opacity=0.75, line=dict(width=0.5, color="white")),
        )
    )

x_axis = dict(title=LABELS[x_feat], range=[x_lo, x_hi])
y_axis = dict(title=LABELS[y_feat], range=[y_lo, y_hi])
if x_feat in BINARY:
    x_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
if y_feat in BINARY:
    y_axis.update(tickvals=[0, 1], ticktext=["없음", "있음"])
fig.update_layout(
    xaxis=x_axis,
    yaxis=y_axis,
    height=600,
    legend=dict(orientation="h", y=-0.2),
    margin=dict(t=30),
)
st.plotly_chart(fig)

if jittered:
    st.caption("0/1 속성 축에서는 점이 겹치지 않게 살짝 흩어 찍었습니다. 경계선과 칸 계산에는 영향이 없습니다.")

if fixed:
    parts = []
    for f, value in fixed.items():
        if f in BINARY:
            parts.append(f"{LABELS[f]} = {'있음' if value >= 0.5 else '없음'}({fmt_num(value)})")
        else:
            parts.append(f"{LABELS[f]} = {fmt_num(value)}")
    st.write("두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + ", ".join(parts))
else:
    st.write("고른 속성이 두 개뿐이라, 따로 중앙값에 세워 둔 속성은 없습니다.")

if line_visible:
    st.write("로지스틱 회귀가 뇌졸중일 확률 0.5로 가르는 선을 검은 선으로 그렸습니다.")
else:
    side = "작아" if p_max <= 0.5 else "커서"
    answer = "아님" if p_max <= 0.5 else "뇌졸중"
    st.write(
        f"로지스틱 회귀의 0.5 경계선은 이 그림 범위 밖에 있어 그리지 못했습니다. "
        f"그림 안에서는 뇌졸중일 확률이 {p_min * 100:.1f}%~{p_max * 100:.1f}%로 모두 0.5보다 {side}, "
        f"어디에 찍어도 '{answer}'이라고 답합니다."
    )

# ---------------------------------------------------------------
# 5. 의사결정트리가 던진 질문
# ---------------------------------------------------------------
st.header("의사결정트리(질문으로 답하는 모델)가 던진 질문")
st.graphviz_chart(build_dot(tree, selected))
st.caption("각 마디에는 그 자리에 온 훈련용 사람 수, 그중 실제 뇌졸중인 사람 수와 비율을 적었습니다. 파란 마디는 '아님', 빨간 마디는 '뇌졸중'이라고 답하는 마디입니다.")

t = tree.tree_
leaf_ids = [i for i in range(t.node_count) if int(t.children_left[i]) == -1]
no_leaves = [i for i in leaf_ids if int(tree.classes_[int(t.value[i][0].argmax())]) == 0]
st.write(
    f"답을 내는 마디는 모두 {len(leaf_ids)}칸이고, 그중 {len(no_leaves)}칸이 '아님'이라고 답합니다."
)

asked_idx = {int(t.feature[i]) for i in range(t.node_count) if int(t.children_left[i]) != -1}
asked = [LABELS[f] for k, f in enumerate(selected) if k in asked_idx]
not_asked = [LABELS[f] for k, f in enumerate(selected) if k not in asked_idx]
if asked:
    line = f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 {', '.join(asked)}입니다."
    if not_asked:
        line += f" (묻지 않은 것: {', '.join(not_asked)})"
else:
    line = f"고른 속성 {len(selected)}개 가운데 이 나무가 실제로 물은 것은 없습니다."
st.write(line)
