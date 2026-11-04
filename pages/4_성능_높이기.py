import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="뇌졸중 예측 실습실 - 성능 높이기", page_icon="⚖️")

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
NOT_AVAILABLE = "계산할 수 없음"

LOGIT_NAME = "로지스틱 회귀(확률로 답하는 모델)"
TREE_NAME = "의사결정트리(질문으로 답하는 모델)"

BLUE = "#1f77b4"
RED = "#d62728"
BEFORE_COLOR = "#ff7f0e"  # 가중치를 맞추기 전 경계선
AFTER_COLOR = "#2ca02c"  # 가중치를 맞춘 뒤 경계선

LINE_N = 400
PLANE_N = 40


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL, encoding="utf-8")


def fmt_num(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def linspace(lo: float, hi: float, n: int) -> list:
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def axis_range(feature: str, series: pd.Series) -> tuple:
    if feature in BINARY:
        return -0.5, 1.5
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.05
    return lo - pad, hi + pad


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


def before_after_table(before: dict, after: dict) -> pd.DataFrame:
    """네 칸과 네 지표를 가중치를 맞추기 전·후로 나란히 놓은 표."""
    mb, ma = metrics_from(before), metrics_from(after)
    cells = [
        ("찾아낸 환자 (TP)", "TP"),
        ("놓친 환자 (FN)", "FN"),
        ("헛짚은 사람 (FP)", "FP"),
        ("맞게 '아님' (TN)", "TN"),
    ]
    rows = [
        {"항목": label, "가중치 맞추기 전": f"{before[k]:,}명", "가중치 맞춘 뒤": f"{after[k]:,}명"}
        for label, k in cells
    ]
    for k in ["정확도", "재현율", "정밀도", "F1"]:
        rows.append({"항목": k, "가중치 맞추기 전": pct(mb[k]), "가중치 맞춘 뒤": pct(ma[k])})
    return pd.DataFrame(rows)


def raw_logit_coefficients(pipe, features: list) -> tuple:
    """크기 맞추기까지 포함한 로지스틱 회귀를 원래 값 기준의 식 w·x + b 로 바꾼다.
    w·x + b = 0 인 곳이 뇌졸중일 확률 0.5로 가르는 선(면)이다."""
    scaler = pipe.named_steps["standardscaler"]
    clf = pipe.named_steps["logisticregression"]
    w = {}
    b = float(clf.intercept_[0])
    for f, coef, mean, scale in zip(features, clf.coef_[0], scaler.mean_, scaler.scale_):
        w[f] = float(coef) / float(scale)
        b -= float(coef) * float(mean) / float(scale)
    return w, b


def boundary_line(w: dict, c: float, x_feat: str, y_feat: str, x_rng: tuple, y_rng: tuple):
    """두 축 위에서 w_x·x + w_y·y + c = 0 인 선의 점들. 그림 안에 없는 부분은 None."""
    wx, wy = w[x_feat], w[y_feat]
    x_span, y_span = x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]
    if wx == 0 and wy == 0:
        return None
    xs, ys = [], []
    if abs(wy) * y_span >= abs(wx) * x_span:  # y를 x로 풀어 쓴다
        for x in linspace(x_rng[0], x_rng[1], LINE_N):
            y = -(wx * x + c) / wy
            inside = y_rng[0] <= y <= y_rng[1]
            xs.append(x if inside else None)
            ys.append(y if inside else None)
    else:  # x를 y로 풀어 쓴다
        for y in linspace(y_rng[0], y_rng[1], LINE_N):
            x = -(wy * y + c) / wx
            inside = x_rng[0] <= x <= x_rng[1]
            xs.append(x if inside else None)
            ys.append(y if inside else None)
    if all(v is None for v in xs):
        return None
    return xs, ys


def boundary_plane(w: dict, c: float, axes: list, ranges: dict):
    """세 축 위에서 w·x + c = 0 인 면. 그림 안에 없는 부분은 NaN. 면이 없으면 None."""
    weights = [w[f] for f in axes]
    spans = [ranges[f][1] - ranges[f][0] for f in axes]
    scores = [abs(wk) * sk for wk, sk in zip(weights, spans)]
    if max(scores) == 0:
        return None
    d = scores.index(max(scores))  # 이 축을 나머지 두 축으로 풀어 쓴다
    a, b = [k for k in range(3) if k != d]
    grid_a = linspace(ranges[axes[a]][0], ranges[axes[a]][1], PLANE_N)
    grid_b = linspace(ranges[axes[b]][0], ranges[axes[b]][1], PLANE_N)
    lo, hi = ranges[axes[d]]
    nan = float("nan")

    A = [[ga for ga in grid_a] for _ in grid_b]
    B = [[gb for _ in grid_a] for gb in grid_b]
    D = []
    for gb in grid_b:
        row = []
        for ga in grid_a:
            v = -(weights[a] * ga + weights[b] * gb + c) / weights[d]
            row.append(v if lo <= v <= hi else nan)
        D.append(row)
    if all(v != v for row in D for v in row):  # 모두 NaN
        return None

    coords = {a: A, b: B, d: D}
    return coords[0], coords[1], coords[2]


def jitter(series_id: pd.Series, offset: float) -> pd.Series:
    """0/1 속성 축에서 점이 겹치지 않도록 번호로 정해지는 작은 흔들림."""
    return ((series_id * offset) % 1 - 0.5) * 0.16


def fixed_text(fixed: dict) -> str:
    parts = []
    for f, value in fixed.items():
        if f in BINARY:
            parts.append(f"{LABELS[f]} = {'있음' if value >= 0.5 else '없음'}({fmt_num(value)})")
        else:
            parts.append(f"{LABELS[f]} = {fmt_num(value)}")
    return ", ".join(parts)


def tick_options(feature: str) -> dict:
    return dict(tickvals=[0, 1], ticktext=["없음", "있음"]) if feature in BINARY else {}


# ---------------------------------------------------------------
# 화면 맨 위: 입력 속성 고르기
# ---------------------------------------------------------------
st.title("⚖️ 성능 높이기")

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

st.caption(
    "뇌졸중(stroke = 1)을 양성으로 둡니다. 뇌졸중을 겪은 사람이 아주 드물어서, "
    "양쪽의 가중치를 맞추기 전과 맞춘 뒤의 결과를 나란히 비교합니다."
)

try:
    raw = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요. ({e})")
    st.stop()

# ---------------------------------------------------------------
# 1. '채점표' 페이지와 똑같이 나누기
#    번호 순으로 정렬해 열 명씩 묶고, 각 묶음의 앞 세 명을 테스트용으로 고정
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

n_test = len(test)
n_test_pos = int((y_test == POSITIVE).sum())
n_train = len(train)
n_train_pos = int((y_train == POSITIVE).sum())
n_train_neg = n_train - n_train_pos

st.write(
    f"훈련용 {n_train:,}명(뇌졸중 {n_train_pos:,}명) · "
    f"테스트용 {n_test:,}명(뇌졸중 {n_test_pos:,}명)"
)
if bmi_median is not None:
    st.caption(f"체질량지수는 비어 있는 사람이 있어 훈련용의 중앙값 {bmi_median:.1f}로 채웠습니다.")

# ---------------------------------------------------------------
# 2. 가중치를 맞추기 전·후 모델 학습
#    class_weight="balanced": 드문 쪽의 한 명에 더 큰 가중치를 주어 양쪽의 합이 같아지게 한다
# ---------------------------------------------------------------
TREE_ARGS = dict(max_depth=3, min_samples_split=5, random_state=0)


def make_logit(balanced: bool):
    clf = LogisticRegression(max_iter=1000, class_weight="balanced" if balanced else None)
    return make_pipeline(StandardScaler(), clf)


def make_tree(balanced: bool):
    return DecisionTreeClassifier(class_weight="balanced" if balanced else None, **TREE_ARGS)


models = {
    LOGIT_NAME: (make_logit(False).fit(X_train, y_train), make_logit(True).fit(X_train, y_train)),
    TREE_NAME: (make_tree(False).fit(X_train, y_train), make_tree(True).fit(X_train, y_train)),
}

weight_neg = n_train / (2 * n_train_neg)
weight_pos = n_train / (2 * n_train_pos) if n_train_pos else float("nan")

# ---------------------------------------------------------------
# 3. 전·후 비교 표
# ---------------------------------------------------------------
st.header("가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    f"가중치를 맞추면 훈련용에서 '아님' 한 명은 {weight_neg:.2f}, 뇌졸중 한 명은 {weight_pos:.2f}로 셉니다. "
    "그러면 뇌졸중인 사람 전체와 아닌 사람 전체의 무게가 같아집니다. 채점은 둘 다 같은 테스트 데이터로 했습니다."
)

cols = st.columns(2)
for col, (name, (before_model, after_model)) in zip(cols, models.items()):
    before = confusion_counts(y_test, before_model.predict(X_test))
    after = confusion_counts(y_test, after_model.predict(X_test))
    with col:
        st.subheader(name)
        st.dataframe(before_after_table(before, after), hide_index=True)
        st.write(
            f"찾아낸 환자 수는 {before['TP']:,}명에서 {after['TP']:,}명으로, "
            f"놓친 환자 수는 {before['FN']:,}명에서 {after['FN']:,}명으로, "
            f"헛짚은 사람 수는 {before['FP']:,}명에서 {after['FP']:,}명으로 바뀌었습니다."
        )

st.caption(
    "찾아낸 환자(TP): 뇌졸중인데 뇌졸중이라고 예측 · 놓친 환자(FN): 뇌졸중인데 아니라고 예측 · "
    "헛짚은 사람(FP): 아닌데 뇌졸중이라고 예측 · 재현율 = TP ÷ (TP + FN) · 정밀도 = TP ÷ (TP + FP) · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). 뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없습니다."
)

# ---------------------------------------------------------------
# 4. 두 속성으로 그리는 산점도 + 로지스틱 회귀의 전·후 0.5 경계선
# ---------------------------------------------------------------
logit_before, logit_after = models[LOGIT_NAME]
coef_before = raw_logit_coefficients(logit_before, selected)
coef_after = raw_logit_coefficients(logit_after, selected)

st.header("테스트 데이터 산점도와 0.5 경계선")
st.caption(f"{LOGIT_NAME}이 뇌졸중일 확률을 0.5로 가르는 선을, 가중치를 맞추기 전과 맞춘 뒤로 겹쳐 그립니다.")

c1, c2 = st.columns(2)
x_feat = c1.selectbox("가로축", selected, format_func=LABELS.get)
y_feat = c2.selectbox("세로축", [f for f in selected if f != x_feat], format_func=LABELS.get)

x_rng = axis_range(x_feat, X_test[x_feat])
y_rng = axis_range(y_feat, X_test[y_feat])

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 둔다
fixed2 = {f: float(X_test[f].median()) for f in selected if f not in (x_feat, y_feat)}

fig2 = go.Figure()

plot_df = test.copy()
plot_df["_x"] = plot_df[x_feat].astype(float)
plot_df["_y"] = plot_df[y_feat].astype(float)
jittered2 = False
if x_feat in BINARY:
    plot_df["_x"] += jitter(plot_df["id"], 0.6180339887)
    jittered2 = True
if y_feat in BINARY:
    plot_df["_y"] += jitter(plot_df["id"], 0.7548776662)
    jittered2 = True

for value, name, color in [(0, "아님(실제)", BLUE), (1, "뇌졸중(실제)", RED)]:
    part = plot_df[plot_df["stroke"] == value]
    fig2.add_trace(
        go.Scatter(
            x=part["_x"],
            y=part["_y"],
            mode="markers",
            name=name,
            marker=dict(size=6, color=color, opacity=0.6, line=dict(width=0.5, color="white")),
        )
    )

outside_notes = []
for label, (w, b), color, dash in [
    ("가중치 맞추기 전 0.5 경계선", coef_before, BEFORE_COLOR, "dash"),
    ("가중치 맞춘 뒤 0.5 경계선", coef_after, AFTER_COLOR, "solid"),
]:
    c = b + sum(w[f] * v for f, v in fixed2.items())
    line = boundary_line(w, c, x_feat, y_feat, x_rng, y_rng)
    if line is None:
        outside_notes.append(label)
        fig2.add_trace(  # 그리지 못해도 범례에는 남겨 둔다
            go.Scatter(x=[None], y=[None], mode="lines", name=f"{label} (그림 밖)", line=dict(color=color, width=4, dash=dash))
        )
    else:
        fig2.add_trace(
            go.Scatter(
                x=line[0],
                y=line[1],
                mode="lines",
                name=label,
                line=dict(color=color, width=4, dash=dash),
                connectgaps=False,
            )
        )

fig2.update_layout(
    xaxis=dict(title=LABELS[x_feat], range=list(x_rng), **tick_options(x_feat)),
    yaxis=dict(title=LABELS[y_feat], range=list(y_rng), **tick_options(y_feat)),
    height=600,
    legend=dict(orientation="h", y=-0.2),
    margin=dict(t=30),
)
st.plotly_chart(fig2)

if jittered2:
    st.caption("0/1 속성 축에서는 점이 겹치지 않게 살짝 흩어 찍었습니다. 경계선 계산에는 영향이 없습니다.")

if fixed2:
    st.write("두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + fixed_text(fixed2))
else:
    st.write("고른 속성이 두 개뿐이라, 따로 중앙값에 세워 둔 속성은 없습니다.")

if outside_notes:
    for label in outside_notes:
        st.write(f"{label}은 이 그림 범위 밖에 있어 그리지 못했습니다.")
else:
    st.write("주황색 점선이 가중치를 맞추기 전, 초록색 실선이 가중치를 맞춘 뒤의 0.5 경계선입니다.")

# ---------------------------------------------------------------
# 5. 세 속성 이상이면 3차원 산점도 + 전·후 0.5 평면
# ---------------------------------------------------------------
st.header("3차원 산점도와 0.5 평면")

if len(selected) < 3:
    st.info("속성을 세 개 이상 고르면 마우스로 돌려 볼 수 있는 3차원 그림이 나타납니다.")
else:
    st.caption(
        f"{LOGIT_NAME}이 뇌졸중일 확률을 0.5로 가르는 면을 가중치를 맞추기 전(주황)과 맞춘 뒤(초록)로 겹쳐 그립니다. "
        "그림을 마우스로 끌어 돌려 볼 수 있습니다."
    )
    d1, d2, d3 = st.columns(3)
    x3 = d1.selectbox("3차원 가로축", selected, index=0, format_func=LABELS.get, key="x3")
    y3 = d2.selectbox(
        "3차원 세로축", [f for f in selected if f != x3], index=0, format_func=LABELS.get, key="y3"
    )
    z3 = d3.selectbox(
        "3차원 높이축",
        [f for f in selected if f not in (x3, y3)],
        index=0,
        format_func=LABELS.get,
        key="z3",
    )
    axes3 = [x3, y3, z3]
    ranges3 = {f: axis_range(f, X_test[f]) for f in axes3}
    fixed3 = {f: float(X_test[f].median()) for f in selected if f not in axes3}

    fig3 = go.Figure()

    pts = test.copy()
    jittered3 = False
    for k, (f, off) in enumerate(zip(axes3, (0.6180339887, 0.7548776662, 0.5698402910))):
        pts[f"_a{k}"] = pts[f].astype(float)
        if f in BINARY:
            pts[f"_a{k}"] += jitter(pts["id"], off)
            jittered3 = True

    for value, name, color in [(0, "아님(실제)", BLUE), (1, "뇌졸중(실제)", RED)]:
        part = pts[pts["stroke"] == value]
        fig3.add_trace(
            go.Scatter3d(
                x=part["_a0"],
                y=part["_a1"],
                z=part["_a2"],
                mode="markers",
                name=name,
                marker=dict(size=3, color=color, opacity=0.7),
            )
        )

    outside3 = []
    for label, (w, b), color in [
        ("가중치 맞추기 전 0.5 평면", coef_before, BEFORE_COLOR),
        ("가중치 맞춘 뒤 0.5 평면", coef_after, AFTER_COLOR),
    ]:
        c = b + sum(w[f] * v for f, v in fixed3.items())
        plane = boundary_plane(w, c, axes3, ranges3)
        if plane is None:
            outside3.append(label)
        else:
            fig3.add_trace(
                go.Surface(
                    x=plane[0],
                    y=plane[1],
                    z=plane[2],
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.45,
                    hoverinfo="skip",
                )
            )
        # 범례용 표식 (면은 범례에 따로 나오지 않으므로 같은 색의 사각형을 둔다)
        fig3.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                name=label + (" (그림 밖)" if plane is None else ""),
                marker=dict(size=8, color=color, symbol="square"),
            )
        )

    scene = {}
    for key, f in zip(("xaxis", "yaxis", "zaxis"), axes3):
        scene[key] = dict(title=LABELS[f], range=list(ranges3[f]), **tick_options(f))
    fig3.update_layout(
        scene=scene,
        height=700,
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=30, l=0, r=0, b=0),
    )
    st.plotly_chart(fig3)

    if jittered3:
        st.caption("0/1 속성 축에서는 점이 겹치지 않게 살짝 흩어 찍었습니다. 평면 계산에는 영향이 없습니다.")
    if fixed3:
        st.write("세 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + fixed_text(fixed3))
    else:
        st.write("고른 속성이 세 개뿐이라, 따로 중앙값에 세워 둔 속성은 없습니다.")
    for label in outside3:
        st.write(f"{label}은 이 그림 범위 밖에 있어 그리지 못했습니다.")
