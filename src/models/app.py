import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from typing import Dict, Any

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Scoring Demo")
st.caption("Модель: RandomForest + препроцессинг (StandardScaler + OneHotEncoder). Метки: 0=bad, 1=good")

MODEL_PATH = "src/models/rf_model.pkl"

FEATURES = [
    "age", "sex", "job", "housing", "saving_accounts",
    "checking_account", "credit_amount", "duration", "purpose",
]

JOB_MAP = {
    "Нет работы, нет квалификации": 0,
    "Есть работа, нет квалификации": 1,
    "Есть работа, есть квалификация": 2,
    "Высшая квалификация / Руководитель": 3,
}
HOUSING_MAP = {"Личное": "own", "Аренда": "rent", "Бесплатное проживание": "free"}
SAVING_MAP = {
    "Нет данных": "no_info", "Низкий уровень": "little",
    "Умеренный уровень": "moderate", "Выше среднего": "quite rich", "Высокий уровень": "rich",
}
CHECKING_MAP = {
    "Нет данных": "no_info", "Низкий уровень": "little",
    "Умеренный уровень": "moderate", "Высокий уровень": "rich",
}
PURPOSE_MAP = {
    "Аудио-/Видеотехника": "radio/TV", "Образование": "education",
    "Мебель/оборудование": "furniture/equipment", "Автокредит": "car",
    "Бизнес": "business", "Ремонт": "repairs",
    "Бытовая техника": "domestic appliances", "Отдых/Другое...": "vacation/others",
}
SEX_VALUES = ["Мужчина", "Женщина"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
CLASSES = np.array(getattr(model, "classes_", [0, 1]))
IDX_BAD = int(np.where(CLASSES == 0)[0][0])
IDX_GOOD = int(np.where(CLASSES == 1)[0][0])

def ru_to_model_values(row_ru: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row_ru)
    row["job"] = JOB_MAP[row["job"]]
    row["housing"] = HOUSING_MAP[row["housing"]]
    row["saving_accounts"] = SAVING_MAP[row["saving_accounts"]]
    row["checking_account"] = CHECKING_MAP[row["checking_account"]]
    row["purpose"] = PURPOSE_MAP[row["purpose"]]
    return row

def coerce_and_order_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURES]

def predict_with_threshold(df_model: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    proba = model.predict_proba(df_model)
    pred = model.predict(df_model)
    proba_bad = proba[:, IDX_BAD]
    proba_good = proba[:, IDX_GOOD]
    pred_thr = (proba_good >= threshold).astype(int)
    return pd.DataFrame({
        "pred": pred,
        "proba_bad": proba_bad,
        "proba_good": proba_good,
        "pred_thr": pred_thr,
    })

with st.sidebar:
    thr = st.slider("Порог для одобрения кредита (класс 1)", 0.0, 1.0, 0.5, 0.01)
    st.caption("Если вероятность ≥ порога, решение = ОДОБРИТЬ, иначе = ОТКЛОНИТЬ.")

tab1, tab2 = st.tabs(["🧍 Single", "📄 Batch CSV"])

with tab1:
    col1, col2 = st.columns(2)
    ui_values = {}
    with col1:
        ui_values["age"] = st.number_input("Возраст", 18, 90, 30)
        ui_values["job"] = st.selectbox("Работа/Квалификация", list(JOB_MAP.keys()), 1)
        ui_values["credit_amount"] = st.number_input("Сумма кредита", 100, 200000, 3000)
        ui_values["duration"] = st.slider("Длительность (мес.)", 4, 72, 24)
        ui_values["sex"] = st.selectbox("Пол", SEX_VALUES, 0)
    with col2:
        ui_values["housing"] = st.selectbox("Жильё", list(HOUSING_MAP.keys()), 0)
        ui_values["saving_accounts"] = st.selectbox("Сбережения", list(SAVING_MAP.keys()), 0)
        ui_values["checking_account"] = st.selectbox("Текущий счёт", list(CHECKING_MAP.keys()), 0)
        ui_values["purpose"] = st.selectbox("Цель кредита", list(PURPOSE_MAP.keys()), 0)

    if st.button("АНАЛИЗ", use_container_width=True):
        try:
            model_row = ru_to_model_values(ui_values)
            df_in = coerce_and_order_df(pd.DataFrame([model_row]))
            res = predict_with_threshold(df_in, threshold=thr).iloc[0]

            base_pred_text = "НАДЁЖЕН (good, класс 1)" if int(res["pred"]) == 1 else "НЕНАДЁЖЕН (bad, класс 0)"
            thr_pred_text = "НАДЁЖЕН (good, класс 1)" if int(res["pred_thr"]) == 1 else "НЕНАДЁЖЕН (bad, класс 0)"

            if "НЕНАДЁЖЕН" in base_pred_text:
                st.error(f"Базовое предсказание модели (argmax): **{base_pred_text}**")
            else:
                st.success(f"Базовое предсказание модели (argmax): **{base_pred_text}**")

            if "НЕНАДЁЖЕН" in thr_pred_text:
                st.error(f"Предсказание по порогу {thr:.2f}: **{thr_pred_text}**")
            else:
                st.info(f"Предсказание по порогу {thr:.2f}: **{thr_pred_text}**")

            st.markdown("**Вероятности классов:**")
            st.json({
                "Вероятность дефолта (bad=0)": float(res["proba_bad"]),
                "Вероятность погашения (good=1)": float(res["proba_good"]),
            })
        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")

with tab2:
    upl = st.file_uploader("Загрузите CSV", type=["csv"])
    if upl is not None:
        try:
            df = pd.read_csv(upl)
            df = coerce_and_order_df(df)
            out = predict_with_threshold(df, threshold=thr)
            result = pd.concat([df.reset_index(drop=True), out], axis=1)
            st.dataframe(result.head(50), use_container_width=True)
            buf = io.BytesIO()
            result.to_csv(buf, index=False)
            st.download_button("⬇️ Скачать predictions.csv", buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка обработки файла: {e}")
