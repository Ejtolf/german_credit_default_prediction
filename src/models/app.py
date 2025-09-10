# app.py
import io
import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# =========================
# Конфиг страницы
# =========================
st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")
st.title("💳 Credit Risk Scoring Demo")
st.caption("Модель: RandomForest + препроцессинг (StandardScaler + OneHotEncoder). Метки: 0=bad, 1=good")

MODEL_PATH = "src/models/rf_model.pkl"

# Признаки в том порядке, в котором ожидались при обучении пайплайна
FEATURES = [
    "age",
    "sex",
    "job",
    "housing",
    "saving_accounts",
    "checking_account",
    "credit_amount",
    "duration",
    "purpose",
]

# =========================
# Справочники мэппинга (RU -> значения обучения)
# =========================
JOB_MAP = {
    "Нет работы, нет квалификации": 0,
    "Есть работа, нет квалификации": 1,
    "Есть работа, есть квалификация": 2,
    "Высшая квалификация / Руководитель": 3,
}

HOUSING_MAP = {
    "Личное": "own",
    "Аренда": "rent",
    "Бесплатное проживание": "free",
}

SAVING_MAP = {
    "Нет данных": "no_info",
    "Низкий уровень": "little",
    "Умеренный уровень": "moderate",
    "Выше среднего": "quite rich",
    "Высокий уровень": "rich",
}

CHECKING_MAP = {
    "Нет данных": "no_info",
    "Низкий уровень": "little",
    "Умеренный уровень": "moderate",
    "Высокий уровень": "rich",
}

PURPOSE_MAP = {
    "Аудио-/Видеотехника": "radio/TV",
    "Образование": "education",
    "Мебель/оборудование": "furniture/equipment",
    "Автокредит": "car",
    "Бизнес": "business",
    "Ремонт": "repairs",
    "Бытовая техника": "domestic appliances",
    "Отдых/Другое...": "vacation/others",
}

SEX_VALUES = ["Мужчина", "Женщина"]  # как обучалась модель

# =========================
# Кэшированная загрузка модели
# =========================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

# Уточняем порядок классов из модели (на случай отличий)
CLASSES = np.array(getattr(model, "classes_", [0, 1]))
IDX_BAD = int(np.where(CLASSES == 0)[0][0])   # класс 0 = bad
IDX_GOOD = int(np.where(CLASSES == 1)[0][0])  # класс 1 = good

# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================
def ru_to_model_values(row_ru: Dict[str, Any]) -> Dict[str, Any]:
    """Конвертирует RU-подписи в значения, на которых обучалась модель."""
    row = dict(row_ru)
    row["job"] = JOB_MAP[row["job"]]
    row["housing"] = HOUSING_MAP[row["housing"]]
    row["saving_accounts"] = SAVING_MAP[row["saving_accounts"]]
    row["checking_account"] = CHECKING_MAP[row["checking_account"]]
    row["purpose"] = PURPOSE_MAP[row["purpose"]]
    # sex уже в нужном виде
    return row

def coerce_and_order_df(df: pd.DataFrame) -> pd.DataFrame:
    """Гарантирует порядок и наличие всех признаков."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Не найдены обязательные колонки: {missing}")
    return df[FEATURES]

def predict_with_threshold(df_model: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Возвращает pred (по argmax), proba_bad/proba_good, а также pred_thr по кастомному порогу для 'good'."""
    proba = model.predict_proba(df_model)
    pred = model.predict(df_model)

    proba_bad = proba[:, IDX_BAD]
    proba_good = proba[:, IDX_GOOD]

    # Кастомный порог: относим к good (1), если proba_good >= threshold, иначе bad (0)
    pred_thr = (proba_good >= threshold).astype(int)

    out = pd.DataFrame({
        "pred": pred,  # 0/1
        "proba_bad": proba_bad,
        "proba_good": proba_good,
        "pred_thr": pred_thr,
    })
    return out

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("Параметры")
    thr = st.slider("Порог для одобрения кредита (класс 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    st.caption("Если вероятность одобрения ≥ порога, решение по порогу = ОДОБРИТЬ, иначе = ОТКЛОНИТЬ.")
    st.divider()
    st.markdown("**Ожидаемые признаки:**")
    st.code(", ".join(FEATURES), language="text")

# =========================
# ТАБЫ
# =========================
tab1, tab2 = st.tabs(["🧍 Single", "📄 Batch CSV"])

# =========================
# TAB 1: Single
# =========================
with tab1:
    st.subheader("Единичное предсказание")

    col1, col2 = st.columns(2)
    ui_values = {}

    with col1:
        ui_values["age"] = st.number_input("Возраст", min_value=18, max_value=90, value=30)
        ui_values["job"] = st.selectbox(
            "Работа/Квалификация",
            list(JOB_MAP.keys()),
            index=1
        )
        ui_values["credit_amount"] = st.number_input("Сумма кредита", min_value=100, max_value=200000, value=3000)
        ui_values["duration"] = st.slider("Длительность (мес.)", 4, 72, 24)
        ui_values["sex"] = st.selectbox("Пол", SEX_VALUES, index=0)

    with col2:
        ui_values["housing"] = st.selectbox("Жильё", list(HOUSING_MAP.keys()), index=0)
        ui_values["saving_accounts"] = st.selectbox("Сбережения", list(SAVING_MAP.keys()), index=0)
        ui_values["checking_account"] = st.selectbox("Текущий счёт", list(CHECKING_MAP.keys()), index=0)
        ui_values["purpose"] = st.selectbox("Цель кредита", list(PURPOSE_MAP.keys()), index=0)

    if st.button("АНАЛИЗ", use_container_width=True):
        try:
            # RU -> значения обучения
            model_row = ru_to_model_values(ui_values)
            df_in = pd.DataFrame([model_row])
            df_in = coerce_and_order_df(df_in)

            res = predict_with_threshold(df_in, threshold=thr).iloc[0]

            base_pred_text = "НАДЁЖЕН (good, класс 1)" if int(res["pred"]) == 1 else "НЕНАДЁЖЕН (bad, класс 0)"
            thr_pred_text = "НАДЁЖЕН (good, класс 1)" if int(res["pred_thr"]) == 1 else "НЕНАДЁЖЕН (bad, класс 0)"

            st.success(f"Базовое предсказание модели (argmax): **{base_pred_text}**")
            st.info(f"Предсказание по порогу {thr:.2f}: **{thr_pred_text}**")

            st.markdown("**Вероятности классов:**")
            st.json({
                "Вероятность дефолта (bad=0)": float(res["proba_bad"]),
                "Вероятность погашения (good=1)": float(res["proba_good"]),
            })

        except Exception as e:
            st.error(f"Ошибка предсказания: {e}")

# =========================
# TAB 2: Batch CSV
# =========================
with tab2:
    st.subheader("Пакетные предсказания (CSV)")
    st.caption("Ожидается CSV **с англ. категориями** как при обучении (sex, housing, saving_accounts, checking_account, purpose). "
               "Колонки: " + ", ".join(FEATURES))

    upl = st.file_uploader("Загрузите CSV", type=["csv"])
    if upl is not None:
        try:
            df = pd.read_csv(upl)

            df = coerce_and_order_df(df)
            out = predict_with_threshold(df, threshold=thr)

            result = pd.concat([df.reset_index(drop=True), out], axis=1)

            st.write("Размер входа:", df.shape, " | Размер выхода:", result.shape)
            st.dataframe(result.head(50), use_container_width=True)

            buf = io.BytesIO()
            result.to_csv(buf, index=False)
            st.download_button("⬇️ Скачать predictions.csv", buf.getvalue(),
                               file_name="predictions.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка обработки файла: {e}")
