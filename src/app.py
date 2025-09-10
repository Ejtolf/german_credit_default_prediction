import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("..models/rf_model.pkl")

model = load_model()
classes = list(model.classes_)
bad_idx = classes.index(0)

st.set_page_config(page_title="Credit Scoring Demo", page_icon="💳", layout="centered")
st.title("💳 Credit Scoring — German Credit (Demo)")
st.write("Введите параметры клиента и получите вероятность риска.")

with st.form("scoring_form"):
    age = st.number_input("Возраст", min_value=18, max_value=85, value=29, step=1)
    sex = st.selectbox("Пол", ["male", "female"])
    job = st.selectbox("Категория работы (job)", [0,1,2,3])
    housing = st.selectbox("Жильё", ["own", "rent", "free"])
    saving_accounts = st.selectbox("Сберегательный счёт", ["no_info","little","moderate","quite rich","rich"])
    checking_account = st.selectbox("Расчётный счёт", ["no_info","little","moderate","rich"])
    credit_amount = st.number_input("Сумма кредита", min_value=100, max_value=20000, value=5000, step=100)
    duration = st.number_input("Срок кредита (мес.)", min_value=4, max_value=72, value=24, step=1)
    purpose = st.selectbox("Цель кредита", [
        "car","radio/TV","furniture/equipment","education","business","domestic appliances","repairs","vacation","retraining","other"
    ])
    monthly_payment = credit_amount / duration

    submitted = st.form_submit_button("Рассчитать риск")

if submitted:
    X_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "job": job,
        "housing": housing,
        "saving_accounts": saving_accounts,
        "checking_account": checking_account,
        "credit_amount": credit_amount,
        "duration": duration,
        "purpose": purpose,
        # дополнительные фичи должны называться как в тренировочном df
        "monthly_payment": monthly_payment
    }])

    proba = model.predict_proba(X_input)[0]
    proba_bad = float(proba[bad_idx])
    proba_good = float(proba[1-bad_idx])

    # Порог можно настраивать
    threshold = st.slider("Порог одобрения (по вероятности bad)", 0.0, 1.0, 0.5, 0.01)
    decision = "❌ Отказать" if proba_bad >= threshold else "✅ Одобрить"

    st.subheader("Результат")
    st.metric(label="Вероятность невозврата (bad)", value=f"{proba_bad:.1%}")
    st.metric(label="Решение", value=decision)

    # Небольшое пояснение
    st.caption(
        "Примечание: порядок классов в модели: "
        f"{classes}. Отображается вероятность класса bad={0}."
    )

    # (опционально) Пояснение важности признаков (если модель/пайтлайн это поддерживают)
    try:
        import numpy as np
        # для деревьев: показать топ-важные фичи после препроцессинга
        if hasattr(model.named_steps["model"], "feature_importances_"):
            st.write("Топ-важные признаки (по модели):")
            importances = model.named_steps["model"].feature_importances_
            # попытка восстановить имена фичей после OHE
            ohe = model.named_steps["preprocessor"].transformers_[1][1].named_steps["encoder"]
            num_feats = model.named_steps["preprocessor"].transformers_[0][2]
            cat_feats = model.named_steps["preprocessor"].transformers_[1][2]
            ohe_names = ohe.get_feature_names_out(cat_feats)
            feature_names = np.concatenate([num_feats, ohe_names])
            top_idx = np.argsort(importances)[-10:][::-1]
            top_table = pd.DataFrame({
                "feature": feature_names[top_idx],
                "importance": importances[top_idx]
            })
            st.dataframe(top_table, use_container_width=True)
    except Exception as e:
        st.caption("Не удалось вывести важности признаков (в демо можно пропустить).")

st.divider()
st.caption("Demo. Не использовать для реальных кредитных решений без калибровки, валидаций и мониторинга.")
