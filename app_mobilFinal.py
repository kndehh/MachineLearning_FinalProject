import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from scipy.stats import skew
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MobilSecond",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background: #1a1a2e;}
    [data-testid="stSidebar"] * {color: #e0e0e0 !important;}
    .sidebar-title {font-size: 22px; font-weight: 700; color: #4fc3f7 !important; margin-bottom: 8px;}
    div[data-testid="stMetric"] {
        background: #16213e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 16px;
    }
    div[data-testid="stMetric"] label {color: #a0c4ff !important; font-size: 13px !important;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important; font-size: 26px !important; font-weight: 700 !important;
    }
    .stButton > button {border-radius: 8px; font-weight: 600;}
    h1 {color: #4fc3f7;}
    h2, h3 {color: #81d4fa;}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
for key in ["page", "df", "X_train", "X_test", "y_train", "y_test",
            "scaler", "scaler_name", "train_cols", "models", "test_size",
            "random_state", "eval_results", "cat_cols", "num_cols"]:
    if key not in st.session_state:
        st.session_state[key] = "Home" if key == "page" else None

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_FILE    = "cars_combined_final1.csv"
TARGET_COL   = "price"
CAT_COLS     = ["car_type", "brand_name", "model", "engine_type", "transmission", "location"]
NUM_COLS     = ["year", "total_km"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data():
    return pd.read_csv(DATA_FILE)

def format_rupiah(value):
    return f"Rp {value:,.0f}"

def eval_table_html(df_res):
    rows_html = ""
    for i, (_, row) in enumerate(df_res.iterrows()):
        bg = "#ffffff" if i % 2 == 0 else "#f0f4f8"
        cells = ""
        for col in df_res.columns:
            val = row[col]
            display = f"{val:.4f}" if isinstance(val, float) else str(val)
            cells += (
                f'<td style="padding:9px 14px; border:1px solid #cdd5df; '
                f'color:#1a1a2e; font-size:14px; background:{bg};">{display}</td>'
            )
        rows_html += f"<tr>{cells}</tr>"
    headers = "".join(
        f'<th style="background:#0f3460; color:#ffffff; padding:10px 14px; '
        f'border:1px solid #0f3460; font-size:14px; font-weight:700; text-align:left;">{c}</th>'
        for c in df_res.columns
    )
    return f"""
    <div style="overflow-x:auto; margin-top:8px; margin-bottom:12px;">
      <table style="width:100%; border-collapse:collapse; border-radius:8px; overflow:hidden;">
        <thead><tr>{headers}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""

def save_pickle(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def list_saved_pickles():
    return sorted([f for f in os.listdir(".") if f.endswith(".pickle")])

# ── Sidebar Nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🚗 MobilSecond</div>', unsafe_allow_html=True)
    st.markdown("**Smart Market Valuation App**")
    st.markdown("---")
    pages = {
        "🏠 Home"                             : "Home",
        "📊 Data Insights (EDA)"              : "EDA",
        "⚙️ Data Preprocessing"               : "Preprocessing",
        "🔍 Model Selection & Evaluation"     : "Model",
        "🔮 Price Prediction"                 : "Prediction",
    }
    for label, key in pages.items():
        is_active = st.session_state.page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.page = key
            st.rerun()
    st.markdown("---")
    st.markdown("""
    <div style="font-size:18px; color:#e0e0e0; line-height:2.2;">
        <div style="margin-bottom:6px; font-weight:700; font-size:16px; color:#4fc3f7;">Dibuat oleh:</div>
        1. Albertus Adrian<br>
        2. Jonathan Raffael<br>
        3. Steven Hosea<br>
        <span style="font-size:14px;">by binus university student</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 0 — HOME
# =============================================================================
if st.session_state.page == "Home":
    st.markdown("""
    <style>
        .home-container {
            display: flex; flex-direction: column;
            align-items: center; justify-content: center;
            text-align: center; padding: 20px 40px;
        }
        .home-headline {
            font-size: 28px; font-weight: 700; color: #4fc3f7;
            line-height: 1.4; max-width: 750px; margin-bottom: 16px;
        }
        .home-subtext {
            font-size: 17px; color: #b0bec5; max-width: 700px;
            line-height: 1.7; margin-bottom: 32px;
        }
    </style>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 3, 1])
    with col_c:
        st.markdown('''
        <div class="home-container">
            <div class="home-headline">
                Berapa harga wajar mobil bekas yang ingin Anda beli?
            </div>
            <div class="home-subtext">
                Harga mobil bekas di Indonesia sangat bervariasi dan sulit diprediksi.
                Dengan machine learning, aplikasi ini membantu Anda memperkirakan
                harga mobil bekas secara akurat berdasarkan data nyata dari OTO.com.
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Data", "3,689")
        c2.metric("Brand", "10")
        c3.metric("Model", "77+")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True, key="btn_start"):
            st.session_state.page = "EDA"
            st.rerun()


# =============================================================================
# PAGE 1 — EDA
# =============================================================================
elif st.session_state.page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    if st.session_state.df is None:
        try:
            df = load_data()
            st.session_state.df = df
        except FileNotFoundError:
            st.error(f"❌ File `{DATA_FILE}` tidak ditemukan. Pastikan file berada di folder yang sama.")
            st.stop()

    df = st.session_state.df
    st.success(f"✅ Dataset dimuat — {df.shape[0]:,} baris, {df.shape[1]} kolom")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗂️ Data Overview", "📈 Histogram & Countplot", "🔥 Heatmap", "📉 Scatter & Box"]
    )

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("5 Baris Pertama")
            st.dataframe(df.head(), use_container_width=True)
        with c2:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe(), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Tipe Data")
            dtype_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index()
            dtype_df.columns = ["Column", "Type"]
            st.dataframe(dtype_df, use_container_width=True)
        with c4:
            st.subheader("Missing Values")
            null_df = pd.DataFrame(df.isnull().sum(), columns=["Null Count"]).reset_index()
            null_df.columns = ["Column", "Null Count"]
            null_df["Status"] = null_df["Null Count"].apply(
                lambda x: "✅ OK" if x == 0 else f"⚠️ {x} missing")
            st.dataframe(null_df, use_container_width=True)

        st.markdown("---")
        c5, c6 = st.columns(2)
        with c5:
            st.subheader("📍 Distribusi Location")
            loc_df = df["location"].value_counts().reset_index()
            loc_df.columns = ["Location", "Jumlah"]
            st.dataframe(loc_df, use_container_width=True, height=300)
        with c6:
            st.subheader("🚗 Distribusi Model")
            model_df = df["model"].value_counts().reset_index()
            model_df.columns = ["Model", "Jumlah"]
            st.dataframe(model_df, use_container_width=True, height=300)

    with tab2:
        # ── Histogram ─────────────────────────────────────────────────────────
        st.subheader("Histogram — Numerikal")
        num_features_eda = [TARGET_COL] + NUM_COLS
        sel_hist = st.multiselect(
            "Pilih kolom numerikal:",
            num_features_eda,
            default=num_features_eda,
            key="hist_sel",
        )
        if sel_hist:
            rows_hist = [sel_hist[i:i+3] for i in range(0, len(sel_hist), 3)]
            for row in rows_hist:
                cols_ui = st.columns(len(row))
                for col_ui, feat in zip(cols_ui, row):
                    with col_ui:
                        fig, ax = plt.subplots(figsize=(5, 3.5))
                        fig.patch.set_facecolor("#0e1117")
                        ax.set_facecolor("#16213e")
                        sns.histplot(df[feat].dropna(), kde=True, ax=ax, color="#4fc3f7")
                        ax.set_title(feat, color="white", fontsize=9)
                        ax.tick_params(colors="white")
                        ax.xaxis.label.set_color("white")
                        ax.yaxis.label.set_color("white")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
        else:
            st.info("Pilih minimal satu kolom untuk ditampilkan.")

        st.markdown("---")

        # ── Countplot ─────────────────────────────────────────────────────────
        st.subheader("Countplot — Kategorikal")
        countplot_cols = [c for c in CAT_COLS if c not in ["model", "location"]]
        sel_count = st.multiselect(
            "Pilih kolom kategorikal:",
            countplot_cols,
            default=countplot_cols,
            key="count_sel",
        )
        if sel_count:
            for col in sel_count:
                fig, ax = plt.subplots(figsize=(10, 4))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#16213e")
                order = df[col].value_counts().index
                sns.countplot(data=df, x=col, order=order, color="#4fc3f7", alpha=0.8, ax=ax)
                ax.set_title(f"Countplot {col}", color="white")
                ax.tick_params(colors="white", axis="x", rotation=45)
                ax.tick_params(colors="white", axis="y")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Pilih minimal satu kolom untuk ditampilkan.")

    with tab3:
        st.subheader("Correlation Heatmap (Numerikal)")
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#0e1117")
        ax.set_facecolor("#0e1117")
        corr = df[[TARGET_COL] + NUM_COLS].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax,
                    annot_kws={"size": 12}, fmt=".2f",
                    linewidths=0.5, linecolor="#1a1a2e")
        ax.tick_params(colors="white")
        plt.xticks(color="white")
        plt.yticks(color="white")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab4:
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Scatter Plot (vs Harga)")
            sel_scatter = st.selectbox("Pilih fitur:", NUM_COLS, key="scatter_sel")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            sns.regplot(x=df[sel_scatter], y=df[TARGET_COL], ax=ax,
                        scatter_kws={"alpha": 0.3, "s": 15, "color": "#4fc3f7"},
                        line_kws={"color": "#ff7043"})
            ax.set_ylabel("Harga (Rp)", color="white")
            ax.set_xlabel(sel_scatter, color="white")
            ax.tick_params(colors="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_right:
            st.subheader("Boxplot Harga per Kategori")
            sel_box = st.selectbox("Pilih kolom:", countplot_cols, key="box_sel")
            order_box = df.groupby(sel_box)[TARGET_COL].median().sort_values(ascending=False).index
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            sns.boxplot(data=df, x=sel_box, y=TARGET_COL, order=order_box,
                        color="#4fc3f7", ax=ax)
            ax.set_title(f"Harga per {sel_box}", color="white")
            ax.tick_params(colors="white", axis="x", rotation=45)
            ax.tick_params(colors="white", axis="y")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# =============================================================================
# PAGE 2 — PREPROCESSING
# =============================================================================
elif st.session_state.page == "Preprocessing":
    st.title("⚙️ Preprocessing")

    if st.session_state.df is None:
        try:
            df = load_data()
            st.session_state.df = df
        except FileNotFoundError:
            st.warning("⚠️ Buka halaman EDA terlebih dahulu.")
            st.stop()

    df = st.session_state.df

    # ── 1. Train-Test Split ───────────────────────────────────────────────────
    st.subheader("1️⃣  Train-Test Split")

    st.markdown("""
    <div style="background:#16213e; border-left:4px solid #4fc3f7; border-radius:8px; padding:12px 16px; margin-bottom:8px;">
        <b>✂️ Apa itu Train-Test Split?</b><br>
        Bayangkan Anda belajar soal ujian. <b>Data training</b> adalah soal latihan yang Anda pelajari,
        sedangkan <b>data test</b> adalah soal ujian sesungguhnya yang belum pernah dilihat.
        Split dilakukan <i>sebelum</i> preprocessing agar model tidak "bocor" informasi dari data test.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        test_size = st.slider("Ukuran Test Set (%)", 10, 40, 20, 5)
        st.caption(f"Train: **{100 - test_size}%** ({int(len(df) * (100-test_size)/100):,} baris) | Test: **{test_size}%** ({int(len(df) * test_size/100):,} baris)")
    with c2:
        random_state = st.number_input("Random State", 0, 999, 42, 1)
        st.caption("Angka acak untuk memastikan hasil split selalu sama setiap kali dijalankan.")

    st.markdown("---")

    # ── 2. Encoding ───────────────────────────────────────────────────────────
    st.subheader("2️⃣  Encoding Kategorikal")
    st.markdown("""
    <div style="background:#16213e; border-left:4px solid #ffa726; border-radius:8px; padding:12px 16px; margin-bottom:12px;">
        <b>🔤 Mengapa perlu encoding?</b><br>
        Model machine learning hanya bisa membaca angka, bukan teks.
        Encoding mengubah teks seperti "Toyota" atau "Manual" menjadi angka yang bisa diproses model.
    </div>
    """, unsafe_allow_html=True)

    enc_col1, enc_col2, enc_col3 = st.columns(3)
    with enc_col1:
        st.markdown("""
        <div style="background:#0f3460; border-radius:10px; padding:14px; text-align:center; min-height:160px;">
            <div style="font-size:28px;">🔢</div>
            <div style="color:#4fc3f7; font-weight:700; margin:6px 0;">One Hot Encoding</div>
            <div style="color:#b0bec5; font-size:13px;">
                Setiap kategori jadi kolom baru berisi 0 atau 1.<br><br>
                <i>Contoh: "Toyota" → kolom brand_Toyota = 1</i><br><br>
                ✅ Cocok untuk data tanpa urutan<br>
                ⚠️ Menghasilkan banyak kolom
            </div>
        </div>
        """, unsafe_allow_html=True)
    with enc_col2:
        st.markdown("""
        <div style="background:#0f3460; border-radius:10px; padding:14px; text-align:center; min-height:160px;">
            <div style="font-size:28px;">🏷️</div>
            <div style="color:#4fc3f7; font-weight:700; margin:6px 0;">Label Encoding</div>
            <div style="color:#b0bec5; font-size:13px;">
                Setiap kategori diubah jadi satu angka unik.<br><br>
                <i>Contoh: Manual=0, Otomatis=1, CVT=2</i><br><br>
                ✅ Lebih ringkas<br>
                ⚠️ Bisa salah memberi urutan pada data
            </div>
        </div>
        """, unsafe_allow_html=True)
    with enc_col3:
        st.markdown("""
        <div style="background:#0f3460; border-radius:10px; padding:14px; text-align:center; min-height:160px;">
            <div style="font-size:28px;">🎯</div>
            <div style="color:#4fc3f7; font-weight:700; margin:6px 0;">Target Encoding</div>
            <div style="color:#b0bec5; font-size:13px;">
                Setiap kategori diganti dengan rata-rata harga kategori tersebut.<br><br>
                <i>Contoh: "BMW" → rata-rata harga BMW</i><br><br>
                ✅ Cocok untuk kolom dengan banyak nilai unik<br>
                ✅ Mempertahankan informasi harga
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    encoding_option = st.radio(
        "Pilih metode encoding:",
        ["One Hot Encoding", "Label Encoding", "Target Encoding"],
        horizontal=True,
        key="enc_radio",
    )

    st.markdown("---")

    # ── 3. Scaling ────────────────────────────────────────────────────────────
    st.subheader("3️⃣  Scaling Numerikal")
    st.markdown("""
    <div style="background:#16213e; border-left:4px solid #66bb6a; border-radius:8px; padding:12px 16px; margin-bottom:12px;">
        <b>📏 Mengapa perlu scaling?</b><br>
        Kolom <i>year</i> bernilai ~2020, sedangkan <i>total_km</i> bisa bernilai 100.000.
        Tanpa scaling, model bisa bias ke fitur dengan angka besar.
        Scaling menyamakan "skala" semua fitur agar model belajar secara adil.
    </div>
    """, unsafe_allow_html=True)

    scl_col1, scl_col2 = st.columns(2)
    with scl_col1:
        st.markdown("""
        <div style="background:#0f3460; border-radius:10px; padding:14px; text-align:center; min-height:130px;">
            <div style="font-size:28px;">📊</div>
            <div style="color:#4fc3f7; font-weight:700; margin:6px 0;">Standard Scaler (Z-score)</div>
            <div style="color:#b0bec5; font-size:13px;">
                Mengubah data sehingga rata-rata = 0 dan standar deviasi = 1.<br><br>
                ✅ Cocok untuk data berdistribusi normal<br>
                ✅ Tidak sensitif terhadap outlier
            </div>
        </div>
        """, unsafe_allow_html=True)
    with scl_col2:
        st.markdown("""
        <div style="background:#0f3460; border-radius:10px; padding:14px; text-align:center; min-height:130px;">
            <div style="font-size:28px;">📐</div>
            <div style="color:#4fc3f7; font-weight:700; margin:6px 0;">Min-Max Normalization</div>
            <div style="color:#b0bec5; font-size:13px;">
                Mengubah semua nilai ke rentang 0 sampai 1.<br><br>
                ✅ Mudah diinterpretasi<br>
                ⚠️ Sensitif terhadap outlier
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    scaler_option = st.radio(
        "Pilih metode scaling:",
        ["Standard Scaler (Z-score)", "Min-Max Normalization"],
        horizontal=True,
        key="scaler_radio",
    )

    st.markdown("---")

    if st.button("🚀 Jalankan Preprocessing", type="primary", use_container_width=True):
        df_proc = df.copy()

        df_proc["total_km"] = df_proc.groupby("model")["total_km"].transform(
            lambda x: x.fillna(x.median())
        )
        df_proc["total_km"] = df_proc["total_km"].fillna(df_proc["total_km"].median())
        df_proc["location"] = df_proc["location"].fillna("Unknown")

        X = df_proc.drop(TARGET_COL, axis=1)
        y = df_proc[TARGET_COL]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=int(random_state)
        )

        num_cols = X_train.select_dtypes(include="number").columns.tolist()
        cat_cols = X_train.select_dtypes(include="object").columns.tolist()

        for col in num_cols:
            col_skew = skew(X_train[col].dropna())
            fill_val = X_train[col].mean() if abs(col_skew) < 0.5 else X_train[col].median()
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col]  = X_test[col].fillna(fill_val)

        for col in cat_cols:
            fill_val = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col]  = X_test[col].fillna(fill_val)

        # Encoding
        if encoding_option == "One Hot Encoding":
            X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
            X_test_enc  = pd.get_dummies(X_test,  columns=cat_cols, drop_first=True)
            X_test_enc  = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
            encoding_name = "OneHotEncoding"

        elif encoding_option == "Label Encoding":
            from sklearn.preprocessing import LabelEncoder
            X_train_enc = X_train.copy()
            X_test_enc  = X_test.copy()
            label_encoders = {}
            for col in cat_cols:
                le = LabelEncoder()
                X_train_enc[col] = le.fit_transform(X_train_enc[col].astype(str))
                X_test_enc[col]  = X_test_enc[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else le.classes_[0]
                )
                X_test_enc[col] = le.transform(X_test_enc[col])
                label_encoders[col] = le
            st.session_state["label_encoders"] = label_encoders
            encoding_name = "LabelEncoding"

        else:  # Target Encoding
            X_train_enc = X_train.copy()
            X_test_enc  = X_test.copy()
            target_means = {}
            for col in cat_cols:
                means = X_train.join(y_train).groupby(col)[TARGET_COL].mean()
                overall_mean = y_train.mean()
                X_train_enc[col] = X_train_enc[col].map(means).fillna(overall_mean)
                X_test_enc[col]  = X_test_enc[col].map(means).fillna(overall_mean)
                target_means[col] = means
            st.session_state["target_means"]    = target_means
            st.session_state["overall_mean_y"]  = y_train.mean()
            encoding_name = "TargetEncoding"

        # Scaling
        if "Standard" in scaler_option:
            scaler = StandardScaler()
            scaler_name = "StandardScaler"
        else:
            scaler = MinMaxScaler()
            scaler_name = "MinMaxScaler"

        X_train_enc[num_cols] = scaler.fit_transform(X_train_enc[num_cols])
        X_test_enc[num_cols]  = scaler.transform(X_test_enc[num_cols])

        st.session_state.X_train         = X_train_enc
        st.session_state.X_test          = X_test_enc
        st.session_state.y_train         = y_train
        st.session_state.y_test          = y_test
        st.session_state.scaler          = scaler
        st.session_state.scaler_name     = scaler_name
        st.session_state.encoding_name   = encoding_name
        st.session_state.encoding_option = encoding_option
        st.session_state.train_cols      = list(X_train_enc.columns)
        st.session_state.test_size       = test_size
        st.session_state.random_state    = random_state
        st.session_state.num_cols        = num_cols
        st.session_state.cat_cols        = cat_cols
        st.session_state.models          = None
        st.session_state.eval_results    = None

        st.success("✅ Preprocessing selesai!")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Total Sampel",     f"{len(df_proc):,}")
        cb.metric("Training Samples", f"{len(X_train_enc):,}")
        cc.metric("Testing Samples",  f"{len(X_test_enc):,}")
        cd.metric("Jumlah Fitur",     len(X_train_enc.columns))

        st.subheader(f"📋 Preview Data setelah {encoding_name} & {scaler_name} (10 baris pertama)")
        st.dataframe(X_train_enc.head(10), use_container_width=True)

    elif st.session_state.X_train is not None:
        enc_name = st.session_state.get("encoding_name", "OHE")
        st.info(
            f"ℹ️ Preprocessing sudah dijalankan — "
            f"Test size: **{st.session_state.test_size}%**, "
            f"Encoding: **{enc_name}**, "
            f"Scaler: **{st.session_state.scaler_name}**. "
            "Ubah setting lalu klik tombol untuk jalankan ulang."
        )


# =============================================================================
# PAGE 3 — MODEL SELECTION & EVALUATION
# =============================================================================
elif st.session_state.page == "Model":
    st.title("🤖 Model Selection & Evaluation")

    if st.session_state.X_train is None:
        st.warning("⚠️ Lakukan Preprocessing terlebih dahulu.")
        st.stop()

    X_train = st.session_state.X_train
    X_test  = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test  = st.session_state.y_test

    MODEL_OPTIONS = {
        "Random Forest"           : "rf",
        "Linear Regression"       : "lr",
        "KNN Regressor"           : "knn",
        "Ridge Regression"        : "ridge",
        "Lasso Regression"        : "lasso",
        "SVR"                     : "svr",
    }

    st.subheader("1️⃣  Pilih Model")
    selected_models = st.multiselect(
        "Pilih satu atau lebih model untuk dilatih:",
        list(MODEL_OPTIONS.keys()),
        default=["Random Forest", "Linear Regression"],
    )

    if st.button("🏋️ Train & Evaluate", type="primary", use_container_width=True):
        if not selected_models:
            st.error("Pilih minimal satu model.")
            st.stop()

        results        = {}
        trained_models = {}
        progress = st.progress(0, "Melatih model...")

        for i, name in enumerate(selected_models):
            progress.progress(i / len(selected_models), f"Melatih {name}...")
            k = MODEL_OPTIONS[name]
            if k == "rf":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif k == "lr":
                model = LinearRegression()
            elif k == "knn":
                model = KNeighborsRegressor(n_neighbors=5)
            elif k == "ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0)
            elif k == "lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=1.0)
            else:
                model = SVR(kernel="rbf", C=1.0, epsilon=0.1)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results[name] = {
                "R² Score" : r2_score(y_test, y_pred),
                "MAE"      : mean_absolute_error(y_test, y_pred),
                "RMSE"     : np.sqrt(mean_squared_error(y_test, y_pred)),
            }
            trained_models[name] = (model, y_pred)

        progress.progress(1.0, "✅ Selesai!")
        st.session_state.models       = trained_models
        st.session_state.eval_results = results

    # ── Evaluation Results ────────────────────────────────────────────────────
    if st.session_state.eval_results:
        results        = st.session_state.eval_results
        trained_models = st.session_state.models

        st.markdown("---")
        st.subheader("📊 Hasil Evaluasi")

        df_results = pd.DataFrame(results).T.reset_index()
        df_results.columns = ["Model", "R² Score", "MAE", "RMSE"]
        st.markdown(eval_table_html(df_results), unsafe_allow_html=True)

        # Best model highlight
        best_model_name = max(results, key=lambda x: results[x]["R² Score"])
        st.markdown(f"""
        <div style="background:#1b4d1b; border:1px solid #66bb6a; border-radius:8px;
             padding:10px 16px; margin-bottom:16px;">
            🏆 <b>Model Terbaik: {best_model_name}</b> dengan R² = {results[best_model_name]['R² Score']:.4f}
        </div>
        """, unsafe_allow_html=True)

        for name, (model, y_pred) in trained_models.items():
            st.markdown(f"### 📌 {name}")
            m = results[name]
            r2   = m["R² Score"]
            mae  = m["MAE"]
            rmse = m["RMSE"]

            c1, c2, c3 = st.columns(3)
            c1.metric("R² Score", f"{r2:.4f}")
            c2.metric("MAE",      format_rupiah(mae))
            c3.metric("RMSE",     format_rupiah(rmse))

            # Interpretasi otomatis
            if r2 >= 0.90:
                r2_label, r2_color = "Sangat Baik ✅", "#66bb6a"
                r2_desc = f"Model mampu menjelaskan {r2*100:.1f}% variasi harga — sangat akurat."
            elif r2 >= 0.75:
                r2_label, r2_color = "Baik 👍", "#ffa726"
                r2_desc = f"Model mampu menjelaskan {r2*100:.1f}% variasi harga — cukup andal."
            elif r2 >= 0.50:
                r2_label, r2_color = "Cukup ⚠️", "#ff7043"
                r2_desc = f"Model hanya menjelaskan {r2*100:.1f}% variasi harga — perlu peningkatan."
            else:
                r2_label, r2_color = "Kurang ❌", "#ef5350"
                r2_desc = f"Model hanya menjelaskan {r2*100:.1f}% variasi harga — model kurang cocok."

            mae_pct  = mae / y_test.mean() * 100
            rmse_pct = rmse / y_test.mean() * 100

            st.markdown(f"""
            <div style="background:#16213e; border-left:4px solid {r2_color};
                 border-radius:8px; padding:14px 18px; margin:10px 0;">
                <b>📋 Interpretasi {name}</b><br><br>
                <b>R² Score ({r2:.4f}) — {r2_label}</b><br>
                {r2_desc}<br><br>
                <b>MAE ({format_rupiah(mae)} | {mae_pct:.1f}% dari rata-rata harga)</b><br>
                Rata-rata selisih antara harga prediksi dan harga aktual adalah <b>{format_rupiah(mae)}</b>.
                Artinya prediksi model meleset sekitar <b>{mae_pct:.1f}%</b> dari harga rata-rata mobil.<br><br>
                <b>RMSE ({format_rupiah(rmse)} | {rmse_pct:.1f}% dari rata-rata harga)</b><br>
                RMSE lebih sensitif terhadap kesalahan besar. Nilai {format_rupiah(rmse)} berarti
                prediksi yang meleset jauh (misal mobil mewah) memberi penalti lebih besar.
                {"✅ Nilai RMSE tidak jauh dari MAE — prediksi konsisten." if rmse/mae < 2 else "⚠️ RMSE jauh lebih besar dari MAE — ada beberapa prediksi yang meleset sangat jauh."}
            </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="#4fc3f7", label="Predictions")
            mn = min(float(y_test.min()), float(y_pred.min()))
            mx = max(float(y_test.max()), float(y_pred.max()))
            ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
            ax.set_xlabel("Harga Aktual (Rp)", color="white")
            ax.set_ylabel("Harga Prediksi (Rp)", color="white")
            ax.set_title(f"Actual vs Predicted — {name}", color="white")
            ax.tick_params(colors="white")
            ax.legend(facecolor="#16213e", labelcolor="white")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.markdown("---")

        # ── Feature Importance (RF only) ──────────────────────────────────────
        rf_trained = next(
            ((m, p) for n, (m, p) in trained_models.items() if "Random Forest" in n), None
        )
        if rf_trained:
            st.subheader("🌲 Feature Importance (Random Forest)")
            importances = pd.Series(
                rf_trained[0].feature_importances_,
                index=st.session_state.train_cols
            ).sort_values(ascending=False).head(15)

            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#16213e")
            importances.plot(kind="bar", ax=ax, color="#4fc3f7", alpha=0.8)
            ax.set_title("Top 15 Feature Importance", color="white")
            ax.tick_params(colors="white", axis="x", rotation=45)
            ax.tick_params(colors="white", axis="y")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        # ── Save Model ────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💾 Simpan Model, Scaler & Encoder")
        col_save1, col_save2, col_save3 = st.columns(3)

        with col_save1:
            save_model_choice = st.selectbox(
                "Pilih model yang ingin disimpan:",
                list(trained_models.keys()),
                key="save_model_sel",
            )
            model_filename = st.text_input(
                "Nama file model (tanpa ekstensi):",
                value="model",
                key="model_fname",
            )
            if st.button("💾 Simpan Model", type="primary", key="btn_save_model"):
                save_pickle(trained_models[save_model_choice][0], f"{model_filename}.pickle")
                st.success(f"✅ Tersimpan: `{model_filename}.pickle`")

        with col_save2:
            if st.session_state.scaler is not None:
                scaler_filename = st.text_input(
                    "Nama file scaler (tanpa ekstensi):",
                    value="scaler",
                    key="scaler_fname",
                )
                if st.button("💾 Simpan Scaler", type="primary", key="btn_save_scaler"):
                    save_pickle(st.session_state.scaler, f"{scaler_filename}.pickle")
                    st.success(f"✅ Tersimpan: `{scaler_filename}.pickle`")

        with col_save3:
            enc_opt = st.session_state.get("encoding_option", "One Hot Encoding")
            encoder_filename = st.text_input(
                "Nama file encoder (tanpa ekstensi):",
                value="encoder",
                key="encoder_fname",
            )
            if st.button("💾 Simpan Encoder", type="primary", key="btn_save_encoder"):
                if enc_opt == "One Hot Encoding":
                    save_pickle({
                        "type"   : "ohe",
                        "columns": st.session_state.train_cols,
                    }, f"{encoder_filename}.pickle")
                elif enc_opt == "Label Encoding" and st.session_state.get("label_encoders"):
                    save_pickle({
                        "type"    : "label",
                        "encoders": st.session_state["label_encoders"],
                    }, f"{encoder_filename}.pickle")
                elif enc_opt == "Target Encoding" and st.session_state.get("target_means"):
                    save_pickle({
                        "type"        : "target",
                        "target_means": st.session_state["target_means"],
                        "overall_mean": st.session_state.get("overall_mean_y", 0),
                    }, f"{encoder_filename}.pickle")
                st.success(f"✅ Tersimpan: `{encoder_filename}.pickle` ({enc_opt})")


# =============================================================================
# PAGE 4 — PREDICTION
# =============================================================================
elif st.session_state.page == "Prediction":
    st.title("🔮 Prediksi Harga Mobil Bekas")
    st.markdown("Pilih model & scaler, masukkan spesifikasi mobil, lalu prediksi harganya.")

    all_pickles  = list_saved_pickles()
    SCALER_KW    = ["scaler", "standardscaler", "minmaxscaler"]
    scaler_files = [f for f in all_pickles if any(k in f.lower() for k in SCALER_KW)]
    model_files  = [f for f in all_pickles if f not in scaler_files]
    # Encoder files: pickle yang bukan model dan bukan scaler
    # Kita deteksi dengan load dan cek key "type"
    encoder_candidates = []
    for f in all_pickles:
        if f in scaler_files:
            continue
        try:
            with open(f, "rb") as fp:
                obj = pickle.load(fp)
            if isinstance(obj, dict) and "type" in obj:
                encoder_candidates.append(f)
            elif hasattr(obj, "predict"):
                pass  # model file
        except Exception:
            pass
    model_files = [f for f in all_pickles
                   if f not in scaler_files and f not in encoder_candidates]

    st.subheader("1️⃣  Pilih Model, Scaler & Encoder")
    col_m, col_s, col_e = st.columns(3)

    with col_m:
        if not model_files:
            st.warning("⚠️ Belum ada file model `.pickle`. Latih model di halaman **Model Selection** dulu.")
            selected_model_file = None
        else:
            selected_model_file = st.selectbox("Pilih model:", model_files, key="sel_model_file")

    with col_s:
        if scaler_files:
            scaler_opts = ["— Tanpa Scaler —"] + scaler_files
            sel_sc = st.selectbox("Pilih scaler:", scaler_opts, key="sel_scaler_file")
            selected_scaler_file = None if sel_sc == "— Tanpa Scaler —" else sel_sc
        else:
            st.info("ℹ️ Tidak ada file scaler.")
            selected_scaler_file = None

    with col_e:
        if encoder_candidates:
            sel_enc = st.selectbox("Pilih encoder:", encoder_candidates, key="sel_encoder_file")
            selected_encoder_file = sel_enc
        else:
            st.warning("⚠️ Belum ada file encoder. Simpan encoder di halaman **Model Selection**.")
            selected_encoder_file = None

    # Load model, scaler, encoder
    loaded_model   = None
    pred_scaler    = None
    loaded_encoder = None
    encoder_type   = "ohe"
    loaded_cols    = None

    if selected_model_file:
        try:
            with open(selected_model_file, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "predict"):
                loaded_model = obj
                st.success(f"✅ Model: `{selected_model_file}` ({type(obj).__name__})")
            else:
                st.error("❌ File bukan model yang valid.")
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")

    if selected_scaler_file:
        try:
            with open(selected_scaler_file, "rb") as f:
                obj = pickle.load(f)
            if hasattr(obj, "transform"):
                pred_scaler = obj
                st.success(f"✅ Scaler: `{selected_scaler_file}` ({type(obj).__name__})")
        except Exception as e:
            st.error(f"❌ Gagal memuat scaler: {e}")

    if selected_encoder_file:
        try:
            with open(selected_encoder_file, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "type" in obj:
                encoder_type   = obj["type"]
                loaded_encoder = obj
                loaded_cols    = obj.get("columns", None)
                st.success(f"✅ Encoder: `{selected_encoder_file}` ({encoder_type.upper()} Encoding)")
        except Exception as e:
            st.error(f"❌ Gagal memuat encoder: {e}")

    # ── Load data untuk dropdown ──────────────────────────────────────────────
    try:
        df_ref = load_data()
    except FileNotFoundError:
        df_ref = None

    # ── Input Form ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("2️⃣  Masukkan Spesifikasi Mobil")

    col1, col2 = st.columns(2)

    with col1:
        if df_ref is not None:
            brand_opts = sorted(df_ref["brand_name"].dropna().unique().tolist())
        else:
            brand_opts = ["Toyota", "Honda", "Daihatsu", "Suzuki", "Mitsubishi",
                          "Nissan", "BMW", "Mercedes Benz", "Mazda", "Hyundai"]
        brand = st.selectbox("Brand", brand_opts)

        if df_ref is not None:
            model_opts = sorted(df_ref[df_ref["brand_name"] == brand]["model"].dropna().unique().tolist())
        else:
            model_opts = ["Avanza", "Innova", "Fortuner"]
        model = st.selectbox("Model", model_opts)

        if df_ref is not None:
            car_type_opts = sorted(df_ref["car_type"].dropna().unique().tolist())
        else:
            car_type_opts = ["MPV", "SUV", "Sedan", "Hatchback", "CrossOver"]
        car_type = st.selectbox("Tipe Kendaraan", car_type_opts)

        year = st.number_input("Tahun", min_value=1990, max_value=2025, value=2020, step=1)

    with col2:
        total_km = st.number_input("Jarak Tempuh (KM)", min_value=0, max_value=500000,
                                   value=50000, step=1000)

        engine_type = st.selectbox("Tipe Mesin", ["Bensin", "Diesel", "Hybrid", "Listrik"])
        transmission = st.selectbox("Transmisi", ["Otomatis", "CVT", "Manual"])

        if df_ref is not None:
            location_opts = sorted(df_ref["location"].dropna().unique().tolist())
            location_opts = [l for l in location_opts if l != "Unknown"]
        else:
            location_opts = ["Jakarta Pusat", "Surabaya", "Bandung", "Bekasi", "Denpasar"]
        location = st.selectbox("Lokasi", location_opts)

    # ── Predict ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("3️⃣  Prediksi")

    if st.button("🔮 Prediksi Harga", type="primary", use_container_width=True):
        if loaded_model is None:
            st.error("❌ Pilih dan muat model terlebih dahulu.")
        else:
            input_dict = {
                "car_type"    : car_type,
                "brand_name"  : brand,
                "model"       : model,
                "year"        : year,
                "total_km"    : float(total_km),
                "location"    : location,
                "engine_type" : engine_type,
                "transmission": transmission,
            }
            input_df = pd.DataFrame([input_dict])
            cat_cols_pred = ["car_type", "brand_name", "model", "engine_type", "transmission", "location"]
            num_cols_pred = ["year", "total_km"]

            # Terapkan encoding sesuai tipe encoder
            if encoder_type == "label" and loaded_encoder:
                input_enc = input_df.copy()
                for col in cat_cols_pred:
                    le = loaded_encoder.get("encoders", {}).get(col)
                    if le:
                        val = str(input_enc[col].iloc[0])
                        val = val if val in le.classes_ else le.classes_[0]
                        input_enc[col] = le.transform([val])[0]

            elif encoder_type == "target" and loaded_encoder:
                input_enc = input_df.copy()
                target_means = loaded_encoder.get("target_means", {})
                overall_mean = loaded_encoder.get("overall_mean", 0)
                for col in cat_cols_pred:
                    if col in target_means:
                        val = str(input_enc[col].iloc[0])
                        input_enc[col] = target_means[col].get(val, overall_mean)

            else:  # OHE
                input_enc = pd.get_dummies(input_df)
                if loaded_cols:
                    input_enc = input_enc.reindex(columns=loaded_cols, fill_value=0)
                elif st.session_state.train_cols:
                    input_enc = input_enc.reindex(columns=st.session_state.train_cols, fill_value=0)

            # Untuk label/target encoding, align kolom juga
            if encoder_type in ("label", "target") and loaded_cols:
                input_enc = input_enc.reindex(columns=loaded_cols, fill_value=0)
            elif encoder_type in ("label", "target") and st.session_state.train_cols:
                input_enc = input_enc.reindex(columns=st.session_state.train_cols, fill_value=0)

            # Scale numerikal
            scaler_to_use = pred_scaler or st.session_state.scaler
            if scaler_to_use is not None:
                try:
                    input_enc[num_cols_pred] = scaler_to_use.transform(input_enc[num_cols_pred])
                except Exception:
                    pass

            prediction = loaded_model.predict(input_enc.values)[0]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #0f3460, #16213e);
                border: 2px solid #4fc3f7; border-radius: 16px;
                padding: 32px; text-align: center; margin: 16px 0;">
                <div style="font-size:14px; color:#a0a0b0; margin-bottom:8px;">
                    Estimasi Harga Mobil Bekas
                </div>
                <div style="font-size:48px; font-weight:800; color:#4fc3f7;">
                    {format_rupiah(prediction)}
                </div>
                <div style="font-size:14px; color:#81d4fa; margin-top:8px;">
                    {brand} {model} {year} • {total_km:,} KM • {transmission}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Kategori harga
            if prediction < 100_000_000:
                cat, clr = "Murah (< 100 Juta)",              "#66bb6a"
            elif prediction < 300_000_000:
                cat, clr = "Menengah (100 - 300 Juta)",       "#ffa726"
            elif prediction < 700_000_000:
                cat, clr = "Mahal (300 Juta - 700 Juta)",     "#42a5f5"
            else:
                cat, clr = "Premium (> 700 Juta)",            "#ef5350"

            st.markdown(f"""
            <div style="text-align:center; margin-top:8px;">
                <span style="background:{clr}22; border:1px solid {clr};
                    border-radius:20px; padding:6px 20px;
                    color:{clr}; font-weight:600; font-size:15px;">
                    {cat}
                </span>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("📋 Ringkasan Input"):
                st.dataframe(
                    pd.DataFrame([input_dict]).T.rename(columns={0: "Value"}),
                    use_container_width=True,
                )
