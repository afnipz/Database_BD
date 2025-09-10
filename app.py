import streamlit as st
import pickle
import numpy as np
import psycopg2
import pandas as pd

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Prediksi Diabetes",
    page_icon="🩺",
    layout="centered"
)

# --- Memuat Model ---
# Fungsi ini di-cache agar model tidak perlu di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model():
    """Memuat model SVC dari file .pkl."""
    try:
        with open('svc_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        st.error("Error: File 'svc_model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

model = load_model()

# --- Konfigurasi Koneksi Database (Supabase/PostgreSQL) ---
DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres.poyqtorvllnwvrqmctnl',
    'password': 'Raesa2345*', 
    'host': 'aws-1-ap-southeast-1.pooler.supabase.com',   
    'port': '6543'
}
# Menggunakan koneksi yang di-cache untuk efisiensi
@st.cache_resource
def get_db_connection():
    """Membuat koneksi ke database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Error koneksi database: {e}")
        return None

# --- UI Aplikasi Streamlit ---
st.title('🩺 Aplikasi Prediksi Diabetes')
st.write("Aplikasi ini menggunakan model *Support Vector Classifier* (SVC) untuk memprediksi apakah seseorang berisiko terkena diabetes berdasarkan data medis.")

# Pilihan mode di sidebar
st.sidebar.header('Pilih Mode Prediksi')
prediction_mode = st.sidebar.radio(
    "Pilih sumber data:",
    ('Prediksi Manual', 'Prediksi dari Database')
)

# --- Logika untuk setiap mode ---

if model is None:
    st.warning("Model tidak dapat dimuat. Fungsionalitas prediksi tidak akan berjalan.")
else:
    # Mode 1: Prediksi Manual dengan Input Form
    if prediction_mode == 'Prediksi Manual':
        st.header('Masukkan Data Pasien Secara Manual')
        st.write("Gunakan slider atau input box di bawah ini untuk memasukkan data.")
        
        with st.form("manual_input_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pregnancies = st.number_input('Jumlah Kehamilan (Pregnancies)', min_value=0, max_value=20, value=1, step=1)
                glucose = st.slider('Kadar Glukosa (Glucose)', 50, 200, 110)
                blood_pressure = st.slider('Tekanan Darah (BloodPressure)', 20, 130, 72)
                skin_thickness = st.slider('Ketebalan Kulit (SkinThickness)', 0, 100, 20)
                
            with col2:
                insulin = st.slider('Kadar Insulin', 0, 900, 79)
                bmi = st.slider('Indeks Massa Tubuh (BMI)', 15.0, 70.0, 32.0, 0.1)
                dpf = st.slider('Riwayat Diabetes Keluarga (DiabetesPedigreeFunction)', 0.0, 2.5, 0.47, 0.001)
                age = st.number_input('Usia (Age)', min_value=1, max_value=120, value=33, step=1)
            
            submit_button = st.form_submit_button(label='🔮 Lakukan Prediksi')

        if submit_button:
            features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
            prediction = model.predict(features)
            outcome = 'Berisiko Diabetes' if prediction[0] == 1 else 'Tidak Berisiko Diabetes'
            
            st.subheader('Hasil Prediksi')
            if outcome == 'Berisiko Diabetes':
                st.error(f"**Hasil:** {outcome}")
            else:
                st.success(f"**Hasil:** {outcome}")
            
            st.write("---")
            st.write("Data yang Anda Masukkan:")
            
            input_data = {
                "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
                "DiabetesPedigreeFunction": dpf, "Age": age
            }
            st.json(input_data)

    # Mode 2: Prediksi dari Database berdasarkan ID
    elif prediction_mode == 'Prediksi dari Database':
        st.header('Ambil Data Pasien dari Database')
        patient_id = st.text_input('Masukkan ID Pasien:')
        
        if st.button('🔍 Cari & Lakukan Prediksi'):
            if not patient_id:
                st.warning("Mohon masukkan ID pasien terlebih dahulu.")
            else:
                conn = get_db_connection()
                if conn is None:
                    st.error("Gagal terhubung ke database. Periksa kembali konfigurasi Anda.")
                else:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("SELECT Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age FROM diabetes_inference WHERE id = %s;", (patient_id,))
                            data = cur.fetchone()
                        
                        if data is None:
                            st.error(f'Data dengan ID {patient_id} tidak ditemukan.')
                        else:
                            features = np.array(data).reshape(1, -1)
                            prediction = model.predict(features)
                            outcome = 'Berisiko Diabetes' if prediction[0] == 1 else 'Tidak Berisiko Diabetes'
                            
                            st.subheader(f'Hasil Prediksi untuk Pasien ID: {patient_id}')
                            if outcome == 'Berisiko Diabetes':
                                st.error(f"**Hasil:** {outcome}")
                            else:
                                st.success(f"**Hasil:** {outcome}")

                            st.write("---")
                            st.write("Data dari Database:")
                            
                            feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                            db_data_df = pd.DataFrame([data], columns=feature_names)
                            st.table(db_data_df)

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat pemrosesan: {e}")
                    finally:
                        if conn:
                            conn.close()

