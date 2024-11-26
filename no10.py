import pickle
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Load model dari file .sav
model = pickle.load(open('model_prediksi_harga_mobil.sav', 'rb'))

# Judul aplikasi
st.title('Prediksi Harga Mobil')

# Menambahkan Sidebar
st.sidebar.title("Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["Dataset", "Grafik", "Prediksi"])


df1 = pd.read_csv('CarPrice_Assignment.csv')
# Menangani halaman berdasarkan pilihan pada sidebar
if page == "Dataset":
    st.header("Dataset")
    # Membuka file CSV  # Pastikan file CarPrice.csv tersedia di folder yang sama
    st.dataframe(df1)

elif page == "Grafik":
    st.header("Grafik Data Mobil")
    
    # Grafik Highway-mpg
    st.write("Grafik Highway-mpg")
    chart_highwaympg = pd.DataFrame(df1, columns=["highwaympg"])
    st.line_chart(chart_highwaympg)

    # Grafik curbweight
    st.write("Grafik Curbweight")
    chart_curbweight = pd.DataFrame(df1, columns=["curbweight"])
    st.line_chart(chart_curbweight)

    # Grafik horsepower
    st.write("Grafik Horsepower")
    chart_horsepower = pd.DataFrame(df1, columns=["horsepower"])
    st.line_chart(chart_horsepower)

elif page == "Prediksi":
    st.header("Prediksi Harga Mobil")
    st.write("Masukkan nilai untuk prediksi harga mobil:")
    
    # Input nilai dari variabel independen
    highwaympg = st.number_input("Highway MPG", min_value=0, step=1)
    curbweight = st.number_input("Curbweight", min_value=0, step=1)
    horsepower = st.number_input("Horsepower", min_value=0, step=1)

    # Tombol prediksi
    if st.button('Prediksi'):
        # Prediksi harga mobil
        car_prediction = model.predict([[highwaympg, curbweight, horsepower]])

        # Konversi ke string
        harga_mobil_str = np.array(car_prediction)
        harga_mobil_float = float(harga_mobil_str[0])  # Mengambil nilai prediksi
        harga_mobil_formatted = f"Rp{harga_mobil_float:,.2f}"  # Format harga dengan tanda ribuan

        # Tampilkan hasil prediksi
        st.success(f"Harga prediksi mobil: {harga_mobil_formatted}")
