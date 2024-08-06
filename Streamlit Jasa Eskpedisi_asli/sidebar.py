# sidebar.py
import streamlit as st
from streamlit_option_menu import option_menu

def show():
    with st.sidebar:
        st.markdown("# Aplikasi SentMart")
        pilihan = option_menu(
            menu_title=None,  # Judul menu
            options=["Dashboard", "Klasifikasi"],  # Opsi menu
            icons=["house", "clipboard"],  # Ikon untuk opsi
            default_index=0,  # Opsi yang dipilih secara default
        )
        return pilihan
