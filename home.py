import streamlit as st
import polars as pl
import plotly.express as px
from utils.snowflake_functions import *

def setup_page():
    data_prep = st.Page(
        "pages/data_prep.py", title='Data Prep', icon=":material/data_check:"
    )
    data_expl = st.Page(
        "pages/data_viz.py", title='Data Explorer', icon=":material/data_exploration:"
    )
    feature_eng = st.Page(
        "pages/features.py", title='Feature Engineering', icon=":material/table_edit:"
    )
    model_build = st.Page(
        "pages/models.py", title='Model Training', icon=":material/model_training:"
    )
    sim_monitor = st.Page(
        "pages/simulator.py", title='Monitor Simulating', icon=":material/monitor_heart:"
    )   

    pg =st.navigation([
        data_expl,
        data_prep,
        feature_eng,
        model_build,
        sim_monitor
    ])

    pg.run()


@st.dialog("Warning")
def reset_all_warning():
    st.write("This will clear all the progress on this site!")
    if st.button("Confirm"):
        keys_to_clear = list(st.session_state.keys())
        for key in keys_to_clear:
            del st.session_state[key]
        st.rerun()


def main():
    st.set_page_config(layout="wide")
    st.title("Intro to Predictive Analytics")
    desc = """
    This app guides you through the key stages of a typical **predictive analytics pipeline**. Load the data and click a step on the sidebar to explore.
    """
    st.markdown(desc)
    
    side_desc = """
    - **Data Explorer**  
    Understand your dataset by visualizing distributions, identifying patterns, and spotting potential issues like missing values or outliers.

    - **Data Prep**  
    Prepare your data for modeling by handling missing values, correcting data types, and removing inconsistencies.

    - **Feature Engineering**  
    Transform raw data into meaningful features through encoding, scaling, and feature selection to improve model performance.

    - **Model Training**  
    Train and evaluate different machine learning models using your processed data, and compare their performance to select the best one.

    - **Monitor Simulating**  
    Simulate predictions on new data, visualize prediction accuracy, and monitor for data drift to ensure your model remains reliable over time.
    """
    st.sidebar.markdown(side_desc)

    load_col, reset_col, _ = st.columns([1,1,3])
    if load_col.button("Load Data", use_container_width=True):
        if 'data' not in st.session_state:
            df, ref, words = fetch_data(use_local=True)
            st.session_state['data'] = {}
            st.session_state['data']['raw'] = df
            st.session_state['ref'] = ref
            st.session_state['words'] = words
        st.toast("Success.")
    
    if reset_col.button("Reset All", use_container_width=True):
        reset_all_warning()
    
    setup_page()
    



if __name__ == "__main__":
    main()