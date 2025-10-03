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
    st.markdown("This is a demo of predictive analytics workflow. Load the data and click a step on the sidebar to explore.")

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