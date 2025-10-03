import streamlit as st
from utils.feature_engs import *


def tweak_features(df):
    st.subheader("Feature Engineering", divider=True)
    st.markdown("In this section, you may add features, transform features, etc.")

    feature_engineer = FeatureEng(df)
    with st.expander("Sample Rows"):
        st.dataframe(feature_engineer.show_samples())

    st.subheader("Add Features", divider=True)
    
    age_col, miles_col, loc_col, cond_col, map_col = st.columns(5)
    if age_col.button("Add Vehicle Age"):
        feature_engineer.add_features(add_age=True)
        st.session_state.data['final'] = feature_engineer.final
        st.rerun()
    
    if miles_col.button("Add Annual Mileage"):
        errors = feature_engineer.add_features(add_annual_mileage=True)
        if errors is not None:
            st.warning(errors)
        else:
            st.session_state.data['final'] = feature_engineer.final
            st.rerun()

    if loc_col.button("Group Locations"):
        feature_engineer.add_features(group_latlon=True)
        st.session_state.data['final'] = feature_engineer.final
        st.rerun()

    if cond_col.button("Encode Condition"):
        feature_engineer.recode_condition()
        st.session_state.data['final'] = feature_engineer.final
        st.rerun()
    
    all_cols = feature_engineer.get_columns()
    if 'group_region' in feature_engineer.get_columns():
        if map_col.button("Show Groups"):
            fig = feature_engineer.show_latlon_groups()
            _, fig_col, _ = st.columns([0.5,1,0.5])
            fig_col.plotly_chart(fig, key='groups_from_lat_lon')

    st.subheader("Select Features", divider=True)
    sel_cols = st.multiselect("Select Columns to be considered in the model:", 
                 options = sorted(all_cols))
    if st.button("Confirm"):
        feature_engineer.select_columns(sel_cols)
        st.session_state.data['final'] = feature_engineer.final
        st.rerun()


if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.toast("No Data Available. Load Data First.")
    elif 'clean' not in st.session_state.data:
        st.toast("Data not cleaned yet. Prep First.")
    else:
        if 'final' in st.session_state.data:
            st.toast("Engineered data exist. Continue working.")
            data_choice = st.sidebar.selectbox("Select a dataset to continue",
                                options=['clean', 'final'],
                                format_func=str.title,
                                key='feature_engineering',
                                index=0)
            df = st.session_state.data[data_choice]
        else:
            df = st.session_state.data['clean']

        tweak_features(df)