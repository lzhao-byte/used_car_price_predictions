import streamlit as st
from utils.data_explorer import *
import numpy as np

st.subheader("Exploration", divider=True)
st.markdown("In this section, you may visualize patterns, explore relationships for an overview of the data.")




def generate_visuals(df):
    config_options = {'height': 500}
    data_explorer = DataExp(df)
    cols = [col for col in df.columns if df[col].dtype in (pl.Int64, pl.Float64) or df[col].n_unique() <= 50]

    overview, feature_dist, joint_dist, feature_target, target_tab = st.tabs([
        "Overview", "Single Attribute", "Double Attributes", "Correlations", "Target"
    ])

    with overview:
        summary = data_explorer.describe()
        st.dataframe(summary)

        fig = data_explorer.show_nulls()
        st.plotly_chart(fig, config=config_options,)

    with feature_dist:
        features = st.multiselect(
            "Test a Feature",
            options = sorted(cols)
        )

        left_col, mid_col, right_col = st.columns(3)
        for idx, feature in enumerate(features):
            config_options = {'height': 200}
            fig = data_explorer.show_feature_dist(feature)
            if idx % 3 == 0:
                left_col.plotly_chart(fig, config=config_options,)
            elif idx % 3 == 2:
                right_col.plotly_chart(fig, config=config_options,)
            else:
                mid_col.plotly_chart(fig, config=config_options,)
  
    
    with joint_dist:
        feature_a = st.selectbox(
            "Select one Feature",
            options = sorted(cols),
            index=0
        )
        opt = sorted(cols)
        opt.remove(feature_a)

        feature_b = st.selectbox(
            "Select another Feature",
            options = opt,
            index=0
        )
        include_target = st.checkbox("Include Target?")
        if include_target:
            target = st.text_input("Target Column Name", value='price', key='target_input_jd_tab')
        if st.button("Draw"):
            config_options = {'height': 300}
            left, right = st.columns(2)
            fig = data_explorer.show_feature_joint_dist(feature_a, feature_b)
            left.plotly_chart(fig, config=config_options,)
            if include_target:
                fig2 = data_explorer.show_feature_joint_dist(feature_a, feature_b, target)
                right.plotly_chart(fig2, config=config_options,)

    with feature_target:
        feature = st.selectbox(
            "Select a Feature",
            options = sorted(cols),
            index=0
            )
        target = st.text_input("Target Column Name", value='price', key='target_input_ft_tab')
        left, right = st.columns(2)
        log_price = left.checkbox("Use log value for target?")
    
        if target.lower() in df.columns:
            if st.button("Show"):
                fig = data_explorer.explore_feature_target(feature, target.lower(), log_target=log_price)
                st.plotly_chart(fig, config=config_options,)
        else:
            st.toast("Target column name error. Please double check.")
            
        if right.checkbox("Show Correlation Matrix?"):
            fig = data_explorer.show_correlation_matrix()
            st.plotly_chart(fig)


    with target_tab:
        config_options = {'height': 400}
        left_box, mid_box, right_box = st.columns(3)
        target = left_box.text_input("Target Column Name", value='price', key='target_input_target_tab')
        
        if target.lower() in df.columns:
            trim_method = mid_box.radio("Trim Method", options=['By %', 'By Value'], horizontal=True)
            if trim_method == 'By %':
                trim_prct = right_box.slider("Select a trim % (both tail)", min_value=1, max_value=20, step=1)
                fig1, fig2 = data_explorer.check_target_dist(trimprct=trim_prct/100)
            else:
                values = right_box.slider("Select min/max values to trim", 
                                          min_value=int(df[target.lower()].min()), 
                                          max_value=int(df[target.lower()].quantile(0.999)),
                                          value=[100,100000], step=500)
                fig1, fig2 = data_explorer.check_target_dist(left_end=values[0], right_end=values[1])
            left_fig, right_fig = st.columns(2)
            left_fig.plotly_chart(fig1, config=config_options,)
            right_fig.plotly_chart(fig2, config=config_options,)











if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.toast("No Data Available. Load Data First.")
    else:
        if len(st.session_state.data) > 1:
            data_choice = st.sidebar.selectbox("Select a dataset for visualization",
                                options=st.session_state.data.keys(),
                                format_func=str.title,
                                key='data_selection',
                                index=0)
            df = st.session_state.data[data_choice]
        else:
            df = st.session_state.data['raw']

        generate_visuals(df)
        