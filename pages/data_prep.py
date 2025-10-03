import streamlit as st
from utils.data_process import *


st.subheader("Data Cleaning", divider=True)
st.markdown("In this section, you may clean data, select attributes.")


def prep_clean_data(df, ref, words):
    config_options={'height': 400}
    data_cleanser = DataPrep(df, ref, words)
    with st.expander("Correct Data Types"):
        st.divider()
        st.markdown("Current schema:")
        st.dataframe(data_cleanser.show_schema())
        st.markdown("Sample rows: ")
        st.dataframe(data_cleanser.show_sample_values())

        st.divider()
        col, from_type, to_type = st.columns(3)
        dt_col = col.text_input("Type the column to fix", value='posting_date')
        from_dtype = from_type.selectbox("From Data Type", options=['String', 'Number'])
        to_dtype = to_type.selectbox("To Data Type", options=['Date', 'Number', 'String'])
        
        if st.button("Correct Data Type"):
            data_cleanser.correct_types(dt_col=dt_col, from_data_type=from_dtype, to_data_type=to_dtype)
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()


    with st.expander("Handle Missing Values"):
        st.divider()
        st.markdown("Sample rows: ")
        st.dataframe(data_cleanser.show_sample_values())
        st.divider()
        sel_col = st.selectbox("Select a column to start", 
                               options=sorted(data_cleanser.clean.columns),
                               index=0)
        datatype, null_counts, null_perc = data_cleanser.get_nulls(sel_col)
        st.caption(f"Data Type: {datatype}; Nulls: {null_counts: ,d}; % Nulls: {null_perc: .1f}%.")
        handle_null_method = st.radio("Choose a strategy to handle missing values",
                                      options=['Remove Null Values', 'Drop Entire Column', 'Fill with "Other"', 'Impute with Mode/Median'],
                                      horizontal=True)
        if st.button("Confirm"):
            data_cleanser.handle_nulls(sel_col, method=handle_null_method)
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()
        if st.button("One-Click Clean"):
            data_cleanser.handle_nulls_all()
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()


    with st.expander("Remove Duplicates"):
        st.divider()
        subset=['year', 'manufacturer', 'model', 'odometer', 
                'title_status', 'condition', 'drive', 'transmission', 
                'price']
        samples = data_cleanser.identify_duplicates(subset)
        st.markdown("Sample Duplicates (Potential)")
        st.dataframe(samples.head(50))

        if st.button("Remove Duplicates"):
            data_cleanser.remove_duplicates(cols=subset)
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()


    with st.expander("Clean Messy Strings"):
        st.divider()
        st.markdown("Sample rows: ")
        st.dataframe(data_cleanser.show_sample_values())
        left, right, _ = st.columns(3)
        if left.button("Sample random column"):
            st.markdown(data_cleanser.show_sample_col_values())
        if right.button("Sample model"):
            st.markdown(data_cleanser.show_sample_col_values("model"))
        st.divider()
        if st.button("Clean string columns"):
            with st.spinner():
                data_cleanser.clean_string_cols()
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()
        if st.button("Show Clean Data"):
            if 'model_clean' in data_cleanser.clean.columns:
                st.dataframe(data_cleanser.clean.select(['manufacturer', 'make_clean', 'model', 'model_clean']).sample(50))
            else:
                st.warning("Column not cleaned yet. Clean the column first.")


    with st.expander("Remove Extreme Values"):
        st.divider()
        cols = data_cleanser.get_cols(dtype='numeric')
        cols = set(cols) - set(['id', 'year', 'lat', 'long'])
        sel_col = st.selectbox("Select a column to start: ", 
                               options = cols,
                               index=0)
        trim_method = st.radio("Trim Method", options=['By %', 'By Value', 'By IQR'], horizontal=True)
        if trim_method == 'By %':
            trim_prct = st.slider("Select a trim % (both tail)", min_value=1, max_value=20, step=1)
        elif trim_method == 'By Value':
            values = st.slider("Select min/max values to trim", 
                                        min_value=int(data_cleanser.clean[sel_col].min()), 
                                        max_value=int(data_cleanser.clean[sel_col].quantile(0.999)),
                                        value=[1,1000], step=1)
            
        if st.button("Submit"):
            if trim_method == 'By %':
                data_cleanser.trim_feature(sel_col, trimprct=trim_prct/100)
            elif trim_method == 'By Value':
                data_cleanser.trim_feature(sel_col, left_end=values[0], right_end=values[1])
            else:
                data_cleanser.trim_feature(sel_col, trim_method='IQR')

            st.session_state['data']['clean'] = data_cleanser.clean

        fig = data_cleanser.show_feature_dist(sel_col)
        st.plotly_chart(fig, config=config_options, key=f"extreme_value_{sel_col}")
        
        
    with st.expander("Limit Scope"):
        st.divider()
        left, right = st.columns(2)
        fig = data_cleanser.draw_latlon()
        left.plotly_chart(fig, config=config_options)

        if left.button("Limit to Contiguous US Boundary"):
            data_cleanser.trim_latlon()
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()

        fig = data_cleanser.show_feature_dist(col='year')
        right.plotly_chart(fig, config=config_options)

        limit_age = right.selectbox("Limit Vehicle Age to Yrs:", 
                                 options = [10, 20, 30, 40],
                                 index=3)
        if right.button("Confirm", key='limit_age_confirm'):
            data_cleanser.trim_age(limits=limit_age)
            st.session_state['data']['clean'] = data_cleanser.clean
            st.rerun()




if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.toast("No Data Available. Load Data First.")
    else:
        if len(st.session_state.data) > 1:
            data_choice = st.sidebar.selectbox("Select a dataset to clean",
                                options=['raw', 'clean'],
                                format_func=str.title,
                                key='data_selection',
                                index=0)
            df = st.session_state.data[data_choice]
        else:
            df = st.session_state.data['raw']
        ref = st.session_state.ref
        words = st.session_state.words

        prep_clean_data(df, ref, words)


