import streamlit as st
import plotly.express as px
import time
import numpy as np
import pandas as pd

def simulate(model, df, train_data_summary, train_model_perf):
    max_step = int(len(df)/10)
    placeholder = st.empty()
    left, mid, right = st.columns(3)


    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'slider_value' not in st.session_state:
        st.session_state.slider_value = 0
    if 'initial_slider' not in st.session_state:
            st.session_state.initial_slider = 0

    with placeholder:
        slider = st.slider("Simulate The Flow of New Data",
                min_value=0,
                max_value=max_step,
                value=st.session_state.slider_value,
                step=10,
                key='initial_slider'
                )
        if np.sum(slider) != st.session_state.slider_value:
            st.session_state.slider_value = np.sum(slider)
        
    st.subheader("Data", divider=True)
    data_container = st.empty()
    st.subheader("Predictions", divider=True)
    model_container = st.empty()
    stats_container = st.empty()

    if left.button("Run Animation"):
        st.session_state.running = True
    if mid.button("Stop Animation"):
        st.session_state.running = False
        update_data_model(data_container, model_container, stats_container,
                          train_data_summary,train_model_perf,
                           df, model, model.target, st.session_state.slider_value)
    if right.button("Reset Animation"):
        st.session_state.running = False
        st.session_state.slider_value = 0
        st.rerun()


    # Animation logic
    if st.session_state.running:
        for i in range(st.session_state.slider_value, max_step+1, 1):
            if not st.session_state.running: # Allow stopping mid-animation
                break
            with placeholder:
                st.slider("Simulate the Flow of New Data", 0, max_step, i, step=1)
            st.session_state.slider_value = i
            update_data_model(data_container, model_container, stats_container,
                              train_data_summary,train_model_perf,
                              df, model, model.target, st.session_state.slider_value)
            time.sleep(2) # Control animation speed
        
        # Reset running state if animation completes
        if st.session_state.slider_value == max_step:
            st.session_state.running = False
            st.session_state.slider_value = 0 # Reset slider for next run
            st.rerun() # Rerun to update button and slider


@st.fragment
def update_data_model(data_container, model_container, stats_container, 
                      train_data_summary,train_model_perf,
                      df, model, target_col, nrows=0, input_col='odometer'):
    
    inputs = df.head(nrows*10).reset_index(drop=True).copy()
    if len(inputs):
        inputs.loc[len(inputs)-10:,target_col]=None
        y_pred = model.predict(data=inputs.drop(columns=target_col))
        inputs['predicted_price'] = y_pred
        data_container.dataframe(inputs, hide_index=True)
        dt_for_test = inputs[[target_col, 'predicted_price']].dropna()

        if len(dt_for_test):
            left, right = model_container.columns(2)
            left.plotly_chart(model.plot_pred_vs_true(y_true=dt_for_test[target_col], y_pred=dt_for_test['predicted_price'], title='Predictions vs. Observations'))
            right.plotly_chart(model.plot_data_dist(y_train=model.input['x_train'][input_col], y_new=inputs.dropna()[input_col], title=f'New vs. Training Data ({input_col})'))

            model_perf = model.evaluate(y_true=dt_for_test[target_col], y_pred=dt_for_test['predicted_price'])
            summary_mdf = pd.DataFrame.from_dict([train_model_perf, model_perf]).T
            summary_mdf.columns = ['Training', 'New']

            data_summary = model.get_data_summary(inputs.dropna()[input_col])
            summary_df = pd.DataFrame.from_dict([train_data_summary, data_summary]).T
            summary_df.columns = ['Training', 'New']
            
            left, right = stats_container.columns(2)
            left.dataframe(summary_mdf.T)
            right.dataframe(summary_df.T)


if __name__ == "__main__":
    st.subheader("Monitoring (Simulation)", divider=True)

    if 'model' not in st.session_state:
        st.toast("No Model Available. Build Model First.")
    else:
        model = st.session_state.model
        df = model.sim
        train_data_summary = model.get_data_summary(model.input['x_train']['odometer'])
        train_model_perf = model.evaluate()
        simulate(model, df, train_data_summary, train_model_perf)
