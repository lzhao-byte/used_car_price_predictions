import streamlit as st
from utils.model_trains import *


def model_training(df):
    st.subheader("Model Training", divider=True)
    st.markdown("In this section, you may select algorithm, train model, etc.")

    with st.expander("Sample Input"):
        st.dataframe(df)

    model_trainer = ModelBuilder(df=df)
    model_trainer._split_data()
    sel_model = st.radio("Select a Model",
                      options=[
                          "linear",
                          "nearest_neighbors",
                          "neural_network",
                          "decision_tree",
                          "random_forest",
                          "xgboost",
                        ],
                        horizontal=True,
                        format_func=lambda x: x.replace("_", " ").title()
                    )
        
    sel_train_opt = st.radio("Select an Encoding Method",
                             options=[
                                 "one-hot",
                                 "ordinal",
                             ],
                             horizontal=True,
                             format_func=lambda x: f"Train with {x.title()} Encoding")
    # scale_target = st.checkbox("Scale Target")
    enable_tuning = st.checkbox("Enable Hyperparameter Tuning")

    with st.expander("Train/Test Split"):
        test_size = st.radio("Select a test size for evalution",
                            options=[0.3, 0.2, 0.1],
                            index=1,
                            horizontal=True,
                            format_func=lambda x: f"{x*100:.0f}%")

    if st.button("Train"):
        placeholder = st.container()
        for message in model_trainer.train(sel_model, sel_train_opt, test_size, enable_tuning=enable_tuning):
            placeholder.markdown(message)
        if "error" not in message.lower():
            st.info("Training Complete.")
            st.session_state['model'] = model_trainer
        else:
            st.warning("Training Error.")

    st.subheader("Model Visuals", divider=True)
    if 'model' in st.session_state:
        model = st.session_state['model']
        left, middle, right = st.columns(3)
        if left.button("Show Model Description"):
            st.write(model._show_model())
        if middle.button("Show Feature Importance"):
            st.pyplot(model._plot_feature_importance())
        if right.button("Show Model Structure"):
            if model.family=='tree':
                st.pyplot(model._plot_tree())
            if model.family=='linear':    
                st.markdown(model._show_structure())
        
        with st.expander("Examine Result"):
            x_test, y_test = model.input['x_test'], model.input['y_test']
            y_pred = model.predict()
            st.subheader("Metrics", divider=True)
            st.write(model_trainer.evaluate( y_test, y_pred))
            st.subheader("Predictions", divider=True)
            st.plotly_chart(model.plot_pred_vs_true(y_test, y_pred))
  


if __name__ == "__main__":
    if 'data' not in st.session_state:
        st.toast("No Data Available. Load Data First.")
    elif 'final' not in st.session_state.data:
        st.toast("Data not ready yet. Prep First.")
    else:
        df = st.session_state.data['final']
        model_training(df)