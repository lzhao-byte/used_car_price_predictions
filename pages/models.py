import streamlit as st
from utils.model_trains import *


def model_training(df):
    st.subheader("Model Training", divider=True)
    st.markdown("In this section, you may select algorithm, train model, etc.")

    with st.expander("Sample Input"):
        st.dataframe(df)

    model_trainer = ModelBuilder(df=df)
    model_trainer._split_data()

    
    with st.expander("Train/Test Split"):
        st.caption("Generally, after your data is clean, the next thing you want to do is to split data into training and test set. This step is especially critical in supervised learning, since you will have training set to train the model, and the test set remains separate from model training process to provide a more realistic view of your model performance as if it were given a complete new collected data. Traditionally test size matters a lot due to data size, but with larger and larger data set available at lower cost, the influence of test size over model performance is shrinking.")
        test_size = st.radio("Select a test size for evalution",
                            options=[0.3, 0.2, 0.1],
                            index=1,
                            horizontal=True,
                            format_func=lambda x: f"{x*100:.0f}%")

    with st.expander("Encoding Categorical Variables"):
        st.caption('Encoding is key when you have categorical variables in the input data set since most models require numbers as input to be able to evaluate similarities/differences among the observations. Different models may require or work better with certain encoding methods, some modern models can also handle encoding internally. Generally it is a better practice to have encoding in your modeling pipeline so that you have a better control over how you would like to convert categories/text into numbers. Below are two commonly used encoding methods among others.')
        sel_train_opt = st.radio("Select an Encoding Method",
                             options=[
                                 "one-hot",
                                 "ordinal",
                             ],
                             horizontal=True,
                             format_func=lambda x: f"Train with {x.title()} Encoding",
                            )

    with st.expander("Model Selection"):
        st.caption('The models shown here are some examples that you can select from a wide range of regression/classification algorithms. In practical, you can have a list of models and compare them against predefined performance metrics, then select the best model as your final production model. Alternatively, you can also leverage advantages of different types of models and ensemble them into a better performed model.')
        st.caption('You may also want to pay attention to the inner logic of different algorithms should you design your own pipeline. For example, nearest neighbors rely heavily on calculating pairwise distance among observations, while tree-based models handles categorical variable well.')
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
                        format_func=lambda x: x.replace("_", " ").title(),
                    )
     
    # scale_target = st.checkbox("Scale Target")
    enable_tuning = st.checkbox("Enable Hyperparameter Tuning")


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