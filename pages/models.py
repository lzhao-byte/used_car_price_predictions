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
        st.caption("""Once your data is clean, the next important step is to split it into training and test sets. 
        This is especially crucial in supervised learning, 
        where the model learns patterns from the training data and is then evaluated on the test data.
        The test set remains completely separate from the training process, providing a more realistic estimate of how the model will perform on new, unseen data.
        This helps ensure that the model generalizes well and doesn't just memorize the training examples.""")
        st.caption("""In the past, the size of the test set was a major consideration due to limited data availability. 
        However, with the increasing accessibility of large datasets, 
        the impact of test size on model performance has become less critical. 
        Still, it's important to maintain a reasonable split 
        (commonly 70–80% for training and 20–30% for testing) to ensure that the model isn't 
        just memorizing the training data but can also make accurate predictions on new inputs.
        """)
        test_size = st.radio("Select a test size for evalution",
                            options=[0.3, 0.2, 0.1],
                            index=1,
                            horizontal=True,
                            format_func=lambda x: f"{x*100:.0f}%")

    with st.expander("Encoding Categorical Variables"):
        st.caption("""Many machine learning models require numerical input to process data effectively. 
        However, real-world datasets often include categorical variables—such as names, colors, or categories—that are represented as text. 
        Encoding is the process of converting these categorical values into numbers so that models can interpret and learn from them. 
        Different models may prefer different encoding techniques. 
        For example, some models (like decision trees) can handle label encoding well, 
        while others (like linear models) may perform better with one-hot encoding. """)
        st.caption("""Although some modern algorithms can handle categorical data internally, 
        it's generally a good practice to include encoding as part of your data preprocessing pipeline. 
        This gives you more control over how categories are represented and ensures consistency across training and prediction.
         Below are two commonly used encoding methods, among others: 
         Label Encoding assigns a unique number to each category, One-Hot Encoding creates a new binary column 
         for each category.""")
        sel_train_opt = st.radio("Select an Encoding Method",
                             options=[
                                 "one-hot",
                                 "ordinal",
                             ],
                             horizontal=True,
                             format_func=lambda x: f"Train with {x.title()} Encoding",
                            )

    with st.expander("Model Selection"):
        st.caption("""The models displayed here are examples from a wide range of regression and classification algorithms 
        you can choose from. In practice, it is common to experiment with multiple models and 
        evaluate their performance using predefined metrics, such as accuracy, precision, recall, or RMSE. 
        This comparison helps identify the model that performs best for your specific problem and dataset, 
        which can then be selected as the final production model. 
        Alternatively, you might combine the strengths of different models using 
        ensemble techniques to achieve better overall performance.""")
        st.caption("""It is also important to understand the underlying mechanics of each algorithm, 
        especially when designing a custom machine learning pipeline. 
        For instance, k-nearest neighbors (KNN) relies heavily on calculating distances between data points, 
        making it sensitive to feature scaling. In contrast, 
        tree-based models (like decision trees or random forests) handle categorical variables 
        well and are less affected by feature scaling.""")        
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
    with st.expander("Hyperparameter Tuning"):
        st.caption("""Hyperparameters are settings or configurations that you choose before training a model. 
        They are not learned from the data — instead, they control how the learning process happens.""")
        st.caption("""Choosing the right hyperparameters can make a huge difference in how well your model performs. 
        Multiple tuning methods are available, in this app, random search is done when hyperparameter tuning is enabled.""")
        st.caption('*It will usually take a while with hyperparameter tuning on, due to a range of parameters to be tested.*')
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