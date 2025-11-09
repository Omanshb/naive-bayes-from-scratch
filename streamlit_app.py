import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from naive_bayes_from_scratch import (
    GaussianNB, MultinomialNB, BernoulliNB,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)


def load_sample_datasets():
    """Load built-in classification datasets."""
    datasets = {}
    
    try:
        from sklearn.datasets import load_iris
        iris = load_iris()
        datasets['Iris (Classification)'] = {
            'data': pd.DataFrame(iris.data, columns=iris.feature_names),
            'target': iris.target,
            'target_names': iris.target_names,
            'description': 'Iris dataset - 3 classes, 4 continuous features'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer()
        datasets['Breast Cancer (Classification)'] = {
            'data': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'target': cancer.target,
            'target_names': cancer.target_names,
            'description': 'Breast cancer dataset - binary, continuous features'
        }
    except ImportError:
        pass
    
    try:
        from sklearn.datasets import load_wine
        wine = load_wine()
        datasets['Wine (Classification)'] = {
            'data': pd.DataFrame(wine.data, columns=wine.feature_names),
            'target': wine.target,
            'target_names': wine.target_names,
            'description': 'Wine dataset - 3 classes, 13 continuous features'
        }
    except ImportError:
        pass
    
    return datasets


def create_confusion_matrix_plot(cm, class_names):
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[str(c) for c in class_names],
        y=[str(c) for c in class_names],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        width=500,
        height=500
    )
    
    return fig


def create_feature_distributions_plot(X, y, feature_names, target_names):
    """Create feature distribution plots per class."""
    n_features = min(6, X.shape[1])  # Show up to 6 features
    n_cols = 2
    n_rows = (n_features + 1) // 2
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[feature_names[i] for i in range(n_features)]
    )
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    
    for i in range(n_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        for idx, class_val in enumerate(np.unique(y)):
            mask = y == class_val
            fig.add_trace(
                go.Histogram(
                    x=X[mask, i],
                    name=f'Class {target_names[idx]}',
                    opacity=0.7,
                    marker_color=colors[idx % len(colors)],
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text='Feature Distributions by Class',
        height=300 * n_rows,
        barmode='overlay'
    )
    
    return fig


def create_probability_distribution_plot(y_true, y_proba, n_classes):
    """Create probability distribution plot."""
    if n_classes == 2:
        df_plot = pd.DataFrame({
            'Probability': y_proba[:, 1],
            'True Class': [f'Class {y}' for y in y_true]
        })
        
        fig = px.histogram(
            df_plot, x='Probability', color='True Class',
            nbins=30, barmode='overlay',
            title='Predicted Probability Distribution (Class 1)',
            labels={'Probability': 'Predicted Probability for Class 1'}
        )
        
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Decision Threshold")
    else:
        max_proba = np.max(y_proba, axis=1)
        df_plot = pd.DataFrame({
            'Max Probability': max_proba,
            'True Class': [f'Class {y}' for y in y_true]
        })
        
        fig = px.histogram(
            df_plot, x='Max Probability', color='True Class',
            nbins=30, barmode='overlay',
            title='Maximum Predicted Probability Distribution'
        )
    
    fig.update_layout(width=700, height=500)
    return fig


def create_roc_curve_plot(y_true, y_proba, n_classes):
    """Create ROC curve for binary classification."""
    if n_classes != 2:
        return None
    
    thresholds = np.linspace(0, 1, 100)
    tpr_list = []
    fpr_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    
    auc = np.trapz(tpr_list, fpr_list)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr_list, y=tpr_list,
        mode='lines',
        name=f'ROC Curve (AUC = {abs(auc):.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=700,
        height=500
    )
    
    return fig


def create_class_distribution_plot(y_train, y_test, target_names):
    """Create class distribution bar plot."""
    train_counts = np.bincount(y_train.astype(int))
    test_counts = np.bincount(y_test.astype(int))
    
    df_plot = pd.DataFrame({
        'Class': list(target_names) * 2,
        'Count': list(train_counts) + list(test_counts),
        'Set': ['Train'] * len(train_counts) + ['Test'] * len(test_counts)
    })
    
    fig = px.bar(
        df_plot, x='Class', y='Count', color='Set',
        barmode='group',
        title='Class Distribution in Train and Test Sets'
    )
    
    fig.update_layout(width=600, height=400)
    return fig


def create_prior_probabilities_plot(model, target_names):
    """Visualize prior probabilities."""
    if not hasattr(model, 'class_prior_'):
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=[str(name) for name in target_names],
            y=model.class_prior_,
            marker_color='lightblue',
            text=[f'{p:.3f}' for p in model.class_prior_],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Prior Probabilities P(y)',
        xaxis_title='Class',
        yaxis_title='Probability',
        width=600,
        height=400
    )
    
    return fig


def create_feature_importance_plot(model, feature_names):
    """Create feature importance plot based on variance (for Gaussian NB)."""
    if not hasattr(model, 'theta_') or not hasattr(model, 'sigma_'):
        return None
    
    # Calculate importance as inverse of average variance
    importance = 1.0 / (np.mean(model.sigma_, axis=0) + 1e-10)
    importance = importance / importance.sum()  # Normalize
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:10]  # Top 10
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker_color='lightgreen'
        )
    ])
    
    fig.update_layout(
        title='Feature Importance (based on variance)',
        xaxis_title='Relative Importance',
        yaxis_title='Feature',
        width=700,
        height=400
    )
    
    return fig


def create_2d_decision_boundary(X, y, model, feature_names):
    """Create 2D decision boundary plot."""
    if X.shape[1] != 2:
        return None
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = go.Figure()
    
    # Decision boundary
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, h),
        y=np.arange(y_min, y_max, h),
        z=Z,
        colorscale='Viridis',
        showscale=True,
        opacity=0.6,
        colorbar=dict(title="Predicted Class")
    ))
    
    # Data points
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    for idx, class_val in enumerate(np.unique(y)):
        mask = y == class_val
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'Class {class_val}',
            marker=dict(
                size=10,
                color=colors[idx % len(colors)],
                line=dict(width=2, color='white')
            )
        ))
    
    fig.update_layout(
        title='Decision Boundary Visualization',
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1],
        width=700,
        height=500
    )
    
    return fig


def main():
    st.set_page_config(
        page_title="Naive Bayes from Scratch",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Naive Bayes from Scratch")
    st.markdown("Implementation using probabilistic Bayes' Theorem for classification")
    
    st.sidebar.header("Data Selection")
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Built-in Datasets", "Upload CSV"]
    )
    
    df = None
    target = None
    target_names = None
    dataset_name = ""
    
    if data_source == "Built-in Datasets":
        datasets = load_sample_datasets()
        
        if not datasets:
            st.error("No built-in datasets available.")
            return
        
        dataset_choice = st.sidebar.selectbox(
            "Select dataset:",
            list(datasets.keys())
        )
        
        if dataset_choice:
            dataset = datasets[dataset_choice]
            df = dataset['data']
            target = dataset['target']
            target_names = dataset['target_names']
            dataset_name = dataset_choice
            
            st.sidebar.info(f"**{dataset_choice}**\n\n{dataset['description']}")
    
    elif data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                dataset_name = uploaded_file.name
                
                if len(df.columns) < 2:
                    st.sidebar.error("Dataset must have at least 2 columns")
                    df = None
                    target = None
                else:
                    target_col = st.sidebar.selectbox(
                        "Select target column:",
                        list(df.columns)
                    )
                    
                    if target_col:
                        target = df[target_col].values
                        df = df.drop(columns=[target_col])
                        unique_classes = np.unique(target)
                        target_names = unique_classes
                        
                        if len(unique_classes) < 2:
                            st.sidebar.error("Target must have at least 2 classes")
                            df = None
                            target = None
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {str(e)}")
                df = None
                target = None
    
    if df is not None and target is not None:
        st.header(f"Dataset: {dataset_name}")
        
        # Data quality checks
        with st.expander("Data Quality Checks", expanded=False):
            checks = []
            checks.append("✓ Dataset loaded successfully")
            
            if df.isnull().any().any():
                checks.append("⚠️ Contains missing values")
            else:
                checks.append("✓ No missing values")
            
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                checks.append(f"⚠️ Non-numeric columns: {', '.join(non_numeric)}")
            else:
                checks.append("✓ All features are numeric")
            
            for check in checks:
                st.text(check)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", df.shape[0])
        with col2:
            st.metric("Features", df.shape[1])
        with col3:
            n_classes = len(np.unique(target))
            st.metric("Classes", n_classes)
        
        with st.expander("Dataset Preview"):
            preview_df = df.copy()
            preview_df['Target'] = target
            st.dataframe(preview_df.head(10))
        
        st.subheader("Model Configuration")
        
        feature_selection = st.multiselect(
            "Select features (leave empty for all):",
            list(df.columns),
            default=[]
        )
        
        if len(feature_selection) == 0:
            feature_selection = list(df.columns)
        
        # Data validation
        non_numeric_cols = df[feature_selection].select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric_cols:
            st.error(f"❌ Non-numeric features: {', '.join(non_numeric_cols)}")
            st.error("Please encode categorical variables or remove them.")
            st.stop()
        
        # Handle missing values
        if df[feature_selection].isnull().any().any():
            st.warning("⚠️ Removing rows with missing values...")
            mask = ~df[feature_selection].isnull().any(axis=1)
            df = df[mask]
            target = target[mask]
            st.info(f"Reduced to {len(df)} samples")
        
        # Check for infinite values
        if np.isinf(df[feature_selection].values).any():
            st.error("❌ Infinite values detected")
            st.stop()
        
        # Check sample size
        if len(df) < 10:
            st.error("❌ Not enough samples (minimum 10)")
            st.stop()
        
        X = df[feature_selection].values
        y = target
        
        # Encode target if needed
        try:
            y = np.array(y)
            if not np.issubdtype(y.dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.info(f"Target encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        except Exception as e:
            st.error(f"❌ Error processing target: {str(e)}")
            st.stop()
        
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError:
            st.warning("⚠️ Cannot stratify - using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        st.markdown("#### Naive Bayes Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            nb_type = st.selectbox(
                "Naive Bayes Type",
                ["Gaussian", "Multinomial", "Bernoulli"],
                help="Gaussian: continuous features | Multinomial: count features | Bernoulli: binary features"
            )
        
        with col2:
            if nb_type == "Gaussian":
                var_smoothing = st.select_slider(
                    "Variance Smoothing",
                    options=[1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
                    value=1e-9,
                    format_func=lambda x: f"{x:.0e}",
                    help="Stability parameter added to variance"
                )
            else:
                alpha = st.slider(
                    "Alpha (Smoothing)",
                    0.0, 2.0, 1.0, 0.1,
                    help="Laplace smoothing parameter"
                )
        
        if nb_type == "Bernoulli":
            binarize = st.slider(
                "Binarize Threshold",
                0.0, 1.0, 0.5, 0.1,
                help="Threshold for converting features to binary"
            )
        
        # Feature scaling option (for Gaussian NB, often helpful)
        if nb_type == "Gaussian":
            scale_features = st.checkbox(
                "Scale features",
                value=True,
                help="Standardize features to have mean=0, std=1"
            )
            
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
        
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {nb_type} Naive Bayes..."):
                try:
                    if nb_type == "Gaussian":
                        model = GaussianNB(var_smoothing=var_smoothing)
                    elif nb_type == "Multinomial":
                        model = MultinomialNB(alpha=alpha)
                    else:  # Bernoulli
                        model = BernoulliNB(alpha=alpha, binarize=binarize)
                    
                    model.fit(X_train, y_train)
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    y_train_proba = model.predict_proba(X_train)
                    y_test_proba = model.predict_proba(X_test)
                    
                    st.session_state['model'] = model
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['y_train_pred'] = y_train_pred
                    st.session_state['y_test_pred'] = y_test_pred
                    st.session_state['y_train_proba'] = y_train_proba
                    st.session_state['y_test_proba'] = y_test_proba
                    st.session_state['feature_names'] = feature_selection
                    st.session_state['target_names'] = target_names
                    st.session_state['n_classes'] = n_classes
                    st.session_state['nb_type'] = nb_type
                    
                    st.success("Model trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Training error: {str(e)}")
                    import traceback
                    with st.expander("Error details"):
                        st.code(traceback.format_exc())
        
        if 'model' in st.session_state:
            model = st.session_state['model']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            y_train_pred = st.session_state['y_train_pred']
            y_test_pred = st.session_state['y_test_pred']
            y_train_proba = st.session_state['y_train_proba']
            y_test_proba = st.session_state['y_test_proba']
            feature_names = st.session_state['feature_names']
            target_names = st.session_state['target_names']
            n_classes = st.session_state['n_classes']
            nb_type = st.session_state['nb_type']
            
            st.header("Model Results")
            
            average_type = 'binary' if n_classes == 2 else 'macro'
            
            train_acc = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred, average=average_type)
            train_recall = recall_score(y_train, y_train_pred, average=average_type)
            train_f1 = f1_score(y_train, y_train_pred, average=average_type)
            
            test_acc = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred, average=average_type)
            test_recall = recall_score(y_test, y_test_pred, average=average_type)
            test_f1 = f1_score(y_test, y_test_pred, average=average_type)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Training Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{train_acc:.4f}")
                    st.metric("Precision", f"{train_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{train_recall:.4f}")
                    st.metric("F1 Score", f"{train_f1:.4f}")
            
            with col2:
                st.subheader("Test Metrics")
                met_col1, met_col2 = st.columns(2)
                with met_col1:
                    st.metric("Accuracy", f"{test_acc:.4f}")
                    st.metric("Precision", f"{test_precision:.4f}")
                with met_col2:
                    st.metric("Recall", f"{test_recall:.4f}")
                    st.metric("F1 Score", f"{test_f1:.4f}")
            
            st.markdown("---")
            
            with st.expander("Model Details"):
                st.write(f"**Naive Bayes Type:** {nb_type}")
                st.write("**Prior Probabilities:**")
                for i, prob in enumerate(model.class_prior_):
                    st.write(f"- P(Class {target_names[i]}) = {prob:.4f}")
            
            st.subheader("Visualizations")
            
            tab_names = ["Feature Distributions", "Confusion Matrix", "Probability Distribution", "Class Distribution"]
            
            if n_classes == 2:
                tab_names.append("ROC Curve")
            
            if hasattr(model, 'class_prior_'):
                tab_names.append("Prior Probabilities")
            
            if nb_type == "Gaussian" and hasattr(model, 'theta_'):
                tab_names.append("Feature Importance")
            
            if len(feature_names) == 2:
                tab_names.append("Decision Boundary")
            
            viz_tabs = st.tabs(tab_names)
            
            tab_idx = 0
            
            # Feature Distributions
            with viz_tabs[tab_idx]:
                try:
                    fig_dist = create_feature_distributions_plot(
                        X_test, y_test, feature_names, target_names
                    )
                    st.plotly_chart(fig_dist, width='stretch')
                    st.info("Feature distributions show how each feature varies across different classes")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            tab_idx += 1
            
            # Confusion Matrix
            with viz_tabs[tab_idx]:
                cm = confusion_matrix(y_test, y_test_pred)
                fig_cm = create_confusion_matrix_plot(cm, target_names)
                st.plotly_chart(fig_cm, width='stretch')
                st.info("Confusion matrix shows classification performance")
            tab_idx += 1
            
            # Probability Distribution
            with viz_tabs[tab_idx]:
                fig_prob = create_probability_distribution_plot(y_test, y_test_proba, n_classes)
                st.plotly_chart(fig_prob, width='stretch')
                st.info("Shows distribution of predicted probabilities")
            tab_idx += 1
            
            # Class Distribution
            with viz_tabs[tab_idx]:
                fig_class = create_class_distribution_plot(y_train, y_test, target_names)
                st.plotly_chart(fig_class, width='stretch')
                st.info("Class distribution in train and test sets")
            tab_idx += 1
            
            # ROC Curve
            if n_classes == 2:
                with viz_tabs[tab_idx]:
                    fig_roc = create_roc_curve_plot(y_test, y_test_proba, n_classes)
                    if fig_roc:
                        st.plotly_chart(fig_roc, width='stretch')
                        st.info("ROC curve shows true positive vs false positive rate")
                tab_idx += 1
            
            # Prior Probabilities
            if hasattr(model, 'class_prior_'):
                with viz_tabs[tab_idx]:
                    fig_prior = create_prior_probabilities_plot(model, target_names)
                    if fig_prior:
                        st.plotly_chart(fig_prior, width='stretch')
                        st.info("Prior probabilities P(y) learned from training data")
                tab_idx += 1
            
            # Feature Importance
            if nb_type == "Gaussian" and hasattr(model, 'theta_'):
                with viz_tabs[tab_idx]:
                    fig_importance = create_feature_importance_plot(model, feature_names)
                    if fig_importance:
                        st.plotly_chart(fig_importance, width='stretch')
                        st.info("Feature importance based on variance (lower variance = more important)")
                tab_idx += 1
            
            # Decision Boundary
            if len(feature_names) == 2:
                with viz_tabs[tab_idx]:
                    try:
                        fig_boundary = create_2d_decision_boundary(
                            X_test, y_test, model, feature_names
                        )
                        if fig_boundary:
                            st.plotly_chart(fig_boundary, width='stretch')
                            st.info("2D decision boundary visualization")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    else:
        st.info("Please select a data source from the sidebar to get started!")


if __name__ == "__main__":
    main()
