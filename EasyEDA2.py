import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
from matplotlib.gridspec import GridSpec
from pandas.api.types import is_numeric_dtype, is_object_dtype
from scipy.stats import percentileofscore
import math

class CategoricalFeatureHandler:
    def __init__(self, dataset):
        self.df = dataset.copy()
        
    def create_categories_info(self, cat_feature, num_feature):
        df = self.df
        
        info_df = (
            df.groupby(cat_feature)
            .agg(
                Median=(num_feature, np.nanmedian),
                Mean=(num_feature, np.nanmean),
                RelMeanDiff=(
                    num_feature,
                    lambda x: (np.nanmean(x) - np.nanmedian(x)) / np.nanmedian(x) * 100
                    if np.nanmedian(x) > 0
                    else 0,
                ),
            )
            .add_prefix(f"{num_feature} ")
        )
         
        for measure in ("Median", "Mean"):
            non_nan_values = df.loc[~df[num_feature].isna(), num_feature]
            info_df[f"{num_feature} {measure}Pctl."] = [
                percentileofscore(non_nan_values, score)
                for score in info_df[f"{num_feature} {measure}"]
            ]

        info_df["Counts"] = df[cat_feature].value_counts()
        info_df["Counts Ratio"] = df[cat_feature].value_counts(normalize=True)
        self.info_df = info_df
        
        self._provide_consistent_cols_order()
        return self.info_df.copy()
    
    def _provide_consistent_cols_order(self):
        (
            self._median_name,
            self._mean_name,
            self._rel_mean_diff_name,
            self._median_pctl_name,
            self._mean_pctl_name,
            self._counts_name,
            self._counts_ratio_name,
        ) = self.info_df.columns

        self.info_df = self.info_df[
            [
                self._counts_name,
                self._counts_ratio_name,
                self._median_name,
                self._median_pctl_name,
                self._mean_name,
                self._mean_pctl_name,
                self._rel_mean_diff_name,
            ]
        ]

        self._n_categories_in = self.info_df.shape[0]
        self._n_stats_in = self.info_df.shape[1]
        self._stat_names_in = self.info_df.columns
        
    def categories_info_plot(self, cat_feature, num_feature, palette="mako_r"):
        self.create_categories_info(cat_feature, num_feature)

        fig_height = 8
        if self._n_categories_in > 5:
            fig_height += (self._n_categories_in - 5) * 0.5

        fig = plt.figure(figsize=(12, fig_height), tight_layout=True)
        
        plt.suptitle(
            f"{cat_feature} vs {self._counts_name} & {self._median_name} & {self._rel_mean_diff_name}"
        )
        gs = GridSpec(nrows=2, ncols=3, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])  # Counts.
        ax2 = fig.add_subplot(gs[0, 1])  # Median.
        ax3 = fig.add_subplot(gs[0, 2])  # Relative Mean Diff.
        ax4 = fig.add_subplot(gs[1, :])  # Descriptive Stats.

        for ax, stat_name in zip(
            (ax1, ax2, ax3),
            (self._counts_name, self._median_name, self._rel_mean_diff_name),
        ):
            self._plot_category_vs_stat_name(ax, stat_name)
            if not ax == ax1:
                plt.ylabel("")

        self._draw_descriptive_stats(ax4)
        sns.set_palette("deep")  # Default palette.
        tab1.pyplot(fig)
        #plt.show()
   
    def _plot_category_vs_stat_name(self, ax, stat_name):
        """Plots a simple barplot (`category` vs `stat_name`) in the current axis."""
        info_df = self.info_df
        order = info_df.sort_values(stat_name, ascending=False).index
        plt.sca(ax)
        plt.yticks(rotation=30)
        sns.barplot(data=info_df, x=stat_name, y=info_df.index, order=order)

    def _draw_descriptive_stats(self, ax4):
        """Draws info from the `info_df` at the bottom of the figure."""
        plt.sca(ax4)
        plt.ylabel("Descriptive Statistics", fontsize=12, weight="bold")
        plt.xticks([])
        plt.yticks([])

        # Spaces between rows and cols. Default axis has [0, 1], [0, 1] range,
        # thus we divide 1 by number of necessary rows / columns.
        xspace = 1 / (self._n_stats_in + 1)  # +1 due to one for a category.
        yspace = 1 / (self._n_categories_in + 1 + 1)  # +2 due to wide header.

        xpos = xspace / 2
        ypos = 1 - yspace
        wrapper = lambda text, width: "\n".join(line for line in wrap(text, width))

        for header in np.r_[["Category"], self._stat_names_in]:
            header = wrapper(header, 15)  # Wrap headers longer than 15 characters.
            plt.text(xpos, ypos, header, ha="center", va="center", weight="bold")
            xpos += xspace

        pattern = "{};{};{:.2%};{:,.1f};{:.0f};{:,.1f};{:.0f};{:+.2f}"
        category_stats = [pattern.format(*row) for row in self.info_df.itertuples()]

        for i, cat_stats in enumerate(category_stats):
            ypos = 1 - (5 / 2 + i) * yspace
            plt.axhline(ypos + yspace / 2, color="black", linewidth=5)
            for j, cat_stat in enumerate(cat_stats.split(";")):
                xpos = (1 / 2 + j) * xspace
                plt.text(xpos, ypos, cat_stat, ha="center", va="center")

def calculate_stats(dataframe):
    result_df = pd.DataFrame(columns=['Attribute', 'Mean', 'Median', 'Rel Mn-Md Diff'])   
    for i, column in enumerate(dataframe.columns):
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            mean = dataframe[column].mean()
            median = dataframe[column].median()
            relative_difference = abs(mean - median) / (median)*100 if median > 0 else 0
            temp_df = pd.DataFrame(
                {'Attribute': column, 
                 'Mean': mean, 
                 'Median': median, 
                 'Rel Mn-Md Diff': relative_difference
                }, index =[i])
            if median >= 10:
                result_df = pd.concat([result_df, temp_df], ignore_index=True)
    return result_df


image_path = 'data_insights_dashboard.png'


uploaded_file = st.sidebar.file_uploader("Choose a file")
st.sidebar.divider()

@st.cache_data
def load_data(data_file):
    file_extension = data_file.name.split('.')[-1].lower()
    if file_extension in ['csv']:
        return pd.read_csv(data_file)
    elif file_extension in ['xls', 'xlsx']:
        return pd.read_excel(data_file)
    elif file_extension in ['json']:
        return pd.read_json(data_file)
    elif file_extension in ['parquet']:
        return pd.read_parquet(data_file)
    else:
        st.sidebar.warning("Please upload a valid data file to proceed.")

if uploaded_file is not None:
    data_df = load_data(uploaded_file)
    
    data_cat = data_df.select_dtypes(include="object")
    data_num = data_df.select_dtypes(include="number")
    data_num_cols = data_num.columns

    st.sidebar.header("Data Cleaning Options")
    if st.sidebar.checkbox("Drop rows with missing values"):
        data_df = data_df.dropna()
    if st.sidebar.checkbox("Fill missing values"):
        fill_value = st.sidebar.text_input("Fill value for missing entries")
        if fill_value:
            data_df = data_df.fillna(fill_value)
    
    selected_cat = st.sidebar.selectbox("Categorical Attribute", list(data_cat))
    selected_num = st.sidebar.selectbox("Numerical Attribute", list(data_num))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Chart", "🗃 Data", ":thermometer: Heat Map", "🔢 Outliers", "📊 Histograms", "🤖 Model Training"])
    
    with tab1:
        category_handler1 = CategoricalFeatureHandler(data_df)
        category_handler1.categories_info_plot(selected_cat, selected_num)
    
    with tab2:
        st.caption(f"Correlations between Categorical attribute {selected_cat} and Numerical Attribute {selected_num}")
        st.dataframe(data_df)
        
        container = st.container()
        colA, colB, colC = container.columns(3)
        colA.write('Number of Numerical Attributes:')
        colA.write(len(data_num.columns))
        colB.write('Number of Categorical Attributes:')
        colB.write(len(data_cat.columns))
        colC.write("Total number of records")
        colC.write(len(data_df))

        dataframe_select = st.sidebar.radio("Select dataframe", ["Full", "Numerical", "Categorical"])
        if dataframe_select == "Full":   
            st.write(data_df)
        elif dataframe_select == "Numerical":
            st.write(data_num)
        elif dataframe_select == "Categorical":
            st.write(data_cat)
            st.write("Categorical attributes details:")
        
    with tab3:
        toggle_heatmap = st.sidebar.checkbox("Full Heatmap")

        corr = data_num.corr()
        triu_mask_full = np.triu(corr)
        high_corr_cols = corr.loc[corr[target_attribute] > 0.6, target_attribute].index
        high_corr = data_num[high_corr_cols].corr()
        triu_mask = np.triu(high_corr)

        if toggle_heatmap:
            st.header("Intercorrelation Matrix Heatmap - Complete")
            fig_hm = plt.figure(figsize=(10, 10))
            plt.style.use('dark_background')
            sns.heatmap(corr, square=True, annot=False, mask=triu_mask_full)
            st.pyplot(fig_hm)
        else:
            st.header("Intercorrelation Matrix Heatmap - Salients")
            fig_hm = plt.figure(figsize=(10, 10))
            plt.style.use('dark_background')
            sns.heatmap(high_corr, square=True, annot=True, linewidth=2, mask=triu_mask, cmap='mako')
            st.pyplot(fig_hm)

    with tab4:
        target_attribute = st.sidebar.selectbox(f"Target Attribute for heatmap & outliers", list(data_num_cols))
        st.header(f"Outliers versus target: **{target_attribute}**")
        data_num_colsx = data_num_cols.drop(["Id", "SalePrice"])
        stats = calculate_stats(data_num[data_num_colsx])
        features = list(stats[stats['Rel Mn-Md Diff'] > 5]['Attribute'])
        num_rows = math.ceil(len(features) / 3)
        outliers = list(data_df[features].max() * 0.8)

        fig_outliers, axes = plt.subplots(nrows=num_rows, ncols=3, figsize=(12, 8), tight_layout=True, sharey=True)
        for i, (feature, outlier) in enumerate(zip(features, outliers)):
            sns.scatterplot(x=data_df[feature], y=data_df[target_attribute], color="navy", ax=axes[i // 3, i % 3])
            df = data_df.loc[data_df[feature] > outlier, [feature, target_attribute]]
            sns.scatterplot(data=df, x=feature, y=target_attribute, ax=axes[i // 3, i % 3], color="red", marker="X")
        st.pyplot(fig_outliers)

    with tab5:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.style.use('dark_background')
        fig_hist, ax_hist = plt.subplots()
        data_num.hist(figsize=(20, 20), xlabelsize=10, ylabelsize=10, color='#D0E11C', bins=30)
        st.pyplot(fig_hist)
        st.dataframe(data_num.describe().T)
    
    with tab6:
        st.header("Model Training")
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error

        target = st.selectbox("Select Target Column", data_num.columns)
        features = st.multiselect("Select Feature Columns", data_num.columns.drop(target))
        
        if st.button("Train Model"):
            X = data_num[features]
            y = data_num[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Mean Squared Error: {mse}")

elif uploaded_file is None:
    st.sidebar.warning("Please upload a data file to proceed.")
    st.image(image_path, caption='Data Insights Dashboard')
    st.stop()
