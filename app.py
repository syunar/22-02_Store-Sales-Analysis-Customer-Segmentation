# import library
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import altair as alt
import plotly.express as px

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.cluster import KMeans

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Customer Segmentation')

#---------------------------------#
##############################################################################
# read data
def load_data():
    files = 'https://raw.githubusercontent.com/syunar/22-02_Store-Sales-Analysis-Customer-Segmentation/main/sample-store.csv'
    df = pd.read_csv(files)
    return df
###############################################################################
# clean data
def clean_data(df_raw):
    df = df_raw.copy()

    # standardize column names
    df.columns = df.columns.str.strip().str.lower()\
                        .str.replace(" ","_")\
                        .str.replace("-","_")\
                        .str.replace("\/","_")\
                        .str.replace(".","")

    # standardize missing values
    df = df.replace(['NO DATA','N/A', 'null', 'Empty', '?', \
                                 'NaN', '', 'nan'],np.nan)
    
    # drop unused columns
    un_cols = ['row_id']
    df.drop(un_cols, axis=1, inplace=True)

    # drop na
    df = df.dropna()

    # drop duplicated datas
    df.drop_duplicates()

    # covert numeric datatype
    nums = ['sales', 'quantity', 'discount','profit']
    for num in nums:
        df[num] = df[num].apply(pd.to_numeric)

    # covert datatime datatype
    dtimes = ['order_date', 'ship_date']
    for dtime in dtimes:
        df[dtime] = df[dtime].apply(pd.to_datetime)

    # create new column discount_tf
    df['discount_tf'] = np.where(df['discount'] > 0, 1, 0)

    return df

############################################################################
# rfm segmentation
def rfm_segm(clean_df):
    df_rfm = clean_df.copy()

    # recency = time since last order
    # frequency = total number of orders
    # monetary = total sales

    df_rfm = df_rfm.groupby('customer_id')\
                .agg({'order_date': lambda x:(dt.datetime(2021,1,1)-x.max()).days,
                    'order_id': lambda x: len(x),
                    'sales': lambda x: x.sum()}) \
                .rename(columns={'order_date': 'recency',
                                'order_id': 'frequency',
                                'sales': 'monetary'}) \
                .reset_index()

    # Discretizer
    trans = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    temp = trans.fit_transform(df_rfm[['recency', 'frequency', 'monetary']])
    rfmscore = pd.DataFrame(temp.astype('int')
                            + 1, columns=['r_score','f_score','m_score'])

    # Flip value of r_score first. If value = 5,
    # it means the last purchase date is long so after filp it should be 1
    rfmscore['r_score'] = 6 - rfmscore['r_score']

    # join dataframe
    df_rfm = df_rfm.join(rfmscore)

    # add column that use for categories
    df_rfm['rfm_score'] = (df_rfm['r_score'].astype(str)
                                + df_rfm['f_score'].astype(str)
                                + df_rfm['m_score'].astype(str)) \
                                .astype('category')

    return df_rfm

###############################################################################
# k mean cluster
def rfm_cluster(df_rfm, n_clusters=5):

    # create data for kmean clustering
    data = df_rfm.drop(['customer_id', 'recency', 'frequency', 'monetary', 'rfm_score'], axis=1).copy()


    # create model baseline for select 'k'
    min_range = 2
    max_range = 16
    inertia = []
    k_list = range(min_range, max_range)

    for k in k_list:
        km = KMeans(n_clusters=k,
                    random_state=42)
        km.fit(data)
        score = km.inertia_
        inertia.append(score)

    # plot inertia vs n_clusters
    source = pd.DataFrame({'n_cluster': k_list,
                            'inertia':inertia})

    line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
        alt.X('n_cluster', title='Number of  Clusters'),
        alt.Y('inertia', title='Inertia')
    ).properties(
        title='To find the best of n_clusters on elbow point'
    )

    st.altair_chart(line_chart)

    
    st.subheader(f'Build Model KMeans(n_clusters={n_clusters})')

    # create model
    km_best = KMeans(n_clusters=n_clusters, random_state=42)
    km_best.fit(data)
    data['cluster'] = km_best.predict(data)


    st.markdown('**Number of Customers in each Clusters**')
    # plot value counts
    pt = data['cluster'].value_counts().reset_index() \
            .rename(columns={'cluster':'count',
                             'index':'cluster'})
    pt['cluster'] = pt['cluster'].astype('int')
    bars = alt.Chart(pt).mark_bar().encode(
        y=alt.X('cluster:N',sort='-x'),
        x='count:Q'
    ).properties(
        title='Number of Customers in each Clusters'
    )
    st.altair_chart(bars)

    fig = px.pie(pt, values='count', names='cluster')
    st.write(fig)





    # plot each cluster to find relation
    pt = data.groupby('cluster').agg({'r_score': 'mean',
                                'f_score': 'mean',
                                'm_score': 'mean'}).reset_index()


    #   covert from wide to long format
    pt = pd.melt(pt, id_vars='cluster', value_vars=['r_score', 'f_score', 'm_score'])

    bars = alt.Chart(pt).mark_bar().encode(
        x=alt.X('variable', title=None, sort=['r_score', 'f_score', 'm_score']),
        y=alt.Y('value', title = 'Score'),
        color='variable',
        column=alt.Column('cluster')
    ).properties(
        title='RFM Score for each Cluster'
    )


    st.altair_chart(bars)


    # return
    df_rfm['cluster'] = data['cluster']
    return df_rfm
################################################################################
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')    
################################################################################
def pipeline():
    df_rfm = rfm_segm(clean_df)
    df_rfm = rfm_cluster(df_rfm, n_clusters)

    st.subheader('Output CSV')
    st.write(df_rfm)
    csv = convert_df(df_rfm)
    st.download_button(
        "Press to Download Output CSV",
        csv,
        "df_rfm.csv",
        "text/csv",
        key='download-csv'
        )
################################################################################
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

with st.sidebar.header('2. Select Number of Clusters'):
    n_clusters = st.sidebar.slider('Select Number of Clusters', 2, 16, 5, 1)
################################################################################
st.write("""
# Customer Segmentation
by using RFM Customer Segmentation methon and then use KMeans Clustering
""")
################################################################################
# Main panel

# Displays the dataset
st.subheader('Dataset')


if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    clean_df = clean_data(df_raw)
    st.markdown('**Glimpse of dataset**')
    st.write(clean_df.head())
    pipeline()
else:
    st.info('Awaiting for CSV file to be uploaded.')
    st.markdown('This dataset is used as the example can upload your CSV on the left sidebar.')
    df_raw = load_data()
    clean_df = clean_data(df_raw)
    st.write(clean_df.head())
    pipeline()
        

