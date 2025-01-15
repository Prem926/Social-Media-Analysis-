import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from wordcloud import WordCloud

# Load the dataset
data = pd.read_csv(r"finalsocial.csv")

# Data preprocessing
# Convert columns to appropriate types
data['Post_Type'] = data['Post_Type'].astype(str)
data['User_Follower'] = pd.to_numeric(data['User_Follower'], errors='coerce').fillna(0)
data['Likes'] = pd.to_numeric(data['Likes'], errors='coerce').fillna(0)
data['Shares'] = pd.to_numeric(data['Shares'], errors='coerce').fillna(0)
data['Comments'] = pd.to_numeric(data['Comments'], errors='coerce').fillna(0)

# Calculate new features
data['Engagement_Rate'] = ((data['Likes'] + data['Comments'] + data['Shares']) / data['User_Follower'] * 100).fillna(0)
data['Viral_Score'] = (data['Shares'] / data['Likes'] * 100).fillna(0)
data['Interaction_Rate'] = (data['Comments'] / data['Likes'] * 100).fillna(0)
data['Total_Engagement'] = data['Likes'] + data['Comments'] + data['Shares']
data['Post_Performance'] = data.apply(lambda x: 'High' if x['Engagement_Rate'] > 5 else ('Medium' if x['Engagement_Rate'] > 2 else 'Low'), axis=1)

# Calculate additional features
data['Engagement_per_Post'] = data['Total_Engagement'] / data['User_Activity'].apply(lambda x: int(x.split()[0]))
data['Post_Frequency'] = data['User_Activity'].apply(lambda x: int(x.split()[0]))
data['Verification_Impact'] = (data['Account_Verification'] == 'Verified') & (data['Engagement_Rate'] > 5)
data['Content_Type_Popularity'] = data['Media_Type'].map({'video': 3, 'image': 2, 'text': 1})
data['Location_Influence'] = data.groupby('Location')['Engagement_Rate'].transform('mean')
data['Language_Influence'] = data.groupby('User_Language')['Engagement_Rate'].transform('mean')
data['Activity_Level'] = data['Post_Frequency'].apply(lambda x: 'High' if x > 5 else ('Medium' if x > 2 else 'Low'))
data['Follower_Growth_Potential'] = data['Engagement_Rate'] * 0.1  # Simplified estimation
data['Platform_Dominance'] = data.groupby('Social_Platform')['Engagement_Rate'].transform('max')

# Streamlit App
st.set_page_config(page_title="Advanced Social Media Analytics", layout="wide")
st.title("Advanced Social Media Engagement Analytics")

# Sidebar filters
st.sidebar.header("Filters")

# Follower Range Filter
follower_ranges = [
    "0-1K", "1K-5K", "5K-10K", "10K-50K", "50K-100K", "100K+"
]
select_all_followers = st.sidebar.checkbox("Select All Follower Ranges")
selected_follower_range = st.sidebar.multiselect(
    "Select Follower Range", follower_ranges, follower_ranges if select_all_followers else []
)

# Location Hierarchy
continents = {
    "North America": ["USA", "Canada"],
    "Europe": ["UK", "France", "Spain", "Italy"],
    "Asia": ["Japan", "Thailand", "Indonesia"],
    "Oceania": ["Australia"]
}
select_all_continents = st.sidebar.checkbox("Select All Continents")
selected_continent = st.sidebar.multiselect(
    "Select Continent", list(continents.keys()), list(continents.keys()) if select_all_continents else []
)
selected_countries = []
selected_cities=[]
if selected_continent:
    for continent in selected_continent:
        selected_countries.extend(continents[continent])
    select_all_cities = st.sidebar.checkbox("Select All Cities")
    selected_cities = st.sidebar.multiselect(
        "Select Cities", 
        data[data['Location'].str.contains('|'.join(selected_countries), na=False)]['Location'].unique(),
        data[data['Location'].str.contains('|'.join(selected_countries), na=False)]['Location'].unique() if select_all_cities else []
    )

# Other filters
post_types = data['Post_Type'].unique()
account_types = data['Account_Type'].unique()
person_types = data['Person_Type'].unique()
social_platforms = data['Social_Platform'].unique()
verification_status = data['Account_Verification'].unique()

select_all_post_types = st.sidebar.checkbox("Select All Post Types")
selected_post_types = st.sidebar.multiselect(
    "Select Post Types", post_types, post_types if select_all_post_types else []
)

select_all_account_types = st.sidebar.checkbox("Select All Account Types")
selected_account_types = st.sidebar.multiselect(
    "Select Account Types", account_types, account_types if select_all_account_types else []
)

select_all_person_types = st.sidebar.checkbox("Select All Person Types")
selected_person_types = st.sidebar.multiselect(
    "Select Person Types", person_types, person_types if select_all_person_types else []
)

select_all_platforms = st.sidebar.checkbox("Select All Social Platforms")
selected_platforms = st.sidebar.multiselect(
    "Select Social Platforms", social_platforms, social_platforms if select_all_platforms else []
)

select_all_verification = st.sidebar.checkbox("Select All Verification Status")
selected_verification = st.sidebar.multiselect(
    "Select Verification Status", verification_status, verification_status if select_all_verification else []
)

# Performance Filter
performance_metrics = ["High", "Medium", "Low"]
select_all_performance = st.sidebar.checkbox("Select All Performance Levels")
selected_performance = st.sidebar.multiselect(
    "Select Performance Level", performance_metrics, performance_metrics if select_all_performance else []
)

# Apply filters
filtered_data = data.copy()

if selected_follower_range:
    follower_mask = pd.Series(False, index=data.index)
    for range_str in selected_follower_range:
        if range_str == "0-1K":
            follower_mask |= (filtered_data['User_Follower'] < 1000)
        elif range_str == "1K-5K":
            follower_mask |= (filtered_data['User_Follower'].between(1000, 5000))
        elif range_str == "5K-10K":
            follower_mask |= (filtered_data['User_Follower'].between(5000, 10000))
        elif range_str == "10K-50K":
            follower_mask |= (filtered_data['User_Follower'].between(10000, 50000))
        elif range_str == "50K-100K":
            follower_mask |= (filtered_data['User_Follower'].between(50000, 100000))
        else:  # 100K+
            follower_mask |= (filtered_data['User_Follower'] > 100000)
    filtered_data = filtered_data[follower_mask]

if selected_cities:
    filtered_data =  filtered_data[filtered_data['Location'].isin(selected_cities)]
if selected_post_types:
    filtered_data = filtered_data[filtered_data['Post_Type'].isin(selected_post_types)]
if selected_account_types:
    filtered_data = filtered_data[filtered_data['Account_Type'].isin(selected_account_types)]
if selected_person_types:
    filtered_data = filtered_data[filtered_data['Person_Type'].isin(selected_person_types)]
if selected_platforms:
    filtered_data = filtered_data[filtered_data['Social_Platform'].isin(selected_platforms)]
if selected_verification:
    filtered_data = filtered_data[filtered_data['Account_Verification'].isin(selected_verification)]
if selected_performance:
    filtered_data = filtered_data[filtered_data['Post_Performance'].isin(selected_performance)]

# Display metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Average Engagement Rate", f"{filtered_data['Engagement_Rate'].mean():.2f}%")
with col2:
    st.metric("Average Viral Score", f"{filtered_data['Viral_Score'].mean():.2f}%")
with col3:
    st.metric("Average Interaction Rate", f"{filtered_data['Interaction_Rate'].mean():.2f}%")
with col4:
    st.metric("Total Posts", len(filtered_data))

# Display filtered data
st.subheader("Filtered Data")
st.dataframe(filtered_data)

# Advanced Analytics
st.header("Advanced Analytics")

# 1. Engagement Trends Over Time
if 'Date' in filtered_data.columns:
    st.subheader("Engagement Trends Over Time")
    fig_trends = px.line(filtered_data, x='Date', 
                         y=['Engagement_Rate', 'Viral_Score', 'Interaction_Rate'],
                         title="Engagement Metrics Over Time")
    st.plotly_chart(fig_trends)
else:
    st.subheader("Engagement Trends Over Time")
    st.write("Date column not found in the dataset.")

# 2. Heatmap for Location Influence
st.subheader("Location Influence on Engagement")
fig_location = px.choropleth(filtered_data, locations='Location', locationmode='country names',
                             color='Engagement_Rate', hover_name='Location',
                             title="Engagement Rate by Location")
st.plotly_chart(fig_location)

# 3. Sentiment Analysis Visualization
st.subheader("Sentiment Analysis of Comments")
sentiments = filtered_data['Comments'].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
fig_sentiment = px.histogram(sentiments, nbins=50, title="Sentiment Distribution")
st.plotly_chart(fig_sentiment)

# 4. User Segmentation
st.subheader("User Segmentation")
# Example clustering using KMeans
numeric_data = filtered_data[['Engagement_Rate', 'User_Follower']].dropna()
kmeans = KMeans(n_clusters=3)
filtered_data['Cluster'] = kmeans.fit_predict(numeric_data)
fig_clusters = px.scatter(filtered_data, x='Engagement_Rate', y='User_Follower', color='Cluster',
                          title="User Segmentation by Engagement and Followers")
st.plotly_chart(fig_clusters)

# 5. Predictive Analytics
st.subheader("Predictive Analytics")
# Example predictive model using Linear Regression
model = LinearRegression()
X = filtered_data[['Likes', 'Comments', 'Shares']].dropna()
y = filtered_data['Engagement_Rate'].dropna()
model.fit(X, y)
filtered_data['Predicted_Engagement'] = model.predict(X)

# Check if 'Date' column exists for plotting
if 'Date' in filtered_data.columns:
    fig_predictions = px.line(filtered_data, x='Date', y='Predicted_Engagement',
                              title="Predicted Engagement Over Time")
    st.plotly_chart(fig_predictions)
else:
    st.write("Date column not found in the dataset for predictive analytics visualization.")

# 6. Post Performance Distribution
st.subheader("Post Performance Distribution")
fig_performance = px.pie(filtered_data, names='Post_Performance', title="Distribution of Post Performance")
st.plotly_chart(fig_performance)

# 7. Engagement by Platform and Post Type
st.subheader("Engagement by Platform and Post Type")
fig_platform_post = px.box(filtered_data, x='Social_Platform', y='Engagement_Rate', color='Post_Type',
                           title="Engagement by Platform and Post Type")
st.plotly_chart(fig_platform_post)

# 8. Correlation Heatmap
st.subheader("Metric Correlations")
# Select only numeric columns for correlation
numeric_cols = filtered_data.select_dtypes(include=['float64', 'int64']).columns
correlation = filtered_data[numeric_cols].corr()
fig_corr = px.imshow(correlation, text_auto=True, title="Correlation between Metrics")
st.plotly_chart(fig_corr)

# 9. Performance Analysis by Account Type
st.subheader("Performance Analysis by Account Type")
fig_account_performance = px.sunburst(filtered_data, path=['Account_Type', 'Post_Performance'],
                                      title="Performance by Account and Post Type")
st.plotly_chart(fig_account_performance)

# 10. Follower Growth Potential
st.subheader("Follower Growth Potential")
fig_growth = px.scatter(filtered_data, x='User_Follower', y='Follower_Growth_Potential',
                        title="Follower Growth Potential vs. Current Followers")
st.plotly_chart(fig_growth)

# Footer with Animation
st.sidebar.markdown("---")
st.sidebar.subheader("Team Jugadoo")
st.sidebar.markdown("<marquee>Created by Team Jugadoo(Prem,Nikhil,Dhun,Harshita)</marquee>", unsafe_allow_html=True)

# Chatbot Integration
st.sidebar.subheader("Chatbot")
st.sidebar.write("Ask questions related to the dataset:")
# Assuming you have a chatbot function
# chatbot_response = chatbot_function(user_input)
# st.sidebar.write(chatbot_response)
