import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from dateutil.relativedelta import relativedelta
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_echarts import st_echarts
import numpy as np
import datetime


st.set_page_config(layout="wide", initial_sidebar_state="auto")
# Define your custom CSS
custom_css = """
<style>
.my-container {
 background-color: #ffffff;
}
</style>
"""

# Simulate some social listening data
base_dir = os.path.dirname(__file__)
sourcesans3 = os.path.join(base_dir, 'SourceSans3.ttf')
sourcesans3bold = os.path.join(base_dir, 'SourceSans3Bold.ttf')
sourcesans3semibold = os.path.join(base_dir, 'SourceSans3SemiBold.ttf')


@st.cache_data
def loaddata():
    df_path = os.path.join(base_dir, 'corp.xlsx')
    df = pd.read_excel(df_path, sheet_name="Data")
    fanpage_df = pd.read_excel(df_path, sheet_name="Fanpage")
    df['PublishedDate'] = pd.to_datetime(df['PublishedDate']).dt.date
    fanpage_df['PublishedDate'] = pd.to_datetime(fanpage_df['PublishedDate']).dt.date
    return df, fanpage_df

df, fanpage_df = loaddata()

# Sidebar for filtering
st.sidebar.header("Filters")
# If this already returns datetime.date
min_date = df["PublishedDate"].min()  
max_date = df["PublishedDate"].max()

# Now use 'min_date' directly in date_input
start_date = st.sidebar.date_input(
    "Start Date",
    value=datetime.date(2024, 9, 1),
    min_value=min_date,
    max_value=max_date
)

end_date = st.sidebar.date_input(
    "End Date",
    value=datetime.date(2024, 9, 30),
    min_value=min_date,
    max_value=max_date
)
platforms = st.sidebar.multiselect("Select Platform", df['Channel'].unique(), df['Channel'].unique())
sentiment = st.sidebar.multiselect("Select Sentiment", df['Sentiment'].unique(), df['Sentiment'].unique())

# Convert start_date and end_date to datetime.date
start_date = pd.to_datetime(start_date).date()
end_date = pd.to_datetime(end_date).date()

def filter_and_compare(df, fanpage_df, start_date, end_date, platforms, sentiment):
    # Filter current period based on the user-selected filters
    current_period = df[(df['PublishedDate'] >= start_date) & 
                        (df['PublishedDate'] <= end_date) & 
                        (df['Channel'].isin(platforms)) & 
                        (df['Sentiment'].isin(sentiment))]
    
    # Calculate the previous period (same day, previous month)
    prev_start = start_date - relativedelta(months=1)
    prev_end = end_date - relativedelta(months=1)
    
    previous_period = df[(df['PublishedDate'] >= prev_start) & 
                         (df['PublishedDate'] <= prev_end) & 
                         (df['Channel'].isin(platforms)) & 
                         (df['Sentiment'].isin(sentiment))]

    # Filter fanpage_df based on the current date range and the same period logic
    filtered_fanpage_df_current = fanpage_df[(fanpage_df['PublishedDate'] >= start_date) & 
                                              (fanpage_df['PublishedDate'] <= end_date)]
    
    filtered_fanpage_df_previous = fanpage_df[(fanpage_df['PublishedDate'] >= prev_start) & 
                                               (fanpage_df['PublishedDate'] <= prev_end)]

    return current_period, previous_period, filtered_fanpage_df_current, filtered_fanpage_df_previous


# Call the function and filter data based on user input
current_period, previous_period, filtered_fanpage_df, filtered_fanpage_df_previous = filter_and_compare(df, fanpage_df, start_date, end_date, platforms, sentiment)
current_period['PublishedDate'] = pd.to_datetime(current_period['PublishedDate']).dt.date
previous_period['PublishedDate'] = pd.to_datetime(previous_period['PublishedDate']).dt.date
filtered_fanpage_df['PublishedDate'] = pd.to_datetime(filtered_fanpage_df['PublishedDate']).dt.date
filtered_fanpage_df_previous['PublishedDate'] = pd.to_datetime(filtered_fanpage_df_previous['PublishedDate']).dt.date

############## Plot the data ##################
#==================================Fanpage Follower======================
with st.container(key = "container", height=200, border = False):
        
    def mention_count(current_period, previous_period):
        # 1. total mentions
        # Count mentions in current and previous period
        current_period['PublishedDate'] = pd.to_datetime(current_period['PublishedDate'])
        previous_period['PublishedDate'] = pd.to_datetime(previous_period['PublishedDate'])
        current_mentions = current_period['Id'].nunique()
        previous_mentions = previous_period['Id'].nunique()
        
        # Calculate change in mentions
        change_in_mentions = current_mentions - previous_mentions
        change_in_mentions_percent = (change_in_mentions / previous_mentions) * 100 if previous_mentions > 0 else 0

        # average perday mentions
        date_diff1 = (current_period['PublishedDate'].max() - current_period['PublishedDate'].min()).days
        if date_diff1 == 0:
            current_mentions_per_day = current_mentions  # Default to 0 mentions per day
        else:
            current_mentions_per_day = current_mentions / (current_period['PublishedDate'].max() - current_period['PublishedDate'].min()).days
        # Calculate the number of days in the previous period
        date_diff = (previous_period['PublishedDate'].max() - previous_period['PublishedDate'].min()).days

        # Avoid division by zero
        if date_diff == 0:
            previous_mentions_per_day = previous_mentions  # Default to 0 mentions per day
        else:
            previous_mentions_per_day = previous_mentions / date_diff
        change_in_mentions_per_day = current_mentions_per_day - previous_mentions_per_day
        change_in_mentions_per_day_percent = (change_in_mentions_per_day / previous_mentions_per_day) * 100 if previous_mentions_per_day > 0 else 1

        # Audien scale
        curent_unique_authors = current_period['AuthorId'].nunique()
        previous_unique_authors = previous_period['AuthorId'].nunique()
        change_in_unique_authors = curent_unique_authors - previous_unique_authors #audience scale
        change_in_unique_authors_percent = (change_in_unique_authors / previous_unique_authors) * 100 if previous_unique_authors > 0 else 0
        # Sentiment NSR
        # Step 1: Remove blank sentiments
        def gen_nsr(df):
            filtered_df = df[df['Sentiment'].str.strip() != '']  # Exclude blank or whitespace-only rows

            # Step 2: Count positive and negative sentiments
            positive_count = filtered_df['Sentiment'].str.lower().value_counts().get('positive', 0)
            negative_count = filtered_df['Sentiment'].str.lower().value_counts().get('negative', 0)

            # Step 3: Calculate total sentiments (excluding blanks)
            total_count = filtered_df['Sentiment'].count()

            # Step 4: Calculate percentages
            percent_positive = (positive_count / total_count) * 100 if total_count > 0 else 0
            percent_negative = (negative_count / total_count) * 100 if total_count > 0 else 0

            # Step 5: Calculate NSR
            if percent_positive + percent_negative != 0:  # To avoid division by zero
                nsr = (percent_positive - percent_negative) / (percent_positive + percent_negative)
            else:
                nsr = 0
            return nsr
        current_nsr = gen_nsr(current_period)
        previous_nsr = gen_nsr(previous_period)
        change_in_nsr = current_nsr - previous_nsr
        # Total engagement
        # Step 1: Convert columns to numeric, coercing errors to NaN
        numeric_columns = ['Likes', 'Comments', 'Views', 'Shares']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Step 2: Replace NaN values with 0 for summation
        df.fillna(0, inplace=True)

        # Step 3: Calculate total engagement as the sum of the specified columns
        df['Total_Engagement'] = df[numeric_columns].sum(axis=1)

        # Step 4: Calculate the overall total engagement
        total_engagement = df['Total_Engagement'].sum()


        return current_mentions, change_in_mentions_percent, current_mentions_per_day, change_in_mentions_per_day_percent, change_in_mentions_per_day, curent_unique_authors, change_in_unique_authors, change_in_unique_authors_percent, current_nsr, total_engagement
    current_mentions, change_in_mentions_percent, current_mentions_per_day, change_in_mentions_per_day_percent, change_in_mentions_per_day, curent_unique_authors, change_in_unique_authors, change_in_unique_authors_percent, current_nsr, total_engagement = mention_count(current_period, previous_period)
    def calculate_metrics(df, gained_col='Follower Gain', lost_col='Follower Lost'):
        # Total followers gained and lost
        total_followers_gained = df[gained_col].sum()
        total_followers_lost = df[lost_col].sum()
        df['Follower'] = pd.to_numeric(df['Follower'], errors='coerce')
        df['Follower'].fillna(0, inplace=True)
        most_recent_date = filtered_fanpage_df["PublishedDate"].max()

        # Filter the row corresponding to the most recent date
        most_recent_row = filtered_fanpage_df[filtered_fanpage_df["PublishedDate"] == most_recent_date]

        # Get the Follower value
        if not most_recent_row.empty:
            total_follower = most_recent_row["Follower"].iloc[0]  # Assuming "Follower" is the column name
        else:
            total_follower = None  # Handle cases where the dataframe is empty        # Net follower growth
        net_growth = total_followers_gained + total_followers_lost
        net_growth = int(round(net_growth, 0))
        # Percentage change

        return total_follower, total_followers_gained, net_growth
    total_follower, total_followers_gained, net_growth = calculate_metrics(filtered_fanpage_df)
    total_follower_last_period, total_followers_gained_last_period, net_growth_last_period = calculate_metrics(filtered_fanpage_df_previous)

    net_follower_growth = (net_growth - net_growth_last_period)/net_growth if net_growth != 0 else 0
    net_follower_growth=round(net_follower_growth, 1)
    percent_growth = (total_follower / total_follower_last_period) if total_followers_gained != 0 else 0
    total_follower = int(round(total_follower, 0))

    change_symbol = "+" if total_followers_gained > 0 else ""
    change_color = "green" if total_followers_gained > 0 else "red"
    metric1, metric2, metric3, metric4, metric5, metric6 = st.columns(6)
    with metric1:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; border-radius: 15px; padding: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);">
                <div style="font-size:18px;">👨‍👨‍👧‍👧 Total Follower</div>
                <div style="font-size:30px; font-weight:bold;">{total_follower:,.0f}</div>
                <div style="font-size:16px; color:{change_color};">
                    {change_symbol}{total_followers_gained:,.0f} followers
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric2:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; border-radius: 15px; padding: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);">
                <div style="font-size:18px;">📢 Total Mentions</div>
                <div style="font-size:30px; font-weight:bold;">{current_mentions:,.0f}</div>
                <div style="font-size:16px; color:{change_color};">
                    {change_symbol}{change_in_mentions_percent:,.0f} mentions
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric3:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; border-radius: 15px; padding: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);">
                <div style="font-size:18px;">📣 Mentions Per Day</div>
                <div style="font-size:30px; font-weight:bold;">{current_mentions_per_day:,.1f}</div>
                <div style="font-size:16px; color:{change_color};">
                    {change_symbol}{change_in_mentions_per_day:,.0f} mentions
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric4:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; border-radius: 15px; padding: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); min-height: 120px;">
                <div style="font-size:18px;">🗣️ Total Authors</div>
                <div style="font-size:30px; font-weight:bold;">{curent_unique_authors:,.0f}</div>
                <div style="font-size:16px; color:{change_color};">
                    {change_symbol}{change_in_unique_authors:,.0f} authors
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric5:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; 
                        border-radius: 15px; padding: 10px; 
                        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); 
                        min-height: 120px;">
                <div style="font-size:18px;">👑 NSR</div>
                <!-- Multiply by 100 to display as a percentage, then add '%' at the end -->
                <div style="font-size:30px; font-weight:bold;">{current_nsr * 100:,.1f}%</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    with metric6:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #FFFFFF; border-radius: 15px; padding: 10px; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); min-height: 120px;">
                <div style="font-size:18px;">👍 Engagement </div>
                <div style="font-size:30px; font-weight:bold;">{total_engagement:,.0f}</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

with st.container(key = "container1", border = True):
    col1, col2 = st.columns([1,2])
    with col1:
        st.subheader("Fanpage Follower Growth")
        filtered_fanpage_df["Follower Gain"] = filtered_fanpage_df["Follower Gain"].apply(lambda x: int(x))
        filtered_fanpage_df["Follower Lost"] = filtered_fanpage_df["Follower Lost"].apply(lambda x: int(x))
        st.area_chart(
            filtered_fanpage_df,
            x="PublishedDate",
            y=["Follower Gain", "Follower Lost"],
            color=["#48C3B1", "#7062D4"], 
            x_label= "",
            y_label= "",
            
            )

        # Table headers
        col11, col12, col13 = st.columns([3, 1, 1])  # Reuse column widths for the rows
        with col11:
            st.markdown("**Metric**")
        with col12:
            st.markdown("**Totals**")
        with col13:
            st.markdown("**Change**")

        # Table rows
        col11, col12, col13 = st.columns([3, 1, 1])
        with col11:
            st.markdown("Followers")
        with col12:
            st.markdown(f"{total_follower:,}")  # Format with commas
        with col13:
            st.markdown(f"🔼 {percent_growth}%")  # Add up-arrow emoji for positive growth

        col11, col12, col13 = st.columns([3, 1, 1])
        with col11:
            st.markdown("Net Follower Growth")
        with col12:
            st.markdown(f"{net_growth:,}")  # Format with commas
        with col13:
            st.markdown(f"🔼 {net_follower_growth}%")


    with col2:
        st.subheader("Mention Trendline")

        def plot_mention_trendline_plotly(current_period, previous_period, date_col='PublishedDate', value_col='Id'):
            # Count occurrences of value_col by date
            current_counts = current_period.groupby(date_col)[value_col].count().reset_index()
            previous_counts = previous_period.groupby(date_col)[value_col].count().reset_index()

            # Align previous period dates to match current period's timeline
            period_length = len(current_counts)
            previous_counts_aligned = previous_counts.copy()
            previous_counts_aligned[date_col] = previous_counts[date_col] + pd.DateOffset(days=period_length)

            # Merge to ensure alignment
            aligned_data = pd.merge(
                current_counts,
                previous_counts_aligned,
                on=date_col,
                how='outer',
                suffixes=('_current', '_previous')
            ).fillna(0)  # Fill missing values with 0

            # Create the Plotly figure
            fig = go.Figure()

            # Plot current period
            fig.add_trace(
                go.Scatter(
                    x=aligned_data[date_col],
                    y=aligned_data[f"{value_col}_current"],
                    mode='lines',
                    name='Mention (Current Period)',
                    line=dict(color='brown', width=2)
                )
            )

            # Plot previous period
            fig.add_trace(
                go.Scatter(
                    x=aligned_data[date_col],
                    y=aligned_data[f"{value_col}_previous"],
                    mode='lines',
                    name='Mention (Previous Period)',
                    line=dict(color='gray', width=1.5, dash='dash')
                )
            )

            # Update layout for better readability
            fig.update_layout(

                xaxis_title='Day of the Month',
                yaxis_title='',
                xaxis=dict(
                    tickmode='array',
                    tickvals=aligned_data[date_col],
                    ticktext=[x.strftime('%d') for x in aligned_data[date_col]],
                    tickangle=45
                ),
                legend_title='Legend',
                margin=dict(l=40, r=40, t=40, b=40),
                hovermode='x unified',
                legend=dict(
                            x=0,
                            y=1))

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)

        plot_mention_trendline_plotly(current_period, previous_period)


    ###########################################break################################################################

with st.container(key = "container2", border = True):
    def top_sources_comparison(current_period, previous_period):
        # Ensure PublishedDate is datetime
        current_period['PublishedDate'] = pd.to_datetime(current_period['PublishedDate'])
        previous_period['PublishedDate'] = pd.to_datetime(previous_period['PublishedDate'])

        # Group by SiteName and count Id (mentions)
        current_counts = current_period.groupby('SiteName')['Id'].count().reset_index()
        current_counts.columns = ['SiteName', 'Current_Mentions']

        previous_counts = previous_period.groupby('SiteName')['Id'].count().reset_index()
        previous_counts.columns = ['SiteName', 'Previous_Mentions']

        # Merge to align current and previous mentions
        merged = pd.merge(current_counts, previous_counts, on='SiteName', how='outer').fillna(0)

        # Calculate percentage change
        merged['Change'] = merged['Current_Mentions'] - merged['Previous_Mentions']
        merged['Change_Percent'] = (merged['Change'] / merged['Previous_Mentions'].replace(0, 1)) * 100

        # Calculate percentage of total mentions
        total_mentions_current = merged['Current_Mentions'].sum()
        merged['Percent_of_Total'] = (merged['Current_Mentions'] / total_mentions_current) * 100

        # Sort by current mentions and take top 5
        top_5 = merged.sort_values(by='Current_Mentions', ascending=False).head(5)

        # Streamlit visualization
        st.write("### Top Sources")

        # Add header row
        header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1, 1, 3])
        with header_col1:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Source</p>", unsafe_allow_html=True)
        with header_col2:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Mentions</p>", unsafe_allow_html=True)
        with header_col3:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Change</p>", unsafe_allow_html=True)
        with header_col4:
            st.markdown("<p style='text-align: center; font-weight: bold;'>Trend</p>", unsafe_allow_html=True)

        # Display rows with horizontal separators
        for index, row in top_5.iterrows():
            trend_icon = "🔼" if row['Change'] > 0 else "🔽"
            site_name = row['SiteName']
            mentions = int(row['Current_Mentions'])  # Convert mentions to integer
            percent_of_total = f"{trend_icon} {row['Percent_of_Total']:.0f}%"
            change = int(row['Change'])  # Convert change to integer
            if mentions != 0:  # Avoid division by zero
                percent_change = change / mentions  # Calculate percent change
                percent_change = f"{trend_icon} {percent_change:.0%}"
            else:
                percent_change = "100%"  # Handle zero mentions gracefully
            # Get the URL for the site
            url = current_period[current_period['SiteName'] == site_name]['UrlTopic'].values[0]

            # Create columns for each row
            col1, col2, col3, col4 = st.columns([2, 1, 1, 3])
            with col1:
                st.markdown(f"<p style='vertical-align: middle;'><a href='{url}'><b>{site_name}</b></a></p>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<p style='text-align: center; vertical-align: middle;'>{mentions:,}</p>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<p style='text-align: center; vertical-align: middle;'>{percent_change}</p>", unsafe_allow_html=True)

            # Plot mention trendline as an area chart
            with col4:
                site_current_period = current_period[current_period['SiteName'] == site_name]
                site_previous_period = previous_period[previous_period['SiteName'] == site_name]

                # Group by date and count mentions
                current_trend = site_current_period.groupby('PublishedDate')['Id'].count().reset_index()
                previous_trend = site_previous_period.groupby('PublishedDate')['Id'].count().reset_index()

                # Plot area chart
                fig = px.area(
                    current_trend,
                    x='PublishedDate',
                    y='Id',
                    title=None
                )

                fig.update_layout(
                    xaxis_title='',
                    yaxis_title='',
                    margin=dict(l=1, r=1, t=1, b=1),
                    yaxis=dict(
                        showticklabels=False,
                        showgrid=False,        # Hides y-axis grid
                        zeroline=False    # This hides the y-axis labels
                        ),
                    hovermode='x unified',
                    height=40
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Add horizontal separator
            st.markdown("<hr style='border: 1px solid #ddd; margin: 0px 0;'>", unsafe_allow_html=True)

        return top_5


    column21, column22 = st.columns(2)
    with column21:
        top_sources_comparison(current_period, previous_period)

    with column22:
        st.markdown("### Top Channels")

        # Calculate value counts and percentages
        channel_counts = df["Channel"].value_counts(normalize=True).reset_index()
        channel_counts.columns = ["Channel", "Percentage"]
        channel_counts["Percentage"] = (channel_counts["Percentage"] * 100).round(1)

        # Take the top 5 channels and group the rest into "Others"
        top_5 = channel_counts.iloc[:5]
        others_sum = channel_counts.iloc[5:]["Percentage"].sum().round(1)
        others = pd.DataFrame([{"Channel": "Others", "Percentage": others_sum}])
        channel_counts = pd.concat([top_5, others], ignore_index=True)

        # -------------------------------------------------------------------
        # Sort by highest to lowest
        channel_counts = channel_counts.sort_values(by="Percentage", ascending=False).reset_index(drop=True)
        # -------------------------------------------------------------------

        # Define custom colors for each channel
        channel_colors = {
            "Facebook": "#2AB2FE",
            "Youtube": "#7218E9",
            "News": "#5ADDAB",
            "Tiktok": "#F4562F",
            "Forum": "#47D45A",
            "Others": "#AEAEAE",
            "Fanpage": "#FCC33E",
            "Social": "#7062D4",
            "Linkedin": "#0A66C2",
        }

        # Map colors based on channels (fallback color if key is missing)
        colors = [
            channel_colors.get(chan, "#ccc")  
            for chan in channel_counts["Channel"]
        ]

        # Create the horizontal bar chart
        fig = go.Figure()

        for i, row in channel_counts.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[row["Percentage"]],
                    y=[row["Channel"]],
                    orientation="h",
                    text=f"{row['Percentage']}%",  
                    textposition="inside",
                    marker=dict(
                        color=colors[i],
                        line=dict(width=0),
                    ),
                    width=0.7,
                    cliponaxis=False,
                    hoverinfo="none",
                    textfont=dict(
                        size=12,
                        family= sourcesans3bold,
                        color="white",
                    ),
                )
            )

        # Add labels at the center of each bar
        annotations = [
            dict(
                x=row["Percentage"] / 2,
                y=i,  
                text=row["Channel"],
                showarrow=False,
                font=dict(size=12, family= sourcesans3bold, color="white"),
                align="center",
            )
            for i, row in channel_counts.iterrows()
        ]
        max_percentage = channel_counts["Percentage"].max()
        # Update layout
        fig.update_layout(
            showlegend=False,  
            # Keep or adjust range based on your data; if some percentages exceed 50, use [0, 100].
            xaxis=dict(visible=False, range=[0, max_percentage + 4]),  
            yaxis=dict(
                visible=False,
                # Reversed so the highest percentage appears at the TOP
                autorange="reversed",
            ),
            margin=dict(t=50, b=1, l=30, r=10),
            annotations=annotations,
            barcornerradius=20
        )

        st.plotly_chart(fig)

#############################################break############################################################
demo = pd.read_excel(os.path.join(base_dir, 'Demo.xlsx'))
total_public_infor = demo['UserId'].nunique()
st.container()
with st.container(border = True):
    st.write("### Demographics")
    st.markdown(f"###### Total number of public audience infor: {total_public_infor:,}")
    demo1, demo2, demo3 = st.columns([1,1,2])
    with demo3:
        
        from collections import defaultdict
        def create_disk_data(df):
            # Count occurrences
            counts = df.groupby(['Vùng miền', 'HomeTown']).size().reset_index(name='Value')

            region_dict = defaultdict(list)

            # Aggregate data into region -> hometown -> value
            for _, row in counts.iterrows():
                region_dict[row['Vùng miền']].append({
                    "name": row['HomeTown'],
                    "value": row['Value']
                })

            # Format into the desired diskData structure
            diskData = [
                {
                    "name": region,
                    "children": hometowns
                }
                for region, hometowns in region_dict.items()
            ]
            return diskData


        # Get hierarchical data
        diskData = create_disk_data(demo)

        # Print result
        option = {
            "title": {"text": "Region and Hometown Distribution", "left": "center"},
            "tooltip": {
                "formatter": "{b}: {c} Occurrences"  # Simple Python-based formatting
            },
            "series": [
                {
                    "name": "Geographic",
                    "type": "treemap",
                    "visibleMin": 1,
                    "label": {"show": True, "formatter": "{b}"},
                    "itemStyle": {
                        "borderColor": "#fff",
                        "borderWidth": 1,
                        "gapWidth": 1
                    },
                    "colorMappingBy": "index",
                    "data": diskData
                }
            ]
        }

        st_echarts(option, height="500px")
    with demo1:
        test1, test2 = st.columns([1,1])
        with test1:
            # Define the symbols (pictorial shapes)
            symbols = [
                "path://M316.275,326.914c-1.545-14.858,0.59-29.846,3.006-40.885c9.592-10.133,17.803-25.214,26.268-49.096c12.748-4.676,25.696-15.472,31.703-42.072c2.77-12.253-2.041-23.582-10.35-31.74c6.022-20.216,29.73-113.881-41.494-132.874C301.724,0.641,280.995,1.824,245.466,5.377c-17.674,1.766-20.185,5.524-33.752,3.553c-15.559-2.263-28.182-6.985-31.977-8.883c-2.186-1.096-22.705,17.113-28.424,31.977c-18.11,47.087-13.244,79.381-8.641,103.806c-0.154,2.501-0.359,4.994-0.359,7.522l4.222,17.893c0.008,0.184-0.004,0.361,0.004,0.545c-9.18,8.23-14.715,20.133-11.791,33.073c6.01,26.612,18.967,37.403,31.718,42.077c8.459,23.864,16.662,38.938,26.248,49.072c2.418,11.04,4.554,26.036,3.01,40.904c-6.469,62.273-137.504,29.471-137.504,138.814c0,16.862,56.506,46.271,197.779,46.271c141.27,0,197.777-29.409,197.777-46.271C453.778,356.386,322.743,389.188,316.275,326.914z",
                "path://M308.753,328.173c-0.672-6.458-0.621-12.933-0.176-19.108c19.357,12.001,39.916,8.522,39.916,8.522s-10.356-21.908-7.879-51.321c4.937-9.611,9.351-20.347,13.338-31.965c9.4-6.165,17.934-17.175,22.475-37.282c1.773-7.846,0.383-15.297-3.008-21.789C411.447,51.846,331.886,0,283.921,0c-9.084,0-18.582,0.865-27.92,2.509C246.653,0.865,237.157,0,228.073,0c-47.967,0-127.529,51.848-89.496,175.238c-3.389,6.49-4.776,13.938-3.004,21.781c5.627,24.922,17.392,35.857,29.336,40.917c16.1,42.502-1.408,79.651-1.408,79.651s20.561,3.48,39.922-8.525c0.445,6.176,0.496,12.652-0.176,19.112c-6.422,61.85-143.68,56.46-143.68,137.87c0,16.748,56.123,45.957,196.434,45.957c140.309,0,196.432-29.209,196.432-45.957C452.433,384.633,315.177,390.023,308.753,328.173z'",
                
            ]

            # Max value for the y-axis
            # Convert int64 to native Python int
            bodyMax = int(demo['Gender'].count())
            male = int(demo[demo['Gender'] == 'Male']['Gender'].count())
            female = int(demo[demo['Gender'] == 'Female']['Gender'].count())

            male = round(male / bodyMax * 100, 1)
            female = round(female / bodyMax * 100, 1)
            
            # Label settings for the bars
            labelSetting = {
                "show": True,
                "position": "top",
                "fontSize": 15,
                "formatter": "{c}%",
                "fontFamily": sourcesans3
            }

            # ECharts option configuration
            option = {
                "tooltip": {},
                "xAxis": {
                    "data": ['Female'],
                    "axisTick": {"show": False},
                    "axisLine": {"show": False},
                    "axisLabel": {"show": True},
                    "fontFamily": sourcesans3
                },
                "yAxis": {
                    "max": 100,
                    "offset": 20,
                    "splitLine": {"show": False}
                },
                "grid": {
                    "top": 'center',
                    "height": 180
                },
                "series": [

                    {
                        "name": 'Female',
                        "type": 'pictorialBar',
                        "symbolClip": True,
                        "symbolBoundingData": 100,
                        "label": labelSetting,
                        "data": [
                            {"value": female, "symbol": symbols[1]},
                        ],
                        "z": 10,
                        "color": "#DA4B88",
                    },
                    {
                        "name": 'Full',
                        "type": 'pictorialBar',
                        "symbolBoundingData": 100,
                        "animationDuration": 0,
                        "itemStyle": {"color": '#ccc'},
                        "data": [
                            {"value": 1, "symbol": symbols[1]},
                        ]
                    }
                ]
            }

            # Display the chart in Streamlit
            st_echarts(options=option, height="600px")

        with test2:

            # ECharts option configuration
            option = {
                "tooltip": {},
                "xAxis": {
                    "data": ['Male'],
                    "axisTick": {"show": False},
                    "axisLine": {"show": False},
                    "axisLabel": {"show": True}
                },
                "yAxis": {
                    "max": 100,
                    "offset": 20,
                    "splitLine": {"show": False}
                },
                "grid": {
                    "top": 'center',
                    "height": 180
                },
                
                "series": [
                    {
                        "name": 'Male',
                        "type": 'pictorialBar',
                        "symbolClip": True,
                        "symbolBoundingData": 100,
                        "label": labelSetting,
                        "data": [
                            {"value": male, "symbol": symbols[0]},
                            
                        ],
                        "z": 11,
                        "color": "#6190D5",
                    },
                    {
                        "name": 'Full',
                        "type": 'pictorialBar',
                        "symbolBoundingData": 100,
                        "animationDuration": 0,
                        "itemStyle": {"color": '#ccc'},
                        "data": [
                            {"value": 1, "symbol": symbols[0]}
                        ]
                    }
                ]
            }

            st_echarts(options=option, height="600px")    
    with demo2:
        # Count occurrences dynamically
        counts = demo.groupby(["Gender", "Age Range"]).size().reset_index(name="Count")

        # Create the bar chart
        fig = go.Figure()

        # Separate data for each gender
        genders = counts["Gender"].unique()
        gender_colors = {
        "Male": "#6190D5",
        "Female": "#DA4B88"
    }
        for gender in genders:
            filtered_data = counts[counts["Gender"] == gender]
            fig.add_trace(
                go.Bar(
                    x=filtered_data["Age Range"],
                    y=filtered_data["Count"],
                    name=gender,
                    marker=dict(color=gender_colors[gender]),
                )
            )

        # Update layout for stacked bar chart
        fig.update_layout(
            title="Gender and Age Range Distribution",
            xaxis=dict(title="Age Range"),
            
            yaxis=dict(title =None, showgrid=False, showticklabels=False),
            barmode="stack",  # Set to "stack" for stacked bars
            template="plotly_white",
            legend=dict(
                x=0.75,  # Horizontal position (0 is left, 1 is right)
                y=0.95,  # Vertical position (0 is bottom, 1 is top)
                bgcolor="rgba(255,255,255,0.5)",  # Background color with transparency
            )
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)


#############################################break############################################################


def sentiment_percentage(df):
    """Calculate Positive, Negative, Neutral percentages."""
    positive = df[df['Sentiment'] == 'Positive']['Sentiment'].count()
    negative = df[df['Sentiment'] == 'Negative']['Sentiment'].count()
    neutral = df[df['Sentiment'] == 'Neutral']['Sentiment'].count()
    total = positive + negative + neutral
    
    # Avoid division by zero:
    if total == 0:
        return 0, 0, 0
    
    positive_pct = round((positive / total) * 100, 1)
    negative_pct = round((negative / total) * 100, 1)
    neutral_pct = round((neutral / total) * 100, 1)
    return positive_pct, negative_pct, neutral_pct

# Example usage:
# positive_percentage, negative_percentage, neutral_percentage = sentiment_percentage(current_period)

# ---------------------------------------------------------------------
# For demonstration only, let's define these manually:
positive_percentage = 30
negative_percentage = 50  # Not displayed
neutral_percentage  = 20  # Not displayed
# ---------------------------------------------------------------------

# We want one bottom slice for Positive (green), and one for the rest (gray).
plot_bgcolor = "#F1F4F8"

# Pie slices in the bottom half:
#  - The top half is an invisible filler (0.5).
#  - The bottom half is split between green (positive) and gray (everything else).
quadrant_colors = [
    plot_bgcolor,  # top (invisible)
    "green",       # Positive portion
    "lightgray",   # Remainder portion
]

# If you want labels, you can modify quadrant_text. Here we show only the green slice label:
quadrant_text = [
    "",  # invisible half
    "<b>Positive</b>",
    "",
]

# The pie values must sum to 1. 
# We reserve 0.5 for the invisible top, then split the bottom 0.5 by positive_percentage.
pie_values = [
    0.5,                                   
    0.5 * (positive_percentage / 100.0),   
    0.5 * (1 - positive_percentage / 100.0),
]

# Calculate the pointer angle.
#  - 0% => angle = π (all the way left)
#  - 100% => angle = 0 (all the way right)
hand_angle = np.pi * (1 - positive_percentage / 100.0)

# Adjust this as needed to match the donut's outer radius
hand_length = 0.3

fig = go.Figure(
    data=[
        go.Pie(
            values=pie_values,
            rotation=90,          # start from the top so the bottom half is visible
            hole=0.5,             # donut hole
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip",
        )
    ],
    layout=go.Layout(
        showlegend=False,
        autosize=True,
        margin=dict(b=0, t=10, l=0, r=0),
        shapes=[
            # Small circle at the center
            go.layout.Shape(
                type="circle",
                x0=0.495, x1=0.505,
                y0=0.495, y1=0.505,
                fillcolor="#333",
                line_color="#333",
            ),
            # The pointer (needle)
            go.layout.Shape(
                type="line",
                x0=0.5,
                y0=0.5,
                x1=0.5 + hand_length * np.cos(hand_angle),
                y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4),
            ),
        ],
        annotations=[
            # Example annotation below
            go.layout.Annotation(
                text=f"<b>Positive:</b> {positive_percentage}%",
                x=0.5, xanchor="center", xref="paper",
                y=0.23, yanchor="bottom", yref="paper",
                showarrow=False,
            )
        ],
    )
)

st.plotly_chart(fig, use_container_width=True)





import numpy as np
import plotly.graph_objects as go
import streamlit as st

def sentiment_percentage(df):
    positive = df[df['Sentiment'] == 'Positive']['Sentiment'].count()
    negative = df[df['Sentiment'] == 'Negative']['Sentiment'].count()   
    neutral = df[df['Sentiment'] == 'Neutral']['Sentiment'].count()
    total = positive + negative + neutral
    
    # Avoid division by zero:
    if total == 0:
        return 0, 0, 0
    
    positive_pct = round((positive / total) * 100, 1)
    negative_pct = round((negative / total) * 100, 1)
    neutral_pct = round((neutral / total) * 100, 1)
    return positive_pct, negative_pct, neutral_pct

# --------------------------------------------------------------------------
# Example: 
# Replace 'current_period' with your actual DataFrame 
# or keep it as is if you've already defined `current_period`.
# --------------------------------------------------------------------------
# positive_percentage, negative_percentage, neutral_percentage = sentiment_percentage(current_period)

# For demo purposes, let's hard-code some example percentages:
positive_percentage = 21
neutral_percentage = 64
negative_percentage = 15

# We’ll create a half-donut with 4 slices:
#   slice 0 = invisible top half  (0.5)
#   slice 1 = green for Positive  (0.5 * positive_percentage/100)
#   slice 2 = gray for Neutral    (0.5 * neutral_percentage/100)
#   slice 3 = red for Negative    (0.5 * negative_percentage/100)
#
# The sum of slices 1..3 should be 0.5 if pos+neu+neg=100.
# The total Pie = 1.0 (0.5 for the top + 0.5 for the bottom).
plot_bgcolor = "#F1F4F8"
quadrant_colors = [
    plot_bgcolor,   # top (invisible filler)
    "green",        # Positive
    "lightgray",    # Neutral
    "red"           # Negative
]

# Show labels on each bottom slice (optional)
quadrant_text = [
    "",  # no label for the invisible half
    f"<b>Positive: {positive_percentage}%</b>",
    f"<b>Neutral: {neutral_percentage}%</b>",
    f"<b>Negative: {negative_percentage}%</b>",
]

pie_values = [
    0.5,  # invisible top half
    0.5 * (positive_percentage / 100.0),
    0.5 * (neutral_percentage / 100.0),
    0.5 * (negative_percentage / 100.0),
]

# --------------------------------------------------------------------------
# Need an "angle" for the pointer. You can decide which metric it reflects.
# Example: We'll point at "positive_percentage" for demonstration. 
# If pos=0 => angle=π (far left); if pos=100 => angle=0 (far right).
# --------------------------------------------------------------------------
pointer_value = positive_percentage
hand_angle = np.pi * (1 - pointer_value / 100.0)

# Adjust needle length for your layout
hand_length = 0.3  

fig = go.Figure(
    data=[
        go.Pie(
            values=pie_values,
            rotation=90,          # start from top=90° so the bottom half is visible
            hole=0.5,             # donut hole
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",      # show the text from quadrant_text
            hoverinfo="skip",     # disable hover if you want
        )
    ],
    layout=go.Layout(
        showlegend=False,
        # Let Streamlit handle responsiveness:
        autosize=True,
        margin=dict(b=0, t=10, l=0, r=0),
        shapes=[
            # The small center circle
            go.layout.Shape(
                type="circle",
                x0=0.495, x1=0.505,
                y0=0.495, y1=0.505,
                fillcolor="#333",
                line_color="#333",
            ),
            # The pointer line
            go.layout.Shape(
                type="line",
                x0=0.5,
                y0=0.5,
                x1=0.5 + hand_length * np.cos(hand_angle),
                y1=0.5 + hand_length * np.sin(hand_angle),
                line=dict(color="#333", width=4),
            ),
        ],
        annotations=[
            # Example annotation below the gauge
            go.layout.Annotation(
                text=f"<b>Pointer value:</b> {pointer_value}%",
                x=0.5, xanchor="center", xref="paper",
                y=0.22, yanchor="bottom", yref="paper",
                showarrow=False,
            )
        ],
    )
)

st.plotly_chart(fig, use_container_width=True)
