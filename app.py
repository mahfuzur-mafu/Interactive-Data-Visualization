import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Air Pollution Impact Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df1 = pd.read_csv('absolute-number-of-deaths-from-ambient-particulate-air-pollution.csv')
    df2 = pd.read_csv('death-rate-from-air-pollution-per-100000.csv')
    df3 = pd.read_csv('death-rates-from-air-pollution.csv')
    df4 = pd.read_csv('share-deaths-indoor-pollution.csv')
    df5 = pd.read_csv('outdoor-pollution-death-rate.csv')
    return df1, df2, df3, df4, df5

try:
    df1, df2, df3, df4, df5 = load_data()
except FileNotFoundError:
    st.error("Data files not found. Please ensure all CSV files are in the same directory as this script.")
    st.stop()

# Data preprocessing
trdf1 = df1.rename(columns={
    'Deaths - Cause: All causes - Risk: Outdoor air pollution - OWID - Sex: Both - Age: All Ages (Number)': 'Total Deaths for Outdoor Air Pollution',
    'Deaths - Cause: All causes - Risk: Household air pollution from solid fuels - Sex: Both - Age: All Ages (Number)': 'Total Deaths for Household Air Pollution from Solid Fuels',
    'Deaths - Cause: All causes - Risk: Air pollution - Sex: Both - Age: All Ages (Number)': 'Total Deaths for Air Pollution'
})
apoh = trdf1[['Entity', 'Year', 'Total Deaths for Air Pollution', 'Total Deaths for Outdoor Air Pollution', 'Total Deaths for Household Air Pollution from Solid Fuels']]

drapph = df2.rename(columns={'Deaths - Cause: All causes - Risk: Air pollution - Sex: Both - Age: Age-standardized (Rate)': 'Death Rate from Air Pollution Per 100000'})
drapmp = df3.rename(columns={
    'Deaths - Cause: All causes - Risk: Ambient particulate matter pollution - Sex: Both - Age: Age-standardized (Rate)': 'Deaths Rate for Ambient Particulate Matter Pollution',
    'Deaths - Cause: All causes - Risk: Ambient ozone pollution - Sex: Both - Age: Age-standardized (Rate)': 'Deaths Rate for Ambient Ozone Pollution'
}).drop(['Deaths - Cause: All causes - Risk: Household air pollution from solid fuels - Sex: Both - Age: Age-standardized (Rate)', 
        'Deaths - Cause: All causes - Risk: Air pollution - Sex: Both - Age: Age-standardized (Rate)', 'Code'], axis=1)

dhapsfp = df4.rename(columns={'Deaths - Cause: All causes - Risk: Household air pollution from solid fuels - Sex: Both - Age: Age-standardized (Percent)': 'Deaths for Household Air Pollution from Solid Fuels (Percent)'})
dfoapp = df5.rename(columns={'Deaths - Cause: All causes - Risk: Outdoor air pollution - OWID - Sex: Both - Age: Age-standardized (Rate)': 'Death for Outdoor Air Pollution - (Per 100K)'})

# Merge data
merap = apoh.merge(drapph, on=['Entity', 'Year'], how='left').merge(drapmp, on=['Entity', 'Year'], how='left').merge(
    dhapsfp[['Entity', 'Year', 'Deaths for Household Air Pollution from Solid Fuels (Percent)']], on=['Entity', 'Year'], how='left').merge(
    dfoapp[['Entity', 'Year', 'Death for Outdoor Air Pollution - (Per 100K)']], on=['Entity', 'Year'], how='left').drop('Code', axis=1)

# Exclude non-country entities for country-level charts
exclude_entities = ['African Region (WHO)', 'East Asia & Pacific (WB)', 'Eastern Mediterranean Region (WHO)', 'Europe & Central Asia (WB)', 
                    'European Region (WHO)', 'G20', 'Latin America & Caribbean (WB)', 'Middle East & North Africa (WB)', 'North America (WB)', 
                    'OECD Countries', 'Region of the Americas (WHO)', 'South Asia (WB)', 'South-East Asia Region (WHO)', 'Sub-Saharan Africa (WB)', 
                    'Western Pacific Region (WHO)', 'World Bank High Income', 'World Bank Lower Middle Income', 'World Bank Upper Middle Income', 
                    'World Bank Low Income']

merap_countries = merap[~merap['Entity'].isin(exclude_entities)]

# Define regions for the "Total Deaths by Region" chart
regions = ['African Region (WHO)', 'East Asia & Pacific (WB)', 'Eastern Mediterranean Region (WHO)', 'Europe & Central Asia (WB)', 
           'European Region (WHO)', 'Latin America & Caribbean (WB)', 'Middle East & North Africa (WB)', 'North America (WB)', 
           'Region of the Americas (WHO)', 'South Asia (WB)', 'South-East Asia Region (WHO)', 'Sub-Saharan Africa (WB)', 
           'Western Pacific Region (WHO)']
merap_regions = merap[merap['Entity'].isin(regions)]

# Title
st.title("Air Pollution Impact Dashboard")

# First Layer: Static KPI Boxes (Cumulative Global Deaths 1990-2019)
df_world = merap[(merap['Entity'] == 'World') & (merap['Year'].between(1990, 2019))]
total_deaths_sum = df_world['Total Deaths for Air Pollution'].sum()
outdoor_deaths_sum = df_world['Total Deaths for Outdoor Air Pollution'].sum()
indoor_deaths_sum = df_world['Total Deaths for Household Air Pollution from Solid Fuels'].sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Total Deaths 1990-2019", 
              value=f"{int(total_deaths_sum):,}" if pd.notna(total_deaths_sum) else "No data")

with col2:
    st.metric(label="Total Outdoor 1990-2019", 
              value=f"{int(outdoor_deaths_sum):,}" if pd.notna(outdoor_deaths_sum) else "No data")

with col3:
    st.metric(label="Total Indoor 1990-2019", 
              value=f"{int(indoor_deaths_sum):,}" if pd.notna(indoor_deaths_sum) else "No data")

# Second Layer: Interactive Filters and Visualizations
col_left, col_right = st.columns([1, 3])

# Left Column: Filters
with col_left:
    st.subheader("Filters")
    
    country = st.selectbox("Country", ['World'] + sorted(merap_countries['Entity'].unique().tolist()), 
                           help="Select 'World' or a specific country.")
    
    year_kpi = st.selectbox("Year (for KPI Boxes)", sorted(merap['Year'].unique().tolist(), reverse=True), 
                            help="Choose a specific year for KPI boxes.", key="year_kpi", index=0)
    
    st.subheader("Bar Chart Race Visualization (Flourish)")
    flourish_html = '''
<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/22721128"><script src="https://public.flourish.studio/resources/embed.js"></script><noscript><img src="https://public.flourish.studio/visualisation/22721128/thumbnail" width="100%" alt="bar-chart-race visualization" /></noscript></div>
    '''

    
    components.html(flourish_html, height=700)  # Use components.html instead of st.components.html

    st.subheader("Bar Chart Race Visualization (Flourish)")
    

    flourish_html = '''
<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/22722339"><script src="https://public.flourish.studio/resources/embed.js"></script><noscript><img src="https://public.flourish.studio/visualisation/22722339/thumbnail" width="100%" alt="bar-chart-race visualization" /></noscript></div>
    '''
    
    components.html(flourish_html, height=700)  
    
    st.subheader("Bar Chart Race Visualization (Flourish)")
    

    flourish_html = '''
<div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/22721916"><script src="https://public.flourish.studio/resources/embed.js"></script><noscript><img src="https://public.flourish.studio/visualisation/22721916/thumbnail" width="100%" alt="bar-chart-race visualization" /></noscript></div>
    '''
    
    components.html(flourish_html, height=700)
    
# Use components.html instead of st.components.html
# Right Column: Visualizations
with col_right:
    # New Feature: KPI Boxes for Selected Country and Year
    st.subheader(f"Air Pollution Deaths in {country} ({year_kpi})")
    df_selected = merap[(merap['Entity'] == country) & (merap['Year'] == year_kpi)]

    col4, col5, col6 = st.columns(3)

    with col4:
        total_deaths = df_selected['Total Deaths for Air Pollution'].iloc[0] if not df_selected.empty else "No data"
        st.metric(label="Total Deaths", 
                  value=f"{int(total_deaths):,}" if isinstance(total_deaths, (int, float)) and pd.notna(total_deaths) else total_deaths)

    with col5:
        outdoor_deaths = df_selected['Total Deaths for Outdoor Air Pollution'].iloc[0] if not df_selected.empty else "No data"
        st.metric(label="Outdoor Deaths", 
                  value=f"{int(outdoor_deaths):,}" if isinstance(outdoor_deaths, (int, float)) and pd.notna(outdoor_deaths) else outdoor_deaths)

    with col6:
        indoor_deaths = df_selected['Total Deaths for Household Air Pollution from Solid Fuels'].iloc[0] if not df_selected.empty else "No data"
        st.metric(label="Indoor Deaths", 
                  value=f"{int(indoor_deaths):,}" if isinstance(indoor_deaths, (int, float)) and pd.notna(indoor_deaths) else indoor_deaths)

    # Chart 1: Total Deaths by Country
    st.subheader("Total Deaths by Country")
    col_rank1, col_year1 = st.columns(2)
    with col_rank1:
        rank_type1 = st.selectbox("Top 20 and Bottom 20", ["Top 20", "Bottom 20"], key="rank1")
    with col_year1:
        year_chart1 = st.selectbox("By the year and All", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                                   key="year1", index=1)

    if year_chart1 == 'All':
        df = merap if country == 'World' else merap_countries[merap_countries['Entity'] == country]
        fig = px.line(df, x='Year', y='Total Deaths for Air Pollution', 
                      title=f"Total Deaths ({'World' if country == 'World' else country}, 1990-2019)")
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart1)].dropna(subset=['Total Deaths for Air Pollution'])
        df = df.sort_values('Total Deaths for Air Pollution', ascending=(rank_type1 == "Bottom 20"))
        df_filtered = df.head(10) if rank_type1 == "Top 20" else df.tail(10)
        fig = px.bar(df_filtered, x='Total Deaths for Air Pollution', y='Entity', 
                     title=f"{rank_type1} Countries by Total Deaths ({year_chart1})")
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 2: Total Deaths Outdoor by Country
    st.subheader("Total Deaths Outdoor by Country")
    col_rank2, col_year2 = st.columns(2)
    with col_rank2:
        rank_type2 = st.selectbox("Top 20 and Bottom 20", ["Top 20", "Bottom 20"], key="rank2")
    with col_year2:
        year_chart2 = st.selectbox("By the year and All", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                                   key="year2", index=1)

    if year_chart2 == 'All':
        df = merap if country == 'World' else merap_countries[merap_countries['Entity'] == country]
        fig = px.line(df, x='Year', y='Total Deaths for Outdoor Air Pollution', 
                      title=f"Outdoor Deaths ({'World' if country == 'World' else country}, 1990-2019)")
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart2)].dropna(subset=['Total Deaths for Outdoor Air Pollution'])
        df = df.sort_values('Total Deaths for Outdoor Air Pollution', ascending=(rank_type2 == "Bottom 20"))
        df_filtered = df.head(20) if rank_type2 == "Top 20" else df.tail(20)
        fig = px.bar(df_filtered, x='Total Deaths for Outdoor Air Pollution', y='Entity', 
                     title=f"{rank_type2} Countries by Outdoor Deaths ({year_chart2})")
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 3: Total Deaths Indoor by Country
    st.subheader("Total Deaths Indoor by Country")
    col_rank3, col_year3 = st.columns(2)
    with col_rank3:
        rank_type3 = st.selectbox("Top 20 and Bottom 20", ["Top 20", "Bottom 20"], key="rank3")
    with col_year3:
        year_chart3 = st.selectbox("By the year and All", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                                   key="year3", index=1)

    if year_chart3 == 'All':
        df = merap if country == 'World' else merap_countries[merap_countries['Entity'] == country]
        fig = px.line(df, x='Year', y='Total Deaths for Household Air Pollution from Solid Fuels', 
                      title=f"Indoor Deaths ({'World' if country == 'World' else country}, 1990-2019)")
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart3)].dropna(subset=['Total Deaths for Household Air Pollution from Solid Fuels'])
        df = df.sort_values('Total Deaths for Household Air Pollution from Solid Fuels', ascending=(rank_type3 == "Bottom 20"))
        df_filtered = df.head(20) if rank_type3 == "Top 20" else df.tail(20)
        fig = px.bar(df_filtered, x='Total Deaths for Household Air Pollution from Solid Fuels', y='Entity', 
                     title=f"{rank_type3} Countries by Indoor Deaths ({year_chart3})")
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 4: Total Deaths per 100K
    st.subheader("Total Deaths per 100K")
    col_rank4, col_year4 = st.columns(2)
    with col_rank4:
        rank_type4 = st.selectbox("Top 20 and Bottom 20", ["Top 20", "Bottom 20"], key="rank4")
    with col_year4:
        year_chart4 = st.selectbox("By the year and All", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                                   key="year4", index=1)

    if year_chart4 == 'All':
        df = merap if country == 'World' else merap_countries[merap_countries['Entity'] == country]
        fig = px.line(df, x='Year', y='Death Rate from Air Pollution Per 100000', 
                      title=f"Death Rate per 100K ({'World' if country == 'World' else country}, 1990-2019)")
        fig.update_layout(hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart4)].dropna(subset=['Death Rate from Air Pollution Per 100000'])
        df = df.sort_values('Death Rate from Air Pollution Per 100000', ascending=(rank_type4 == "Bottom 20"))
        df_filtered = df.head(20) if rank_type4 == "Top 20" else df.tail(20)
        fig = px.bar(df_filtered, x='Death Rate from Air Pollution Per 100000', y='Entity', 
                     title=f"{rank_type4} Countries by Death Rate per 100K ({year_chart4})")
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True)

    # Chart 5: Total Deaths by Region
    st.subheader("Total Deaths by Region")
    metric_region = st.selectbox("Indoor and Outdoor and All", ["All", "Indoor", "Outdoor"], key="metric_region")
    
    metric_mapping = {
        "All": "Total Deaths for Air Pollution",
        "Indoor": "Total Deaths for Household Air Pollution from Solid Fuels",
        "Outdoor": "Total Deaths for Outdoor Air Pollution"
    }
    year = st.selectbox("Year (for Charts)", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                        help="Choose a specific year or 'All' for trends (used in charts).", index=1)
    selected_metric = metric_mapping[metric_region]

    # Use the global year filter for simplicity (can add a separate year filter if needed)
    if year == 'All':
        df = merap_regions
        fig = px.line(df, x='Year', y=selected_metric, color='Entity', 
                      title=f"Deaths by Region (1990-2019, {metric_region})")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_regions[merap_regions['Year'] == int(year)].dropna(subset=[selected_metric])
        df = df.sort_values(selected_metric, ascending=False)
        fig = px.bar(df, x=selected_metric, y='Entity', 
                     title=f"Deaths by Region ({year}, {metric_region})")
        fig.update_layout(hovermode="y unified")
        st.plotly_chart(fig, use_container_width=True)
        
    # New Chart: Projected Upcoming Deaths
    projection_data = {
    'Year': [2025, 2026, 2027, 2028, 2029],
    'Total Deaths': [8000000, 7900000, 7800000, 7600000, 7400000],
    'Indoor Deaths': [3000000, 2900000, 2800000, 2700000, 2600000],
    'Outdoor Deaths': [4000000, 3900000, 3800000, 3700000, 3600000]
}
    df_projections = pd.DataFrame(projection_data)
    st.subheader("Projected Upcoming Deaths")
    projection_period = st.selectbox("Filter by", ["2 years", "5 years"], key="projection_period")
    
    # Filter data based on projection period
    max_year = 2026 if projection_period == "2 years" else 2029
    df_plot = df_projections[df_projections['Year'] <= max_year]
    
    # Create line chart with custom styles
    fig = go.Figure()
    
    # Total Deaths (solid line with circles)
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Total Deaths'],
        mode='lines+markers',
        name='Total',
        line=dict(dash='solid', color='gray'),
        marker=dict(symbol='circle', size=8)
    ))
    
    # Indoor Deaths (dashed line with circles)
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Indoor Deaths'],
        mode='lines+markers',
        name='Indoor',
        line=dict(dash='dash', color='lightgray'),
        marker=dict(symbol='circle', size=8)
    ))
    
    # Outdoor Deaths (dotted line with circles)
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Outdoor Deaths'],
        mode='lines+markers',
        name='Outdoor',
        line=dict(dash='dot', color='darkgray'),
        marker=dict(symbol='circle', size=8)
    ))
    
    fig.update_layout(
        title="Projected Upcoming Deaths",
        xaxis_title="Year",
        yaxis_title="Number of Deaths",
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(tickmode='linear', tick0=2025, dtick=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    
    # Categorize death rates into Low, Medium, Risk, Very Risk
def categorize_death_rate(df):
    df = df.copy()
    df['Death Rate Category'] = pd.qcut(
        df['Death Rate from Air Pollution Per 100000'],
        q=4,
        labels=['Low', 'Medium', 'Risk', 'Very Risk'],
        duplicates='raise'
    )
    return df

# Apply categorization to each year
merap_countries = merap_countries.groupby('Year').apply(categorize_death_rate).reset_index(drop=True)

# Chart 7: Death Rate Categories by Country (Choropleth Map)
st.subheader("Death Rate Categories by Country")

# Create choropleth map with slider
fig = px.choropleth(
    merap_countries,
    locations='Entity',
    locationmode='country names',
    color='Death Rate Category',
    hover_name='Entity',
    hover_data={'Death Rate from Air Pollution Per 100000': ':.2f', 'Year': False, 'Death Rate Category': False},
    animation_frame='Year',
    category_orders={'Death Rate Category': ['Low', 'Medium', 'Risk', 'Very Risk']},
    color_discrete_map={
        'Low': '#00FF00',      # Green
        'Medium': '#FFFF00',   # Yellow
        'Risk': '#FFA500',     # Orange
        'Very Risk': '#FF0000' # Red
    },
    title="Death Rate Categories by Country (Per 100,000)"
)

fig.update_layout(
    geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
    margin={"r":0, "t":50, "l":0, "b":0},
    height=800
)

# Update slider appearance
fig.update_layout(
    sliders=[{
        'active': len(merap_countries['Year'].unique()) - 1,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Year: ',
            'visible': True,
            'xanchor': 'right'
        },
        'pad': {'b': 10, 't': 10},
        'len': 0.9,
        'x': 0.1,
        'y': 0
    }]
)

st.plotly_chart(fig, use_container_width=True)

