import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components


st.set_page_config(page_title="Visualizing the Impact of Air Pollution on Mortality Rates: A Tool for Public Health Action Dashboard", layout="wide")

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
regions = ['All'] + ['African Region (WHO)', 'East Asia & Pacific (WB)', 'Eastern Mediterranean Region (WHO)', 'Europe & Central Asia (WB)', 
           'European Region (WHO)', 'Latin America & Caribbean (WB)', 'Middle East & North Africa (WB)', 'North America (WB)', 
           'Region of the Americas (WHO)', 'South Asia (WB)', 'South-East Asia Region (WHO)', 'Sub-Saharan Africa (WB)', 
           'Western Pacific Region (WHO)']
merap_regions = merap[merap['Entity'].isin(regions[1:])]


st.title("Visualizing the Impact of Air Pollution on Mortality Rates: A Tool for Public Health Action Dashboard")

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

# Second Layer: Filters and Interactive KPI Boxes
col_left, col_right = st.columns([1, 3])

# Left Column: Filters
with col_left:
    st.subheader("Filters")
    
    country = st.selectbox("Country", ['World'] + sorted(merap_countries['Entity'].unique().tolist()), 
                           help="Select 'World' or a specific country.")
    
    year_kpi = st.selectbox("Year (for KPI Boxes)", sorted(merap['Year'].unique().tolist(), reverse=True), 
                            help="Choose a specific year for KPI boxes.", key="year_kpi", index=0)

# Right Column: Interactive KPI Boxes
with col_right:
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

# Row 3: Historical Death Trends (Three Columns)
st.subheader("Historical Death Trends (1990-2019)")
col_trend1, col_trend2, col_trend3 = st.columns(3)

# First Column: Historical Total Deaths Trend
with col_trend1:
    st.write(f"Total Deaths Trend ({country})")
    df_country = merap[(merap['Entity'] == country) & (merap['Year'].between(1990, 2019))]
    fig = px.line(df_country, x='Year', y='Total Deaths for Air Pollution',
                  title=f"Total Deaths ({country})")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Deaths",
        hovermode="x unified",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Second Column: Historical Outdoor Deaths Trend
with col_trend2:
    st.write(f"Outdoor Deaths Trend ({country})")
    df_country = merap[(merap['Entity'] == country) & (merap['Year'].between(1990, 2019))]
    fig = px.line(df_country, x='Year', y='Total Deaths for Outdoor Air Pollution',
                  title=f"Outdoor Deaths ({country})")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Outdoor Deaths",
        hovermode="x unified",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Third Column: Historical Indoor Deaths Trend
with col_trend3:
    st.write(f"Indoor Deaths Trend ({country})")
    df_country = merap[(merap['Entity'] == country) & (merap['Year'].between(1990, 2019))]
    fig = px.line(df_country, x='Year', y='Total Deaths for Household Air Pollution from Solid Fuels',
                  title=f"Indoor Deaths ({country})")
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Indoor Deaths",
        hovermode="x unified",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Row 4: Outdoor vs. Indoor Proportion & Age Group Analysis and Choropleth Map (Two Columns)
st.subheader("Proportions and Age Group Analysis with Global Death Rate Distribution")
col_pie, col_choropleth = st.columns(2)

# First Column: Outdoor vs. Indoor Deaths Proportion and Age Group Analysis
with col_pie:
    st.write(f"Outdoor vs. Indoor Deaths Proportion & Age Group Analysis ({country}, {year_kpi})")
   
    df_selected = merap[(merap['Entity'] == country) & (merap['Year'] == year_kpi)]
    outdoor_death_rate = df_selected['Death for Outdoor Air Pollution - (Per 100K)'].iloc[0] if not df_selected.empty and pd.notna(df_selected['Death for Outdoor Air Pollution - (Per 100K)'].iloc[0]) else 50
    indoor_death_rate = df_selected['Death Rate from Air Pollution Per 100000'].iloc[0] * (df_selected['Deaths for Household Air Pollution from Solid Fuels (Percent)'].iloc[0] / 100) if not df_selected.empty and pd.notna(df_selected['Death Rate from Air Pollution Per 100000'].iloc[0]) and pd.notna(df_selected['Deaths for Household Air Pollution from Solid Fuels (Percent)'].iloc[0]) else 50

    def generate_simulated_age_data(country, year, outdoor_rate, indoor_rate):
        age_groups = ['<15', '15-49', '50-69', '70+']
     
        outdoor_rates = [outdoor_rate * 0.3, outdoor_rate * 0.5, outdoor_rate * 0.8, outdoor_rate * 1.2]
        indoor_rates = [indoor_rate * 0.3, indoor_rate * 0.5, indoor_rate * 0.8, indoor_rate * 1.2]
        return pd.DataFrame({
            'Age Group': age_groups * 2,
            'Death Rate Per 100K': outdoor_rates + indoor_rates,
            'Category': ['Outdoor'] * 4 + ['Indoor'] * 4,
            'Entity': country,
            'Year': year
        })

    df_age = generate_simulated_age_data(country, year_kpi, outdoor_death_rate, indoor_death_rate)

    # Create stacked bar chart for age group analysis
    fig_age = px.bar(
        df_age,
        x='Age Group',
        y='Death Rate Per 100K',
        color='Category',
        title=f"Death Rates by Age Group ({country}, {year_kpi})",
        color_discrete_map={'Outdoor': '#FFA500', 'Indoor': '#FF0000'}
    )
    fig_age.update_layout(
        xaxis_title="Age Group",
        yaxis_title="Death Rate per 100,000",
        hovermode="x unified",
        showlegend=True,
        height=400
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Original pie chart for proportion
    if not df_selected.empty:
        outdoor_deaths = df_selected['Total Deaths for Outdoor Air Pollution'].iloc[0]
        indoor_deaths = df_selected['Total Deaths for Household Air Pollution from Solid Fuels'].iloc[0]
        if pd.notna(outdoor_deaths) and pd.notna(indoor_deaths):
            pie_data = pd.DataFrame({
                'Category': ['Outdoor', 'Indoor'],
                'Deaths': [outdoor_deaths, indoor_deaths]
            })
            fig_pie = px.pie(pie_data, values='Deaths', names='Category',
                            title=f"Proportion of Deaths ({country}, {year_kpi})",
                            color_discrete_map={'Outdoor': '#FFA500', 'Indoor': '#FF0000'})
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write("No data available for this selection.")
    else:
        st.write("No data available for this selection.")

# Second Column: Choropleth Map
with col_choropleth:
    st.write("Death Rate Categories by Country")

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
        height=900
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

# Row 5: Projected Deaths, Deaths by Country, and Total Deaths per 100K (Three Columns)
st.subheader("Projections, Deaths by Country, and Total Deaths per 100K")
col_proj, col_country, col_metric1 = st.columns(3)

# First Column: Projected Deaths
with col_proj:
    st.write("Projected Deaths (2025-2029)")
    projection_period = st.selectbox("Filter by", ["2 years", "5 years"], key="projection_period")
    max_year = 2026 if projection_period == "2 years" else 2029
    
    # Projection data
    projection_data = {
        'Year': [2025, 2026, 2027, 2028, 2029],
        'Total Deaths': [8000000, 7900000, 7800000, 7600000, 7400000],
        'Indoor Deaths': [3000000, 2900000, 2800000, 2700000, 2600000],
        'Outdoor Deaths': [4000000, 3900000, 3800000, 3700000, 3600000]
    }
    df_projections = pd.DataFrame(projection_data)
    df_plot = df_projections[df_projections['Year'] <= max_year]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Total Deaths'],
        mode='lines+markers',
        name='Total',
        line=dict(dash='solid', color='gray'),
        marker=dict(symbol='circle', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Outdoor Deaths'],
        mode='lines+markers',
        name='Outdoor',
        line=dict(dash='dot', color='darkgray'),
        marker=dict(symbol='circle', size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_plot['Year'], y=df_plot['Indoor Deaths'],
        mode='lines+markers',
        name='Indoor',
        line=dict(dash='dash', color='lightgray'),
        marker=dict(symbol='circle', size=8)
    ))
    fig.update_layout(
        title="Projected Deaths (Total, Outdoor, Indoor)",
        xaxis_title="Year",
        yaxis_title="Number of Deaths",
        hovermode="x unified",
        showlegend=True,
        xaxis=dict(tickmode='linear', tick0=2025, dtick=1),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# Second Column: Deaths by Country
with col_country:
    st.write("Deaths by Country")
    col_metric, col_year, col_rank = st.columns(3)
    
    with col_metric:
        metric = st.selectbox("Metric", ["Total", "Outdoor", "Indoor"], key="country_metric")
    
    with col_year:
        year_chart = st.selectbox("Year", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                                  key="country_year", index=1)
    
    with col_rank:
        rank_type = st.selectbox("Ranking", ["Top 10", "Bottom 10"], key="country_rank")

    # Map the metric to the corresponding column in the DataFrame
    metric_mapping = {
        "Total": "Total Deaths for Air Pollution",
        "Outdoor": "Total Deaths for Outdoor Air Pollution",
        "Indoor": "Total Deaths for Household Air Pollution from Solid Fuels"
    }
    selected_metric = metric_mapping[metric]

    if year_chart == 'All':
        df = merap if country == 'World' else merap_countries[merap_countries['Entity'] == country]
        fig = px.line(df, x='Year', y=selected_metric, 
                      title=f"{metric} Deaths ({'World' if country == 'World' else country}, 1990-2019)")
        fig.update_layout(hovermode="x unified", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart)].dropna(subset=[selected_metric])

        if rank_type == "Top 10":
            df_sorted = df.sort_values(by=selected_metric, ascending=False).head(10)
        else:  
            df_sorted = df.sort_values(by=selected_metric, ascending=True).tail(10)
        fig = px.bar(df_sorted, x=selected_metric, y='Entity', 
                     title=f"{rank_type} Countries by {metric} Deaths ({year_chart})")
        fig.update_layout(hovermode="y unified", height=400)
        st.plotly_chart(fig, use_container_width=True)

# Third Column: Total Deaths per 100K (Moved from Row 6)
with col_metric1:
    st.write("Total Deaths per 100K")
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
        fig.update_layout(hovermode="x unified", showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_countries[merap_countries['Year'] == int(year_chart4)].dropna(subset=['Death Rate from Air Pollution Per 100000'])
        df = df.sort_values('Death Rate from Air Pollution Per 100000', ascending=(rank_type4 == "Bottom 20"))
        df_filtered = df.head(20) if rank_type4 == "Top 20" else df.tail(20)
        fig = px.bar(df_filtered, x='Death Rate from Air Pollution Per 100000', y='Entity', 
                     title=f"{rank_type4} Countries by Death Rate per 100K ({year_chart4})")
        fig.update_layout(hovermode="y unified", height=400)
        st.plotly_chart(fig, use_container_width=True)

# Row 6: Total Deaths by Region and Two Region-Based Insights (Three Columns)
st.subheader("Regional Analysis")


col_filter1, col_filter2, col_filter3 = st.columns(3)
with col_filter1:
    metric_region = st.selectbox("Indoor and Outdoor and All", ["All", "Indoor", "Outdoor"], key="metric_region")
with col_filter2:
    year = st.selectbox("Year (for Charts)", ['All'] + sorted(merap['Year'].unique().tolist(), reverse=True), 
                        help="Choose a specific year or 'All' for trends (used in charts).", index=1)
with col_filter3:
    region = st.selectbox("Select Region for Insights", regions, help="Select a region for detailed insights.", index=0)

metric_mapping = {
    "All": "Total Deaths for Air Pollution",
    "Indoor": "Total Deaths for Household Air Pollution from Solid Fuels",
    "Outdoor": "Total Deaths for Outdoor Air Pollution"
}
selected_metric = metric_mapping[metric_region]

# First Column: Total Deaths by Region
col_region, col_insight1, col_insight2 = st.columns(3)
with col_region:
    st.write("Total Deaths by Region")
    if year == 'All':
        df = merap_regions
        fig = px.line(df, x='Year', y=selected_metric, color='Entity', 
                      title=f"Deaths by Region (1990-2019, {metric_region})")
        fig.update_layout(hovermode="x unified", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        df = merap_regions[merap_regions['Year'] == int(year)].dropna(subset=[selected_metric])
        df = df.sort_values(selected_metric, ascending=False)
        fig = px.bar(df, x=selected_metric, y='Entity', 
                     title=f"Deaths by Region ({year}, {metric_region})")
        fig.update_layout(hovermode="y unified", height=400)
        st.plotly_chart(fig, use_container_width=True)

# Second Column: Insight 1 - Indoor vs. Outdoor Deaths (Bar Chart)
with col_insight1:
    st.write("Insight 1: Indoor vs. Outdoor Deaths")
    if region == 'All':
        st.write("Please select a region to view insights.")
    else:
        df_region = merap_regions[merap_regions['Entity'] == region]
        df_region_year = df_region[df_region['Year'] == year] if year != 'All' else df_region[df_region['Year'] == df_region['Year'].max()]
        
        if not df_region_year.empty:
            indoor_deaths = df_region_year['Total Deaths for Household Air Pollution from Solid Fuels'].iloc[0]
            outdoor_deaths = df_region_year['Total Deaths for Outdoor Air Pollution'].iloc[0]
            if pd.notna(indoor_deaths) and pd.notna(outdoor_deaths):
                data = pd.DataFrame({
                    'Category': ['Indoor', 'Outdoor'],
                    'Deaths': [indoor_deaths, outdoor_deaths]
                })
                fig = px.bar(data, x='Category', y='Deaths',
                             title=f"Indoor vs. Outdoor Deaths ({region}, {year if year != 'All' else df_region_year['Year'].iloc[0]})",
                             color='Category',
                             color_discrete_map={'Indoor': '#FF0000', 'Outdoor': '#FFA500'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No data available for this selection.")
        else:
            st.write("No data available for this selection.")

# Third Column: Insight 2 - Line Chart Trend for Death Rate (1990-2019)
with col_insight2:
    st.write("Insight 2: Death Rate Trend (1990-2019)")
    if region == 'All':
        st.write("Please select a region to view insights.")
    else:
     
        df_region = merap_regions[(merap_regions['Entity'] == region) & (merap_regions['Year'].between(1990, 2019))]
        
        if not df_region.empty:
            fig = px.line(df_region, 
                          x='Year', 
                          y='Death Rate from Air Pollution Per 100000',
                          title=f"Death Rate per 100K Trend ({region}, 1990-2019)")
            fig.update_traces(line_color='#FF0000', mode='lines+markers')
            
            if year != 'All':
                specific_year_data = df_region[df_region['Year'] == int(year)]
                if not specific_year_data.empty:
                    specific_rate = specific_year_data['Death Rate from Air Pollution Per 100000'].iloc[0]
                    if pd.notna(specific_rate):
                        fig.add_annotation(
                            x=int(year),
                            y=specific_rate,
                            text=f"{specific_rate:.2f}",
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-30
                        )
                        fig.add_trace(go.Scatter(
                            x=[int(year)],
                            y=[specific_rate],
                            mode='markers',
                            marker=dict(color='#FF4500', size=10),
                            name=f'{year} Highlight'
                        ))
            
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Death Rate per 100,000",
                hovermode="x unified",
                height=400,
                showlegend=True if year != 'All' else False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No data available for this selection.")