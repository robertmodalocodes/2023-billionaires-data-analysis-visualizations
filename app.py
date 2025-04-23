from pathlib import Path
import folium.map
import pandas as pd
import calendar
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lists
import geopandas as gpd

from functools import partial
from faicons import icon_svg
import altair as alt
import plotly.express as px
import folium
from folium.features import GeoJson
from folium.plugins import MarkerCluster
from folium.plugins import HeatMap
from plotly.colors import qualitative
from plotly import graph_objects as go
from shiny.express import render, input, ui
from shiny import reactive
from shiny.ui import page_navbar
from shinywidgets import render_plotly, render_altair, render_widget

# ui.page_opts(title="Billionaires Descriptive Statistics Dashboard",
#               # page_fn=partial(page_navbar, id="page"),
#              fillable=False)

# Global variables
FONT_COLOR = "#4C78A8"
FONT_TYPE = "Arial"


# Helper functions
def style_plotly_chart(fig, yaxis_title, xaxis_title):
    fig.update_layout(
        xaxis_title=xaxis_title,  # Remove x-axis label
        yaxis_title=yaxis_title,  # Change y-axis label
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Remove background color
        showlegend=False,  # Remove the legend
        coloraxis_showscale=False,
        font=dict(family="Arial", size=12, color=FONT_COLOR),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# Mapping function
def plot_global_folium_map(mapdata: pd.DataFrame, variable: str, fill_color: str, legend_name: str):

    df_map = mapdata.copy()

    map = folium.Map(location=[51.165691, 10.451526], zoom_start=1.5)

    folium.Choropleth(
        geo_data=df_map.to_json(),
        name="choropleth",
        data=df_map,
        columns=["country", variable],
        key_on="feature.properties.name",  # important
        fill_color=fill_color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
    ).add_to(map)
    folium.LayerControl().add_to(map)

    return map

# Mapping function for US data


def plot_us_folium_map(mapdata: pd.DataFrame, variable: str, fill_color: str, legend_name: str):

    df_map = mapdata.copy()

    # 40.669626, -97.543859
    map = folium.Map(location=[40.669626, -97.543859], zoom_start=4)

    folium.Choropleth(
        geo_data=df_map.to_json(),
        name="choropleth",
        data=df_map,
        columns=["state", variable],
        key_on="feature.properties.NAME_x",  # important
        fill_color=fill_color,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
    ).add_to(map)
    folium.LayerControl().add_to(map)

    return map

# Age histogram function


def create_age_histogram(df, hue_column):
    mean_age = df['age'].mean()
    median_age = df['age'].median()

    # Custom color mapping
    # Male: navy, Female: grey
    color_map_gender = {'M': 'darkblue', 'F': 'grey'}

    color_map_self_made = {True: 'darkblue', False: 'royalblue'}

    cdmap = color_map_gender if hue_column == 'gender' else color_map_self_made

    # Create histogram with grouped bars
    fig = px.histogram(df, x='age', color=hue_column,
                       nbins=10,
                       # If the hue values are known use color_discrete_sequence instead
                       color_discrete_map=cdmap,
                       title='Age distribution of billionaires',
                       labels={'age': 'Age'},
                       barmode='group')

    # Add mean line
    fig.add_vline(x=mean_age, line_dash='dash', line_color='red',
                  annotation_text=f'Mean: {mean_age:.2f}',
                  annotation_position='top right',
                  annotation_font_color='red')

    # Add median line
    fig.add_vline(x=median_age, line_dash='dot', line_color='blue',
                  annotation_text=f'Median: {median_age:.2f}',
                  annotation_position='bottom right',
                  annotation_font_color='green')

    # Update layout to match original style
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Age',
        yaxis_title=None,
        legend_title_text=hue_column,
        title_font_size=16,
        title_x=0.5,
        bargap=0.2,  # Space between bars in same location
        bargroupgap=0.1,  # Space between bar groups
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
            dtick=10,
            range=[0, 100]
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='whitesmoke'
        )
    )

    return fig


# Creat birth decade plot
def create_birthdecade_plot(df, hue_column):
    # Male: navy, Female: grey
    color_map_gender = {'M': 'darkblue', 'F': 'grey'}
    color_map_self_made = {True: 'darkblue', False: 'royalblue'}
    cdmap = color_map_gender if hue_column == 'gender' else color_map_self_made

    fig = px.bar(df,
                 x='birthDecade',
                 y='count',
                 color=hue_column,
                 color_discrete_map=cdmap)

    # fig.update_traces(marker_color=color())
    fig.update_layout(
        plot_bgcolor='white',
        xaxis_title='Birth Year Range',
        yaxis_title=None,
        legend_title_text=hue_column,
        title_font_size=16,
        title_x=0.5,
        bargap=0.2,  # Space between bars in same location
        bargroupgap=0.1,  # Space between bar groups
        hovermode='x unified',
        template='plotly_white',
        xaxis=dict(
            showgrid=False,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            showgrid=False,
            gridcolor='whitesmoke'
        )
    )

    return fig


# Create birth month plot
def create_birthmonth_plot(df, hue_column, y_axis_title):

    MONTH_NAMES = lists.MONTH_NAMES

    category = 'Gender' if hue_column == 'gender' else 'Self Made'

    # Create figure
    fig = go.Figure()

    # Add traces for each gender (assuming 'M' and 'F' exist)
    # colors = ['#1f77b4', '#ff7f0e']  # Tab10 first two colors
    colors_1 = ['grey', 'darkblue']
    colors_2 = ['royalblue', 'darkblue']
    colorset = colors_1 if hue_column == 'gender' else colors_2

    for i, hue_column in enumerate(df.columns):
        fig.add_trace(go.Bar(
            x=df.index.map(MONTH_NAMES),
            y=df[hue_column],
            name=hue_column,
            marker_color=colorset[i],
            text=df[hue_column],
            textposition='auto'
        ))

    # Update layout
    fig.update_layout(
        plot_bgcolor='white',
        barmode='stack',
        title=dict(
            text=f'Billionaires by birth month and {category}',
            font=dict(size=16, family=FONT_TYPE,
                      color=None),
            x=0.5
        ),
        xaxis=dict(
            title='Birth Month',
            title_font=dict(
                size=14, family=FONT_TYPE, color=None),
            tickfont=dict(
                size=12, family=FONT_TYPE, color=None),
            type='category',
            categoryorder='array',
            categoryarray=list(MONTH_NAMES.values())
        ),
        yaxis=dict(
            title=y_axis_title,
            title_font=dict(
                size=14, family=FONT_TYPE, color=None),
            tickfont=dict(
                size=12, family=FONT_TYPE, color=None)
        ),
        legend=dict(
            title=dict(text=f'{category}', font=dict(
                family=FONT_TYPE, size=12)),
            font=dict(size=10, family=FONT_TYPE),
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        ),
        margin=dict(t=40),
        width=None,
        height=None
    )

    return fig


# Main dataset


@reactive.calc
def data():
    infile = Path(__file__).parent / "bill_stats.csv"
    df = pd.read_csv(infile)

    df.replace(
        {
            np.nan: 0
        }, inplace=True)

    df = df.astype(
        {
            'category': 'category',
            'status': 'category',
            'gender': 'category',
            'birthDate': 'datetime64[ns]',
            'date': 'datetime64[ns]',
            'age': 'int',
            'birthYear': 'int',
            'birthMonth': 'int',
            'birthDay': 'int',
            'population_country': 'int'
        }
    )

    df['gdp_country'] = df['gdp_country'] \
        .str.replace('$', '').str.replace(',', '').str.strip()
    df['gdp_country'] = pd.to_numeric(df['gdp_country'])
    df['gdp_country'] = df['gdp_country'].fillna(0)
    df['gdp_country'] = df['gdp_country'].astype('int')

    df['nBillionairesCtr'] = df.groupby(
        'country')['country'].transform('count')

    df['totalWealthCtr'] = df.groupby('country')['finalWorth'].transform('sum')
    df['meanWealthCtr'] = df.groupby('country')['finalWorth'].transform('mean')

    df['nBillionairesIndy'] = df.groupby(
        'industries')['industries'].transform('count')

    df['totalWealthCtrIndy'] = df.groupby(
        'industries')['finalWorth'].transform('sum')
    df['meanWealthCtrIndy'] = df.groupby(
        'industries')['finalWorth'].transform('mean')

    df['birthDecade'] = pd.cut(df['birthYear'], bins=range(
        1920, 2005, 10), include_lowest=True, right=False)
    df['birthDecade'] = df['birthDecade'].astype('str')
    df['birthDecade'] = df['birthDecade'].str.replace(
        '[', '').str.replace(',', ' -').str.replace(')', '')
    df['birthDecade'] = df['birthDecade'].replace('2000+', np.nan)

    df = df.drop(columns=["category", "organization",
                 "status", "lastName", "firstName", "title"], axis=1)

    return df


# USA Dataset
@reactive.calc
def data_usa():
    data_usa = data().copy()
    df = data_usa[(data_usa["country"] == "United States") | (
        data_usa["countryOfCitizenship"] == "United States")]

    df[["city", "state", "residenceStateRegion"]] = df[[
        "city", "state", "residenceStateRegion"]].replace({0: "Unknown"})

    return df


# USA map dataset
@reactive.calc
def united_states_map_data():
    mapfile_base = Path(__file__).parent / \
        "./tl_2023_us_state/tl_2023_us_state.shp"
    datafile_us_pop = Path(__file__).parent / \
        "./datasets/NST-EST2023-ALLDATA.csv"
    datafile_us_gdp = Path(__file__).parent / "./datasets/us_state_gdp.csv"
    datafile_us = data_usa().copy()

    df_mapfile_base = gpd.read_file(mapfile_base)
    df_us_pop = pd.read_csv(datafile_us_pop)
    df_us_gdp = pd.read_csv(datafile_us_gdp, sep=";")
    df_us = datafile_us

    # Data processing
    df_us["nBillionaires"] = df_us.groupby('state', observed=False)[
        'state'].transform('count')
    df_us['totalWealth'] = df_us.groupby('state', observed=False)[
        'finalWorth'].transform('sum')
    df_us['meanWealth'] = df_us.groupby('state', observed=False)[
        'finalWorth'].transform('mean')

    df_us["nBillionaires"] = df_us["nBillionaires"].fillna(0).astype('int')

    df_to_join_us_state = df_us[[
        'state', 'nBillionaires', 'totalWealth', 'meanWealth']].copy()
    df_to_join_us_state.dropna(inplace=True)

    states_grouped = df_to_join_us_state.drop_duplicates().reset_index()
    states_grouped.rename(columns={'index': 'origin_idx'}, inplace=True)

    pop_us_states_to_merge = df_us_pop[['NAME', 'POPESTIMATE2023']].copy()

    states_grouped_first = states_grouped.merge(
        pop_us_states_to_merge, left_on='state', right_on='NAME', how='left')
    states_grouped_first.drop(['origin_idx'], axis=1, inplace=True)

    df_us_gdp['State'] = df_us_gdp['State'].str.lstrip().str.rstrip()
    states_grouped_first['state'] = states_grouped_first['state'].str.lstrip(
    ).str.rstrip()

    states_grouped_first = states_grouped_first.merge(
        df_us_gdp, left_on='state', right_on='State', how='left')
    states_grouped_first.drop(['State'], axis=1, inplace=True)

    df_mapfile_base['NAME'] = df_mapfile_base['NAME'].str.lstrip().str.rstrip()
    states_grouped_first['state'] = states_grouped_first['state'].str.lstrip(
    ).str.rstrip()

    states_grouped_first.rename(columns={'totalWealth': 'billTotalWealth',
                                         'meanWealth': 'billMeanWealth'},
                                inplace=True)

    shp_us_states_economics = df_mapfile_base.merge(
        states_grouped_first, left_on='NAME', right_on='state', how='left')

    # shp_us_states_econ_no_hawaii = shp_us_states_economics.drop(shp_us_states_economics.loc[shp_us_states_economics['NAME_y'] == 'Hawaii'].index).copy()
    # drop state of hawaii to fit the US mainland map in full extent

    # shp_us_states_economics.drop('NAME_y', axis=1, inplace=True)
    # change to shp_us_states_econ_no_hawaii instead if that dataframe is to be used

    return shp_us_states_economics


# Global map dataset
@reactive.calc
def global_map_data():
    mapfile = Path(__file__).parent / \
        "./world-administrative-boundaries/world-administrative-boundaries.shp"

    datafile = data().copy()

    df_global_map = gpd.read_file(mapfile)

    df_to_join_countries = datafile[['latitude_country',
                                     'longitude_country',
                                     'country',
                                     'nBillionairesCtr',
                                     'totalWealthCtr',
                                     'meanWealthCtr',
                                     'population_country',
                                     'gdp_country']].copy()

    df_to_join_countries.dropna(inplace=True)
    df_to_join_countries.drop_duplicates(inplace=True)

    df_to_join_countries = df_to_join_countries.replace(
        {
            "United States": "United States of America",
            "United Kingdom": "U.K. of Great Britain and Northern Ireland",
            "Russia": "Russian Federation"
        })

    df_global_map = pd.merge(left=df_global_map,
                             right=df_to_join_countries,
                             left_on='name',
                             right_on='country',
                             how='left')

    df_global_map = df_global_map.dropna(
        axis="index").drop_duplicates(keep="first")

    return df_global_map


ui.page_opts(title="Billionaires Descriptive Statistics Dashboard",
             page_fn=partial(page_navbar, id="page"),
             fillable=False)


# Page 1 - Global panel
with ui.nav_panel("Global"):

    # Customizing dashboard page with html and css
    ui.tags.style(
        """
        body {
            background-color: #5DADE2;
        }

        .container, .container-fluid, .container-xxl, .container-xl, .container-lg, .container-md, .container-sm {
            --bs-gutter-x: 1.5rem;
            --bs-gutter-y: 0;
            width: 86%;
            padding-right: calc(var(--bs-gutter-x)* .5);
            padding-left: calc(var(--bs-gutter-x)* .5);
            margin-right: auto;
            margin-left: auto;
        }

        @media (min-width: 576px) {
            .container-sm, .container {
                max-width: 540px;
            }
        }

        @media (min-width: 768px) {
            .container-md, .container-sm, .container {
                max-width: 720px;
            }
        }

        @media (min-width: 992px) {
            .container-lg, .container-md, .container-sm, .container {
                max-width: 960px;
            }
        }

        """
    )

    # Section 1 - Overview KPI cards/value boxes
    with ui.layout_column_wrap(width=(1/3)):
        with ui.value_box(showcase=icon_svg("person"), theme="bg-gradient-indigo-purple"):
            "Total billionaires"
            @render.ui
            def total_billionaires():
                df = data()
                return ui.markdown(f"{len(df)}")

        with ui.value_box(showcase=icon_svg("chart-line"), theme="text-green"):
            "Global average billionaires wealth"
            @render.ui
            def average_wealth():
                df = data()
                global_mean_wealth = df['finalWorth'].mean()
                global_mean_wealth_str = f"$ {global_mean_wealth:,.2f} (million)"
                return ui.markdown(global_mean_wealth_str)

        with ui.value_box(showcase=icon_svg("chart-bar"), theme="bg-gradient-purple-blue"):
            "Global total billionaires wealth"
            @render.ui
            def total_wealth():
                df = data()
                global_total_wealth = df['finalWorth'].sum()
                global_total_wealth_str = f"$ {global_total_wealth:,.0f} (million)"
                return ui.markdown(global_total_wealth_str)

    # Section 2 - Billionaires ranking
    with ui.navset_card_underline(id="names", header=None, footer=ui.input_numeric("n_billionaires",
                                                                                   "Number of billionaires to show", 10, min=10, max=40)):

        # Unorganized
        with ui.nav_panel("Overall"):
            "Global billionaires ranking"

            @render_plotly
            def plot_names():
                df = data()

                fig = px.bar(df.head(input.n_billionaires()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        # Organize by country
        with ui.nav_panel("Country"):
            "Billionaires by country"

            ui.input_select("country_list", "Select country",
                            sorted(lists.COUNTRY_LIST),
                            multiple=False,
                            selected="United States")

            @render_plotly
            def plot_names_ctr():
                df = data().copy()

                df_grouped_ctr = df[df["country"] == input.country_list()]

                fig = px.bar(df_grouped_ctr.head(input.n_billionaires()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='Blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        # Organize by industry
        with ui.nav_panel("Industry"):
            "Billionaires by industry"

            ui.input_select("industry_list", "Select industry",
                            sorted(lists.INDUSTRY_LIST),
                            multiple=False,
                            selected=None)

            @render_plotly
            def plot_names_indy():
                df = data().copy()

                df_grouped_indy = df[df["industries"] == input.industry_list()]

                fig = px.bar(df_grouped_indy.head(input.n_billionaires()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='Blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        # Organize by country and industry
        with ui.nav_panel("Country & Industry"):
            "Billionaires by country and industry"

            ui.input_select("country_list_ctr", "Select country",
                            sorted(lists.COUNTRY_LIST),
                            multiple=False,
                            selected="United States")

            ui.input_select("industry_list_indy", "Select industry",
                            sorted(lists.INDUSTRY_LIST),
                            multiple=False,
                            selected="Diversified")

            @render_plotly
            def plot_names_ctr_indy():
                df = data().copy()

                df_grouped_ctr_indy = df[(df["country"] == input.country_list_ctr()) & (
                    df["industries"] == input.industry_list_indy())]

                fig = px.bar(df_grouped_ctr_indy.head(input.n_billionaires()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='Blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        with ui.nav_menu("Other"):
            with ui.nav_panel("Pie chart"):

                ui.input_select(id="pie_gender_selfmade",
                                label="Select category",
                                choices={"gender": "Gender",
                                         "selfMade": "Self Made"},
                                selected=None,
                                multiple=False)

                @render_plotly
                def plot_pie_gender_selfmade():

                    df = data().copy()

                    color_map_gender = {'M': 'darkblue', 'F': 'grey'}
                    color_map_self_made = {
                        True: 'darkblue', False: 'royalblue'}
                    cdmap = color_map_gender if input.pie_gender_selfmade(
                    ) == 'gender' else color_map_self_made

                    group = df.groupby(input.pie_gender_selfmade(
                    ), observed=False).size().reset_index(name='nBill')

                    fig = px.pie(group,
                                 values='nBill',
                                 names=input.pie_gender_selfmade(),
                                 color=input.pie_gender_selfmade(),
                                 title=f'Billionaires by {input.pie_gender_selfmade()}',
                                 color_discrete_map=cdmap)

                    fig.update_layout(
                        legend_title_text=input.pie_gender_selfmade())

                    return fig

            with ui.nav_control():
                ui.a("Shiny", href="https://shiny.posit.co", target="_blank")

    # Section 3 - Grouping by industry and country
    with ui.layout_column_wrap(width=(1/2)):
        with ui.navset_card_underline(id="bill_industry", header="Industry", footer=ui.input_numeric("number_indy", "Number of industries", 5, min=5, max=20)):
            with ui.nav_panel("Top"):
                "Top industries with most billionaires"

                @render_plotly
                def plot_top_industry():
                    df = data()

                    top_industry = df.groupby("industries").size().nlargest(
                        input.number_indy()).reset_index(name='count')

                    fig = px.bar(top_industry.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="industries",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Bottom"):
                "Industries with least billionaires"

                @render_plotly
                def plot_bottom_industry():
                    df = data()

                    bottom_industry = df.groupby("industries").size().nsmallest(
                        input.number_indy()).reset_index(name='count')

                    fig = px.bar(bottom_industry.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="industries",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Top Mean Worth"):
                "Industries with most average billionaire wealth"

                @render_plotly
                def plot_top_industry_meanwealth():
                    df = data()

                    top_industry_meanwealth = df.groupby("industries")['finalWorth'].mean().nlargest(
                        input.number_indy()).reset_index(name='mean_wealth')

                    fig = px.bar(top_industry_meanwealth.sort_values(by="mean_wealth", ascending=True),
                                 x="mean_wealth",
                                 y="industries",
                                 color="mean_wealth",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title="Average Wealth (Million $)")

                    return fig

            with ui.nav_menu("More"):
                with ui.nav_panel("Total Worth"):
                    "Industries with most billionaire wealth"

                    @render_plotly
                    def plot_top_industry_totalwealth():
                        df = data()

                        top_industry_sumwealth = df.groupby("industries")['finalWorth'].sum().nlargest(
                            input.number_indy()).reset_index(name='total_wealth')

                        fig = px.bar(top_industry_sumwealth.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="industries",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title="Total Worth (Million $)")

                        return fig

                with ui.nav_panel("Industry pie chart"):

                    @render_plotly
                    def plot_pie_industry():

                        df = data().copy()

                        group = df.groupby(
                            'industries', observed=False).size().reset_index(name='nBill')

                        fig = px.pie(group,
                                     values='nBill',
                                     names='industries',
                                     color='industries',
                                     title=f'Billionaires by industry',
                                     color_discrete_sequence=px.colors.qualitative.G10)  # plotly colors

                        fig.update_layout(
                            legend_title_text='Industry')

                        return fig

        with ui.navset_card_underline(id="bill_country", header="Country", footer=ui.input_numeric("number_ctr", "Number of countries", 5, min=5, max=20)):
            with ui.nav_panel("Top"):
                "Countries with most billionaires"

                @render_plotly
                def plot_top_country():
                    df = data()

                    top_country = df.groupby("country").size().nlargest(
                        input.number_ctr()).reset_index(name='count')

                    fig = px.bar(top_country.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="country",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Bottom"):
                "Countries with least billionaires"

                @render_plotly
                def plot_bottom_country():
                    df = data()

                    bottom_country = df.groupby("country").size().nsmallest(
                        input.number_ctr()).reset_index(name='count')

                    fig = px.bar(bottom_country.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="country",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Top Mean Wealth"):
                "Countries most average billionaires wealth"

                @render_plotly
                def plot_top_country_meanwealth():
                    df = data()

                    top_country_meanwealth = df.groupby("country")['finalWorth'].mean().nlargest(
                        input.number_ctr()).reset_index(name='mean_wealth')

                    fig = px.bar(top_country_meanwealth.sort_values(by="mean_wealth", ascending=True),
                                 x="mean_wealth",
                                 y="country",
                                 color="mean_wealth",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title='Average Wealth (million $)')

                    return fig

            with ui.nav_menu("More"):
                with ui.nav_panel("Bottom Mean Wealth"):
                    "Countries least average billionaires wealth"

                    @render_plotly
                    def plot_bottom_country_meanwealth():
                        df = data()

                        bottom_country_meanwealth = df.groupby("country")['finalWorth'].mean().nsmallest(
                            input.number_ctr()).reset_index(name='mean_wealth')

                        fig = px.bar(bottom_country_meanwealth.sort_values(by="mean_wealth", ascending=True),
                                     x="mean_wealth",
                                     y="country",
                                     color="mean_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Average Wealth (million $)')

                        return fig

                with ui.nav_panel("Top Total Wealth"):
                    "Countries most billionaires wealth combined"

                    @render_plotly
                    def plot_top_country_totalwealth():
                        df = data()

                        top_country_totalwealth = df.groupby("country")['finalWorth'].sum().nlargest(
                            input.number_ctr()).reset_index(name='total_wealth')

                        fig = px.bar(top_country_totalwealth.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="country",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Total Wealth (million $)')

                        return fig

                with ui.nav_panel("Bottom Total Wealth"):
                    "Countries least average billionaires wealth"

                    @render_plotly
                    def plot_bottom_country_totalwealth():
                        df = data()

                        bottom_country_totalwealth = df.groupby("country")['finalWorth'].sum().nlargest(
                            input.number_ctr()).reset_index(name='total_wealth')

                        fig = px.bar(bottom_country_totalwealth.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="country",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Total Wealth (million $)')

                        return fig

    # Section 4 - Total Wealth across Industries by Countries
    with ui.card():

        ui.input_numeric("barh_ctr", "Number of countries",
                         6, min=6, max=20)

        # ui.input_numeric("barh_indy", "Number of industries", 6, min=6, max=40)

        @render.plot
        def plot_ind_ctr_stacked_test():

            df = data().copy()

            # Calculate total wealth by country
            total_wealth_by_country = df.groupby(
                'country')['finalWorth'].sum().reset_index()
            # Select the top 10 countries by total wealth
            top_countries = total_wealth_by_country.nlargest(
                input.barh_ctr(), 'finalWorth')['country']
            # Filter data for the top 10 countries
            filtered_data = df[df['country'].isin(top_countries)]
            # Group data by country and industries, calculate total wealth
            wealth_by_industry_country = filtered_data.groupby(['country', 'industries'])[
                'finalWorth'].sum().reset_index()
            # Pivot the data for easier plotting (industries as columns)
            wealth_pivot = wealth_by_industry_country.pivot(
                index='industries', columns='country', values='finalWorth').fillna(0)
            # Calculate total wealth per industry and sort industries in descending order
            wealth_pivot['TotalWealth'] = wealth_pivot.sum(
                axis=1)  # Add a column for total wealth
            wealth_pivot = wealth_pivot.sort_values(
                by='TotalWealth', ascending=True)  # Sort by total wealth
            # Drop the helper column after sorting
            wealth_pivot = wealth_pivot.drop(columns='TotalWealth')

            # Convert the pivot table to long format
            # Create the stacked bar chart
            # plt.figure(figsize=(15, 5))
            # plt.figure()
            fig = wealth_pivot.plot(kind='barh', stacked=True,
                                    figsize=(10, 5), cmap='tab10')

            # Customize the plot
            plt.title(
                'Accumulated Billionaire Wealth Across Industries for Top n Countries', fontsize=16, pad=20)
            plt.xlabel('Accumulated Wealth (million $)', fontsize=14)
            plt.ylabel('Industry', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.legend(title='Country', bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()

            return fig

    # Section 5 - Grouping by age and birth year range
    with ui.layout_column_wrap(width=(1/2)):
        with ui.navset_card_underline(id="age_dist",
                                      header="Age Distribution",
                                      footer=ui.input_select("gender_selfmade", "Select group", [
                                          'gender', 'selfMade'],
                                          multiple=False, selected=None)):
            with ui.nav_panel("All category"):

                @render_plotly
                def plot_agedist_all():

                    df = data()

                    # Should return the column name for selected category
                    hue = input.gender_selfmade()

                    fig = create_age_histogram(df, hue_column=hue)

                    return fig

            with ui.nav_panel("Industry"):
                "By Industry"

                ui.input_select("gender_selfmade_select_indy", "Select industry",
                                sorted(lists.INDUSTRY_LIST),
                                multiple=False,
                                selected=None)

                @render_plotly
                def plot_agedist_indy():

                    df = data()

                    df_industry = df[df['industries'] ==
                                     input.gender_selfmade_select_indy()]

                    # Should return the column name for selected category
                    hue = input.gender_selfmade()

                    fig = create_age_histogram(df_industry, hue_column=hue)

                    return fig

            with ui.nav_panel("Country"):
                "By Country"

                ui.input_select("gender_selfmade_select_ctr", "Select country",
                                sorted(lists.COUNTRY_LIST),
                                multiple=False,
                                selected='United States')

                @render_plotly
                def plot_agedist_ctr():

                    df = data()

                    df_country = df[df['country'] ==
                                    input.gender_selfmade_select_ctr()]

                    # Should return the column name for selected category
                    hue = input.gender_selfmade()

                    fig = create_age_histogram(df_country, hue_column=hue)

                    return fig

        with ui.navset_card_underline(id="birth_decade_dist",
                                      header="Distribution by birth year range",
                                      footer=ui.input_select("gender_selfmade_bdec",
                                                             "Select group",
                                                             ['gender',
                                                              'selfMade'],
                                                             multiple=False, selected=None)):

            with ui.nav_panel("Overall"):

                @render_plotly
                def birth_decade_overall():

                    df = data()

                    hue = input.gender_selfmade_bdec()

                    decade_grouped = df.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count').drop(index=[16, 17])

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

            with ui.nav_panel("Industry"):
                "Distribution of billionaires by birth year range and industry"

                ui.input_select("birth_decade_indy", "Select industry",
                                sorted(lists.INDUSTRY_LIST),
                                multiple=False,
                                selected=None)

                @render_plotly
                def birth_decade_industry():

                    df = data()

                    df = df[df['industries'] == input.birth_decade_indy()]

                    hue = input.gender_selfmade_bdec()

                    decade_grouped = df.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count')

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

            with ui.nav_panel("Country"):
                "Distribution of billionaires by birth year range and country"

                ui.input_select("birth_decade_ctr", "Select country",
                                sorted(lists.COUNTRY_LIST),
                                multiple=False,
                                selected='United States')

                @render_plotly
                def birth_decade_country():

                    df = data()

                    df = df[df['country'] == input.birth_decade_ctr()]

                    hue = input.gender_selfmade_bdec()

                    decade_grouped = df.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count')

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

    # Section 6 : Correlations / scatter plots
    with ui.layout_column_wrap(width=(1/2)):
        with ui.navset_card_underline(id="scatterplots", footer=None):

            # "Scatter plots of numeric variables with number of billionaires, total wealth, and average wealth"
            with ui.nav_panel("Scatter Plots"):

                ui.input_select("x_variable", "Select x variable:", sorted(
                    lists.NUMERIC_COLUMNS), multiple=False, selected=None)

                ui.input_select("y_variable", "Select y variable:", sorted(
                    lists.NUMERIC_COLUMNS), multiple=False, selected="nBillionairesCtr")

                ui.input_checkbox("add_trendline", "Add trendline", False)

                @render_plotly
                def plot_correlations():

                    df = data().copy()

                    # df_num = df[df.select_dtypes(
                    #    include=['number']).columns]

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=df[input.x_variable()],
                        y=df[input.y_variable()],
                        mode='markers',
                        marker_color='rgba(20, 60, 120, 0.8)',
                        hoverinfo='text',  # Hover text over each data point
                        text=df['country']
                    ))

                    # Calculate trendline
                    x_data = df[input.x_variable()]
                    y_data = df[input.y_variable()]
                    z = np.polyfit(x_data, y_data, 1)
                    trendline = np.poly1d(z)(np.linspace(
                        x_data.min(), x_data.max(), 100))

                    # Add trendline based on event listener (checkbox)
                    if input.add_trendline() is True:
                        fig.add_trace(go.Scatter(
                            x=np.linspace(x_data.min(), x_data.max(), 100),
                            y=trendline,
                            mode='lines',
                            line=dict(color='firebrick', width=3, dash='dot'),
                            name=f'Trendline: y = {z[0]:.2f}x + {z[1]:.2f}'
                        ))
                    else:
                        pass

                    # fig.update_traces(marker_color=color())

                    fig.update_layout(
                        plot_bgcolor='white',
                        legend_title_text=None,
                        title_font_size=16,
                        title_x=0.5,
                        xaxis=dict(
                            title_text=input.x_variable(),
                            showgrid=False,
                            gridcolor='lightgray',
                        ),
                        yaxis=dict(
                            title_text=input.y_variable(),
                            showgrid=False,
                            gridcolor='whitesmoke'
                        ))

                    return fig

            # "Correlations between numeric variables with number of billionaires, total wealth, and average wealth"
            with ui.nav_panel("Correlations"):

                @render_plotly
                def correlations():

                    df = data().copy()

                    df_num = df[df.select_dtypes(include=['number']).columns]

                    # Drop irrelevant columns/variables
                    df_num = df_num.drop(
                        ['rank', 'finalWorth', 'age', 'birthYear', 'birthMonth', 'birthDay', 'nBillionairesIndy', 'totalWealthCtrIndy', 'meanWealthCtrIndy'], axis='columns')

                    corr = df_num.corr()

                    fig = go.Figure(data=go.Heatmap(
                                    z=corr.values,
                                    x=corr.columns,
                                    y=corr.columns,
                                    colorscale='Blues',
                                    showscale=False,
                                    # Show 2 decimal places
                                    texttemplate="%{z:.2f}",
                                    textfont=dict(
                                        color='white', family=FONT_TYPE),
                                    ))

                    fig.update_layout(
                        xaxis=dict(
                            showticklabels=True,
                            tickangle=-45,
                            title_font=dict(color=FONT_COLOR,
                                            family=FONT_TYPE),
                            tickfont=dict(color=FONT_COLOR, family=FONT_TYPE),
                            side='bottom'  # Puts x-axis labels at bottom
                        ),
                        yaxis=dict(
                            showticklabels=True,
                            autorange='reversed',  # Top-left to bottom-right matrix orientation
                            title_font=dict(color=FONT_COLOR,
                                            family=FONT_TYPE),
                            tickfont=dict(color=FONT_COLOR, family=FONT_TYPE),
                        ),
                        font=dict(family=FONT_TYPE, color=FONT_COLOR),
                        # Increase left margin for y-axis labels
                        margin=dict(t=30, l=150),
                        width=800,  # Adjust based on number of columns
                        height=700  # Keep square aspect ratio
                    )

                    return fig

        with ui.navset_card_underline(id="birth_months", footer=None):

            with ui.nav_panel("Country"):
                ui.input_select("bmonth_category",
                                "Select group category",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_ctr",
                                "Select country",
                                sorted(lists.COUNTRY_LIST_ALL),
                                selected="All",
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender():

                    hue_column = input.bmonth_category()

                    # Data processing
                    df = data().copy()

                    if input.bmonth_ctr() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['country'] == input.bmonth_ctr()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column]).size(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Billionaires')

                    return birth_month_plot

            with ui.nav_panel("Industry"):
                ui.input_select("bmonth_category_indy",
                                "Select group category",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_indy",
                                "Select industry",
                                sorted(lists.INDUSTRY_LIST_ALL),
                                selected=None,
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender_indy():

                    hue_column = input.bmonth_category_indy()

                    # Data processing
                    df = data().copy()

                    if input.bmonth_indy() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['industries'] == input.bmonth_indy()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column]).size(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Billionaires')

                    return birth_month_plot

            with ui.nav_menu("Other"):

                with ui.nav_panel("Total wealth by country"):
                    ui.input_select("bmonth_category_ctr_total",
                                    "Select group category",
                                    ['gender', 'selfMade'],
                                    selected=None,
                                    multiple=False)

                    ui.input_select("bmonth_ctr_total",
                                    "Select country",
                                    sorted(lists.COUNTRY_LIST_ALL),
                                    selected="All",
                                    multiple=False)

                    @render_plotly
                    def plot_birthmonth_gender_ctr_total():

                        hue_column = input.bmonth_category_ctr_total()

                        # Data processing
                        df = data().copy()

                        if input.bmonth_ctr_total() == 'All':
                            df_use = df
                        else:
                            df_use = df[df['country'] ==
                                        input.bmonth_ctr_total()]

                        df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].sum(
                        ).reset_index(name='count')

                        rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                        df_bmonth_category.drop(
                            index=rows_to_drop, inplace=True)

                        df_pivoted = df_bmonth_category.pivot(
                            index='birthMonth', columns=hue_column, values='count')

                        birth_month_plot = create_birthmonth_plot(
                            df_pivoted, hue_column=hue_column, y_axis_title='Total Billionaire Wealth')

                        return birth_month_plot

                with ui.nav_panel("Average wealth by country"):
                    ui.input_select("bmonth_category_ctr_mean",
                                    "Select group category",
                                    ['gender', 'selfMade'],
                                    selected=None,
                                    multiple=False)

                    ui.input_select("bmonth_ctr_mean",
                                    "Select country",
                                    sorted(lists.COUNTRY_LIST_ALL),
                                    selected="All",
                                    multiple=False)

                    @render_plotly
                    def plot_birthmonth_gender_ctr_mean():

                        hue_column = input.bmonth_category_ctr_mean()

                        # Data processing
                        df = data().copy()

                        if input.bmonth_ctr_mean() == 'All':
                            df_use = df
                        else:
                            df_use = df[df['country'] ==
                                        input.bmonth_ctr_mean()]

                        df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].mean(
                        ).reset_index(name='count')

                        rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                        df_bmonth_category.drop(
                            index=rows_to_drop, inplace=True)

                        df_pivoted = df_bmonth_category.pivot(
                            index='birthMonth', columns=hue_column, values='count')

                        birth_month_plot = create_birthmonth_plot(
                            df_pivoted, hue_column=hue_column, y_axis_title='Average Billionaire Wealth')

                        return birth_month_plot

                with ui.nav_panel("Total wealth by industry"):
                    ui.input_select("bmonth_category_indy_total",
                                    "Select group category",
                                    ['gender', 'selfMade'],
                                    selected=None,
                                    multiple=False)

                    ui.input_select("bmonth_indy_total",
                                    "Select industry",
                                    sorted(lists.INDUSTRY_LIST_ALL),
                                    selected=None,
                                    multiple=False)

                    @render_plotly
                    def plot_birthmonth_gender_indy_total():

                        hue_column = input.bmonth_category_indy_total()

                        # Data processing
                        df = data().copy()

                        if input.bmonth_indy_total() == 'All':
                            df_use = df
                        else:
                            df_use = df[df['industries'] ==
                                        input.bmonth_indy_total()]

                        df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].sum(
                        ).reset_index(name='count')

                        rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                        df_bmonth_category.drop(
                            index=rows_to_drop, inplace=True)

                        df_pivoted = df_bmonth_category.pivot(
                            index='birthMonth', columns=hue_column, values='count')

                        birth_month_plot = create_birthmonth_plot(
                            df_pivoted, hue_column=hue_column, y_axis_title='Total Billionaire Wealth')

                        return birth_month_plot

                with ui.nav_panel("Average wealth by industry"):
                    ui.input_select("bmonth_category_indy_mean",
                                    "Select group category",
                                    ['gender', 'selfMade'],
                                    selected=None,
                                    multiple=False)

                    ui.input_select("bmonth_indy_mean",
                                    "Select industry",
                                    sorted(lists.INDUSTRY_LIST_ALL),
                                    selected=None,
                                    multiple=False)

                    @render_plotly
                    def plot_birthmonth_gender_indy_mean():

                        hue_column = input.bmonth_category_indy_mean()

                        # Data processing
                        df = data().copy()

                        if input.bmonth_indy_mean() == 'All':
                            df_use = df
                        else:
                            df_use = df[df['industries'] ==
                                        input.bmonth_indy_mean()]

                        df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].mean(
                        ).reset_index(name='count')

                        rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                        df_bmonth_category.drop(
                            index=rows_to_drop, inplace=True)

                        df_pivoted = df_bmonth_category.pivot(
                            index='birthMonth', columns=hue_column, values='count')

                        birth_month_plot = create_birthmonth_plot(
                            df_pivoted, hue_column=hue_column, y_axis_title='Average Billionaire Wealth')

                        return birth_month_plot

    # Section 7 - Maps
    with ui.navset_card_underline(id="chloropleth_map"):

        # Billionaires distribution map
        with ui.nav_panel("Billionaires Distribution"):
            "Number of Billionaires"

            @render.ui
            def plot_bill_map():

                mapdata = global_map_data()

                bill_dist_map = plot_global_folium_map(mapdata,
                                                       "nBillionairesCtr", "GnBu", "Number of Billionaires")

                return bill_dist_map

        # Total billionaires wealth by country map
        with ui.nav_panel("Total Wealth"):
            "Total billionaires wealth by country"

            @render.ui
            def plot_bill_map_total():

                mapdata = global_map_data()

                total_wealth_map = plot_global_folium_map(mapdata,
                                                          "totalWealthCtr", "GnBu", "Total wealth ($ million)")

                return total_wealth_map

        # Average billionaires wealth by country map
        with ui.nav_panel("Average Wealth"):
            "Average billionaires wealth by country"

            @render.ui
            def plot_bill_map_avr():

                mapdata = global_map_data()

                avr_wealth_map = plot_global_folium_map(mapdata,
                                                        "meanWealthCtr", "GnBu", "Average wealth ($ million)")

                return avr_wealth_map

        # Population by country map
        with ui.nav_panel("Population"):
            "Population"

            @render.ui
            def plot_pop_map():

                mapdata = global_map_data()

                pop_map = plot_global_folium_map(mapdata,
                                                 "population_country", "Reds", "population_country")

                return pop_map

        # GDP by country map
        with ui.nav_panel("GDP"):
            "Gross Domestic Product"

            @render.ui
            def plot_gdp_map():

                mapdata = global_map_data()

                gdp_map = plot_global_folium_map(mapdata,
                                                 "gdp_country", "Greens", "GDP")

                return gdp_map

    # Section 8 - Dataset
    with ui.card():
        ui.card_header("Sample Data")

        ui.input_select("select_button_option",
                        "Option:",
                        ["Top 100 ", "Random 100", "Bottom 100"],
                        multiple=False,
                        selected=None)

        # Define buttons within a horizontal container for proper display
        ui.input_action_button("show_dataframe", "Show data", width="15%")

        @render.data_frame
        @reactive.event(input.show_dataframe)
        def show_data():

            df = data().copy()

            if input.select_button_option() == "Top 100":
                df = df.head(100)
            elif input.select_button_option() == "Random 100":
                df = df.sample(100)
            elif input.select_button_option() == "Bottom 100":
                df = df.tail(100)

            # input.show_dataframe()
            return render.DataGrid(
                data=df,
                selection_mode='row',
                filters=True
            )

    # Section 9 - Panel footer (info)
    with ui.card_footer():
        @render.ui
        def panel_footer():
            return ui.markdown("Created by Robert Modalo")


# Page 2 - United States panel
with ui.nav_panel("United States"):

    # Section 1 - Overview KPI cards/value boxes
    with ui.layout_column_wrap(width=(1/3)):
        with ui.value_box(showcase=icon_svg("person"), theme="bg-gradient-indigo-purple"):
            "Total billionaires in the US"
            @render.ui
            def total_billionaires_usa():
                df = data_usa()
                return ui.markdown(f"{len(df)}")

        with ui.value_box(showcase=icon_svg("chart-line"), theme="text-green"):
            "US average billionaires wealth"
            @render.ui
            def average_wealth_usa():
                df = data_usa()
                usa_mean_wealth = df['finalWorth'].mean()
                usa_mean_wealth_str = f"$ {usa_mean_wealth:,.2f} (million)"
                return ui.markdown(usa_mean_wealth_str)

        with ui.value_box(showcase=icon_svg("chart-bar"), theme="bg-gradient-purple-blue"):
            "US total billionaires wealth"
            @render.ui
            def total_wealth_usa():
                df = data_usa()
                usa_total_wealth = df['finalWorth'].sum()
                usa_total_wealth_str = f"$ {usa_total_wealth:,.0f} (million)"
                return ui.markdown(usa_total_wealth_str)

    # Section 2 - Billionaires ranking
    with ui.navset_card_underline(id="names_usa", header=None, footer=ui.input_numeric("n_billionaires_us",
                                                                                       "Number of billionaires to show", 10, min=10, max=40)):

        # Unorganized
        with ui.nav_panel("Overall"):
            "US billionaires ranking"

            ui.input_select("us_state_list",
                            "Select State:",
                            sorted(lists.US_STATES_ALL),
                            multiple=False,
                            selected="All")

            @render_plotly
            def plot_names_usa():

                df = data_usa().copy()

                data_to_use: pd.DataFrame

                if input.us_state_list() == "All":
                    data_to_use = df
                else:
                    data_to_use = df[df["state"] == input.us_state_list()]

                fig = px.bar(data_to_use.head(input.n_billionaires_us()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        # Organize by industry
        with ui.nav_panel("Industry"):
            "US billionaires by industry"

            ui.input_select("us_industry_list", "Select industry:",
                            sorted(lists.INDUSTRY_LIST_ALL),
                            multiple=False,
                            selected="All")

            @render_plotly
            def plot_names_indy_usa():
                df = data_usa().copy()

                df_grouped_indy_us: pd.DataFrame

                if input.us_industry_list() == "All":
                    df_grouped_indy_us = df
                else:
                    df_grouped_indy_us = df[df["industries"]
                                            == input.us_industry_list()]

                fig = px.bar(df_grouped_indy_us.head(input.n_billionaires_us()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='Blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        # Organize by state and industry
        with ui.nav_panel("State & Industry"):
            "US billionaires by state and industry"

            ui.input_select("us_state_indy_list", "Select state",
                            sorted(lists.US_STATES_ALL),
                            multiple=False,
                            selected="All")

            ui.input_select("us_state_indy_list_indy", "Select industry",
                            sorted(lists.INDUSTRY_LIST_ALL),
                            multiple=False,
                            selected="All")

            @render_plotly
            def plot_names_state_indy_usa():
                df = data_usa().copy()

                df_grouped: pd.DataFrame

                if (input.us_state_indy_list() == "All") & (input.us_state_indy_list_indy() == "All"):
                    df_grouped = df
                elif (input.us_state_indy_list() == "All") & (input.us_state_indy_list_indy() == input.us_state_indy_list_indy()):
                    df_grouped = df[df["industries"] ==
                                    input.us_state_indy_list_indy()]
                elif (input.us_state_indy_list() == input.us_state_indy_list()) & (input.us_state_indy_list_indy() == "All"):
                    df_grouped = df[df["state"] == input.us_state_indy_list()]
                else:
                    df_grouped = df[(df["state"] == input.us_state_indy_list()) & (
                        df["industries"] == input.us_state_indy_list_indy())]

                fig = px.bar(df_grouped.head(input.n_billionaires_us()),
                             x='personName',
                             y='finalWorth',
                             color='finalWorth',
                             color_continuous_scale='Blues')
                # fig.update_traces(marker_color=color())
                fig = style_plotly_chart(
                    fig, yaxis_title='Worth (million $)', xaxis_title=None)

                return fig

        with ui.nav_menu("Others"):
            with ui.nav_panel("Pie Chart"):

                ui.input_select(id="pie_gender_selfmade_us",
                                label="Select category",
                                choices={"gender": "Gender",
                                         "selfMade": "Self Made"},
                                selected=None,
                                multiple=False)

                @render_plotly
                def plot_pie_gender_selfmade_us():

                    df = data_usa().copy()

                    color_map_gender = {'M': 'darkblue', 'F': 'grey'}
                    color_map_self_made = {
                        True: 'darkblue', False: 'royalblue'}
                    cdmap = color_map_gender if input.pie_gender_selfmade_us(
                    ) == 'gender' else color_map_self_made

                    group = df.groupby(input.pie_gender_selfmade_us(
                    ), observed=False).size().reset_index(name='nBill')

                    fig = px.pie(group,
                                 values='nBill',
                                 names=input.pie_gender_selfmade_us(),
                                 color=input.pie_gender_selfmade_us(),
                                 title=f'Billionaires by {input.pie_gender_selfmade_us()}',
                                 color_discrete_map=cdmap)

                    fig.update_layout(
                        legend_title_text=input.pie_gender_selfmade_us())

                    return fig

            with ui.nav_control():
                ui.a("Shiny", href="https://shiny.posit.co", target="_blank")

    # Section 3 - Grouping by industry and states
    with ui.layout_column_wrap(width=(1/2)):
        with ui.navset_card_underline(id="bill_industry_us", header="Industry", footer=ui.input_numeric("number_indy_us", "Number of industries", 5, min=5, max=20)):
            with ui.nav_panel("Top"):
                "Top industries in the US with most billionaires"

                @render_plotly
                def plot_top_industry_us():
                    df = data_usa()

                    top_industry_us = df.groupby(
                        "industries").size().nlargest(input.number_indy_us()).reset_index(name='count')

                    fig = px.bar(top_industry_us.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="industries",
                                 color="count",
                                 color_continuous_scale="Blues")
                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Bottom"):
                "Industries with least billionaires"

                @render_plotly
                def plot_bottom_industry_us():
                    df = data_usa()

                    bottom_industry = df.groupby("industries").size().nsmallest(
                        input.number_indy_us()).reset_index(name='count')

                    fig = px.bar(bottom_industry.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="industries",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Top Mean Worth"):
                "Industries with most average billionaire wealth"

                @render_plotly
                def plot_top_industry_meanwealth_us():
                    df = data_usa()

                    top_industry_meanwealth_us = df.groupby("industries")['finalWorth'].mean().nlargest(
                        input.number_indy_us()).reset_index(name='mean_wealth')

                    fig = px.bar(top_industry_meanwealth_us.sort_values(by="mean_wealth", ascending=True),
                                 x="mean_wealth",
                                 y="industries",
                                 color="mean_wealth",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title="Average Wealth (Million $)")

                    return fig

            with ui.nav_menu("More"):
                with ui.nav_panel("Total Worth"):
                    "Industries with most billionaire wealth"

                    @render_plotly
                    def plot_top_industry_totalwealth_us():
                        df = data_usa()

                        top_industry_sumwealth_us = df.groupby("industries")['finalWorth'].sum().nlargest(
                            input.number_indy_us()).reset_index(name='total_wealth')

                        fig = px.bar(top_industry_sumwealth_us.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="industries",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title="Total Worth (Million $)")

                        return fig

                with ui.nav_panel("Industry pie chart"):

                    @render_plotly
                    def plot_pie_industry_us():

                        df = data_usa().copy()

                        group_us = df.groupby(
                            'industries', observed=False).size().reset_index(name='nBill')

                        fig = px.pie(group_us,
                                     values='nBill',
                                     names='industries',
                                     color='industries',
                                     title=f'Billionaires by industry',
                                     color_discrete_sequence=px.colors.qualitative.G10)  # plotly colors

                        fig.update_layout(
                            legend_title_text='Industry')

                        return fig

        with ui.navset_card_underline(id="bill_states_us", header="States", footer=ui.input_numeric("number_states_us", "Number of states", 5, min=5, max=20)):
            with ui.nav_panel("Top"):
                "States with most billionaires"

                @render_plotly
                def plot_top_states_us():
                    df = data_usa()

                    top_states_us = df.groupby("state").size().nlargest(
                        input.number_states_us()).reset_index(name='count')

                    fig = px.bar(top_states_us.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="state",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Bottom"):
                "States with least billionaires"

                @render_plotly
                def plot_bottom_states_us():
                    df = data_usa()

                    bottom_states_us = df.groupby("state").size().nsmallest(
                        input.number_states_us()).reset_index(name='count')

                    fig = px.bar(bottom_states_us.sort_values(by="count", ascending=True),
                                 x="count",
                                 y="state",
                                 color="count",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title=None)

                    return fig

            with ui.nav_panel("Top Mean Wealth"):
                "States most average billionaires wealth"

                @render_plotly
                def plot_top_states_meanwealth_us():
                    df = data_usa()

                    top_states_meanwealth_us = df.groupby("state")['finalWorth'].mean().nlargest(
                        input.number_states_us()).reset_index(name='mean_wealth')

                    fig = px.bar(top_states_meanwealth_us.sort_values(by="mean_wealth", ascending=True),
                                 x="mean_wealth",
                                 y="state",
                                 color="mean_wealth",
                                 color_continuous_scale="Blues")

                    fig = style_plotly_chart(
                        fig, yaxis_title=None, xaxis_title='Average Wealth (million $)')

                    return fig

            with ui.nav_menu("More"):
                with ui.nav_panel("Bottom Mean Wealth"):
                    "States least average billionaires wealth"

                    @render_plotly
                    def plot_bottom_states_meanwealth_us():
                        df = data_usa()

                        bottom_states_meanwealth_us = df.groupby("state")['finalWorth'].mean().nsmallest(
                            input.number_states_us()).reset_index(name='mean_wealth')

                        fig = px.bar(bottom_states_meanwealth_us.sort_values(by="mean_wealth", ascending=True),
                                     x="mean_wealth",
                                     y="state",
                                     color="mean_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Average Wealth (million $)')

                        return fig

                with ui.nav_panel("Top Total Wealth"):
                    "States most billionaires wealth combined"

                    @render_plotly
                    def plot_top_states_totalwealth_us():
                        df = data_usa()

                        top_states_totalwealth_us = df.groupby("state")['finalWorth'].sum().nlargest(
                            input.number_states_us()).reset_index(name='total_wealth')

                        fig = px.bar(top_states_totalwealth_us.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="state",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Total Wealth (million $)')

                        return fig

                with ui.nav_panel("Bottom Total Wealth"):
                    "States least average billionaires wealth"

                    @render_plotly
                    def plot_bottom_states_totalwealth_us():
                        df = data()

                        bottom_states_totalwealth_us = df.groupby("state")['finalWorth'].sum().nlargest(
                            input.number_ctr()).reset_index(name='total_wealth')

                        fig = px.bar(bottom_states_totalwealth_us.sort_values(by="total_wealth", ascending=True),
                                     x="total_wealth",
                                     y="state",
                                     color="total_wealth",
                                     color_continuous_scale="Blues")

                        fig = style_plotly_chart(
                            fig, yaxis_title=None, xaxis_title='Total Wealth (million $)')

                        return fig

    # Section 4 - Total Wealth across Industries by States
    with ui.card():

        ui.input_numeric("barh_state", "Number of states",
                         5, min=5, max=20)

        # ui.input_numeric("barh_indy", "Number of industries", 6, min=6, max=40)

        @render.plot
        def plot_ind_state_stacked_test():

            df = data_usa().copy()

            # Calculate total wealth by state
            total_wealth_by_state = df.groupby(
                'state')['finalWorth'].sum().reset_index()
            # Select the top n state by total wealth
            top_states = total_wealth_by_state.nlargest(
                input.barh_ctr(), 'finalWorth')['state']
            # Filter data for the top n state
            filtered_data = df[df['state'].isin(top_states)]
            # Group data by state and industry, calculate total wealth
            wealth_by_industry_state = filtered_data.groupby(['state', 'industries'])[
                'finalWorth'].sum().reset_index()
            # Pivot the data for easier plotting (industries as columns)
            wealth_pivot = wealth_by_industry_state.pivot(
                index='industries', columns='state', values='finalWorth').fillna(0)
            # Calculate total wealth per industry and sort industries in descending order
            wealth_pivot['TotalWealth'] = wealth_pivot.sum(
                axis=1)  # Add a column for total wealth
            wealth_pivot = wealth_pivot.sort_values(
                by='TotalWealth', ascending=True)  # Sort by total wealth
            # Drop the helper column after sorting
            wealth_pivot = wealth_pivot.drop(columns='TotalWealth')

            # Convert the pivot table to long format
            # Create the stacked bar chart
            # plt.figure(figsize=(15, 5))
            # plt.figure()
            fig = wealth_pivot.plot(kind='barh', stacked=True,
                                    figsize=(10, 5), cmap='tab10')

            # Customize the plot
            plt.title(
                'Accumulated Billionaire Wealth Across Industries for Top n States', fontsize=16, pad=20)
            plt.xlabel('Accumulated Wealth (million $)', fontsize=14)
            plt.ylabel('Industry', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.legend(title='State', bbox_to_anchor=(
                1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()

            return fig

    # Section 5 - Grouping by age and birth year range
    with ui.layout_column_wrap(width=(1/2)):
        with ui.navset_card_underline(id="age_dist_us",
                                      header="Age Distribution",
                                      footer=ui.input_select("gender_selfmade_us", "Select group", [
                                          'gender', 'selfMade'],
                                          multiple=False, selected=None)):

            with ui.nav_panel("All category"):

                @render_plotly
                def plot_agedist_all_usa():

                    df = data_usa()

                    # Should return the column name for selected category
                    hue = input.gender_selfmade_us()

                    fig = create_age_histogram(df, hue_column=hue)

                    return fig

            with ui.nav_panel("Industry"):
                "By Industry"

                ui.input_select("gender_selfmade_select_indy_us", "Select industry",
                                sorted(lists.INDUSTRY_LIST),
                                multiple=False,
                                selected=None)

                @render_plotly
                def plot_agedist_indy_usa():

                    df = data_usa()

                    df_industry = df[df['industries'] ==
                                     input.gender_selfmade_select_indy_us()]

                    # Should return the column name for selected category
                    hue = input.gender_selfmade_us()

                    fig = create_age_histogram(df_industry, hue_column=hue)

                    return fig

            with ui.nav_panel("State"):
                "By State"

                ui.input_select("gender_selfmade_select_state", "Select state",
                                sorted(lists.US_STATES_ALL),
                                multiple=False,
                                selected='United States')

                @render_plotly
                def plot_agedist_state_us():

                    df = data().copy()

                    df_state: pd.DataFrame

                    if input.gender_selfmade_select_state() == "All":
                        df_state = df
                    else:
                        df_state = df[df['state'] ==
                                      input.gender_selfmade_select_state()]

                    # Should return the column name for selected category
                    hue = input.gender_selfmade_us()

                    fig = create_age_histogram(df_state, hue_column=hue)

                    return fig

        with ui.navset_card_underline(id="birth_decade_dist_us",
                                      header="Distribution by birth year range",
                                      footer=ui.input_select("gender_selfmade_bdec_us",
                                                             "Select group",
                                                             ['gender',
                                                              'selfMade'],
                                                             multiple=False, selected=None)):

            with ui.nav_panel("Overall"):

                @render_plotly
                def birth_decade_overall_us():

                    df = data_usa().copy()

                    hue = input.gender_selfmade_bdec_us()

                    decade_grouped = df.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count').drop(index=df.loc[df["birthDecade"] == 0].index)

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

            with ui.nav_panel("Industry"):
                "Distribution of billionaires by birth year range and industry"

                ui.input_select("birth_decade_indy_us", "Select industry:",
                                sorted(lists.INDUSTRY_LIST),
                                multiple=False,
                                selected=None)

                @render_plotly
                def birth_decade_industry_us():

                    df = data_usa().copy()

                    df = df[df['industries'] == input.birth_decade_indy_us()]

                    hue = input.gender_selfmade_bdec_us()

                    decade_grouped = df.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count')

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

            with ui.nav_panel("State"):
                "Distribution of billionaires by birth year range and state"

                ui.input_select("birth_decade_state", "Select state:",
                                sorted(lists.US_STATES_ALL),
                                multiple=False,
                                selected='All')

                @render_plotly
                def birth_decade_state_us():

                    df = data_usa().copy()

                    df_state = pd.DataFrame

                    if input.birth_decade_state() == "All":
                        df_state = df
                    else:
                        df_state = df[df['state'] ==
                                      input.birth_decade_state()]

                    hue = input.gender_selfmade_bdec_us()

                    decade_grouped = df_state.groupby(['birthDecade', hue]).size(
                    ).reset_index(name='count')

                    fig = create_birthdecade_plot(
                        decade_grouped, hue_column=hue)

                    return fig

    # Section 6
    with ui.navset_card_underline(id="birth_months_us", footer=None):

        with ui.nav_panel("States"):
            ui.input_select("bmonth_category_us",
                            "Select group category:",
                            ['gender', 'selfMade'],
                            selected=None,
                            multiple=False)

            ui.input_select("bmonth_state",
                            "Select state:",
                            sorted(lists.US_STATES_ALL),
                            selected="All",
                            multiple=False)

            @render_plotly
            def plot_birthmonth_gender_us():

                hue_column = input.bmonth_category_us()

                # Data processing
                df = data_usa().copy()

                df_use: pd.DataFrame

                if input.bmonth_state() == 'All':
                    df_use = df
                else:
                    df_use = df[df['state'] == input.bmonth_state()]

                df_bmonth_category = df_use.groupby(['birthMonth', hue_column]).size(
                ).reset_index(name='count')

                # important
                rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                df_bmonth_category.drop(
                    index=rows_to_drop, inplace=True)  # important

                df_pivoted = df_bmonth_category.pivot(
                    index='birthMonth', columns=hue_column, values='count')

                birth_month_plot = create_birthmonth_plot(
                    df_pivoted, hue_column=hue_column, y_axis_title='Billionaires')

                return birth_month_plot

        with ui.nav_panel("Industry"):
            ui.input_select("bmonth_category_indy_us",
                            "Select group category:",
                            ['gender', 'selfMade'],
                            selected=None,
                            multiple=False)

            ui.input_select("bmonth_indy_us",
                            "Select industry:",
                            sorted(lists.INDUSTRY_LIST_ALL),
                            selected=None,
                            multiple=False)

            @render_plotly
            def plot_birthmonth_gender_indy_us():

                hue_column = input.bmonth_category_indy_us()

                # Data processing
                df = data_usa().copy()

                df_use: pd.DataFrame

                if input.bmonth_indy_us() == 'All':
                    df_use = df
                else:
                    df_use = df[df['industries'] == input.bmonth_indy_us()]

                df_bmonth_category = df_use.groupby(['birthMonth', hue_column]).size(
                ).reset_index(name='count')

                rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                df_bmonth_category.drop(index=rows_to_drop, inplace=True)

                df_pivoted = df_bmonth_category.pivot(
                    index='birthMonth', columns=hue_column, values='count')

                birth_month_plot = create_birthmonth_plot(
                    df_pivoted, hue_column=hue_column, y_axis_title='Billionaires')

                return birth_month_plot

        with ui.nav_menu("Other"):

            with ui.nav_panel("Total wealth by state"):
                ui.input_select("bmonth_category_state_total",
                                "Select group category:",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_state_total",
                                "Select state:",
                                sorted(lists.US_STATES_ALL),
                                selected="All",
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender_state_total_us():

                    hue_column = input.bmonth_category_state_total()

                    # Data processing
                    df = data_usa().copy()

                    df_use: pd.DataFrame

                    if input.bmonth_state_total() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['state'] ==
                                    input.bmonth_state_total()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].sum(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(
                        index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Total Billionaire Wealth')

                    return birth_month_plot

            with ui.nav_panel("Average wealth by state"):
                ui.input_select("bmonth_category_state_mean",
                                "Select group category:",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_state_mean",
                                "Select state:",
                                sorted(lists.US_STATES_ALL),
                                selected="All",
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender_state_mean():

                    hue_column = input.bmonth_category_state_mean()

                    # Data processing
                    df = data_usa().copy()

                    df_use: pd.DataFrame

                    if input.bmonth_state_mean() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['state'] ==
                                    input.bmonth_state_mean()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].mean(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(
                        index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Average Billionaire Wealth')

                    return birth_month_plot

            with ui.nav_panel("Total wealth by industry"):
                ui.input_select("bmonth_category_indy_total_us",
                                "Select group category:",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_indy_total_us",
                                "Select industry:",
                                sorted(lists.INDUSTRY_LIST_ALL),
                                selected='All',
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender_indy_total_us():

                    hue_column = input.bmonth_category_indy_total_us()

                    # Data processing
                    df = data_usa().copy()

                    df_use: pd.DataFrame

                    if input.bmonth_indy_total_us() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['industries'] ==
                                    input.bmonth_indy_total_us()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].sum(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(
                        index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Total Billionaire Wealth')

                    return birth_month_plot

            with ui.nav_panel("Average wealth by industry"):
                ui.input_select("bmonth_category_indy_mean_us",
                                "Select group category:",
                                ['gender', 'selfMade'],
                                selected=None,
                                multiple=False)

                ui.input_select("bmonth_indy_mean_us",
                                "Select industry:",
                                sorted(lists.INDUSTRY_LIST_ALL),
                                selected='All',
                                multiple=False)

                @render_plotly
                def plot_birthmonth_gender_indy_mean_us():

                    hue_column = input.bmonth_category_indy_mean_us()

                    # Data processing
                    df = data_usa().copy()

                    df_use: pd.DataFrame

                    if input.bmonth_indy_mean_us() == 'All':
                        df_use = df
                    else:
                        df_use = df[df['industries'] ==
                                    input.bmonth_indy_mean_us()]

                    df_bmonth_category = df_use.groupby(['birthMonth', hue_column])['finalWorth'].mean(
                    ).reset_index(name='count')

                    rows_to_drop = df_bmonth_category.loc[df_bmonth_category['birthMonth'] == 0].index

                    df_bmonth_category.drop(
                        index=rows_to_drop, inplace=True)

                    df_pivoted = df_bmonth_category.pivot(
                        index='birthMonth', columns=hue_column, values='count')

                    birth_month_plot = create_birthmonth_plot(
                        df_pivoted, hue_column=hue_column, y_axis_title='Average Billionaire Wealth')

                    return birth_month_plot

    # Section 7 - USA States maps
    with ui.navset_card_underline(id="chloropleth_map_us"):

        # Billionaires distribution map
        with ui.nav_panel("US Billionaires Distribution"):
            "Number of Billionaires"

            @render.ui
            def plot_bill_map_us():

                mapdata = united_states_map_data()

                bill_dist_map = plot_us_folium_map(mapdata,
                                                   "nBillionaires", "GnBu", "Number of Billionaires")

                return bill_dist_map

        # Total billionaires wealth by country map
        with ui.nav_panel("Total Wealth"):
            "Total billionaires wealth by state"

            @render.ui
            def plot_bill_map_total_us():

                mapdata = united_states_map_data()

                total_wealth_map = plot_us_folium_map(mapdata,
                                                      "billTotalWealth", "GnBu", "Total wealth ($ million)")

                return total_wealth_map

        # Average billionaires wealth by country map
        with ui.nav_panel("Average Wealth"):
            "Average billionaires wealth by state"

            @render.ui
            def plot_bill_map_avr_us():

                mapdata = united_states_map_data()

                avr_wealth_map = plot_us_folium_map(mapdata,
                                                    "billMeanWealth", "GnBu", "Average wealth ($ million)")

                return avr_wealth_map

        # Population by country map
        with ui.nav_panel("Population"):
            "Population by state"

            @render.ui
            def plot_pop_map_us():

                mapdata = united_states_map_data()

                pop_map = plot_us_folium_map(mapdata,
                                             "POPESTIMATE2023", "Reds", "Population")

                return pop_map

        # GDP by country map
        with ui.nav_panel("GDP"):
            "Gross Domestic Product by state"

            @render.ui
            def plot_gdp_map_us():

                mapdata = united_states_map_data()

                gdp_map = plot_us_folium_map(mapdata,
                                             "gdp_2023_q4", "Greens", "GDP")

                return gdp_map

    # Section 8 - Sample data (US)
    with ui.card():
        ui.card_header("Sample Data")

        ui.input_select("select_button_option_us",
                        "Option:",
                        ["Top 100 ", "Random 100", "Bottom 100"],
                        multiple=False,
                        selected="Top 100")

        # Define buttons within a horizontal container for proper display
        ui.input_action_button("show_dataframe_us", "Show data", width="15%")

        @render.data_frame
        @reactive.event(input.show_dataframe_us)  # Event listener (important)
        def show_data_us():

            df = data_usa().copy()

            if input.select_button_option_us() == "Top 100":
                df = df.head(100)
            elif input.select_button_option_us() == "Random 100":
                df = df.sample(100)
            elif input.select_button_option_us() == "Bottom 100":
                df = df.tail(100)

            # input.show_dataframe()
            return render.DataGrid(
                data=df,
                selection_mode='row',
                filters=True
            )

    # Section 9 - Panel footer (info)
    with ui.card_footer():
        @render.ui
        def panel_foote_us():
            return ui.markdown("Created by Robert Modalo")
