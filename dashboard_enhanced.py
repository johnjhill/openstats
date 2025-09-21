import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
import json

from data_extractor import StatXploreAPI, collect_comprehensive_uc_data
from ux_components import (
    create_uc_education_module,
    create_contextual_tooltip,
    create_insight_card,
    create_data_story_narrative,
    create_accessibility_features,
    create_enhanced_chart_title,
    ACCESSIBLE_COLORS,
    get_accessible_color_palette
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedUCDashboard:
    def __init__(self, data_dir="dashboard_ready"):
        """
        Enhanced Universal Credit Dashboard with educational content and accessibility features
        """
        self.data_dir = data_dir
        self.data = None
        self.app = dash.Dash(__name__, assets_folder='assets')
        self.app.title = "Universal Credit Analytics Dashboard - UK Government Data"

        # Use accessible color scheme
        self.colors = ACCESSIBLE_COLORS

        self.setup_layout()
        self.setup_callbacks()

    def load_data(self, force_refresh=False):
        """Load dashboard-ready data with better error handling and user feedback"""
        try:
            # Check for dashboard-ready data first
            if os.path.exists(self.data_dir) and not force_refresh:
                parquet_files = [f for f in os.listdir(self.data_dir)
                               if f.startswith('uc_dashboard_ready_') and f.endswith('.parquet')]

                if parquet_files:
                    latest_file = max(parquet_files, key=lambda x: x.split('_')[3])
                    file_path = os.path.join(self.data_dir, latest_file)

                    logger.info(f"Loading dashboard-ready data from {file_path}")
                    self.data = pd.read_parquet(file_path)
                    return True

            # Fallback to existing data directory
            fallback_dir = "uc_dashboard_data"
            if os.path.exists(fallback_dir):
                parquet_files = [f for f in os.listdir(fallback_dir)
                               if f.endswith('.parquet')]

                if parquet_files:
                    latest_file = max(parquet_files, key=lambda x: os.path.getmtime(os.path.join(fallback_dir, x)))
                    file_path = os.path.join(fallback_dir, latest_file)

                    logger.info(f"Loading fallback data from {file_path}")
                    self.data = pd.read_parquet(file_path)
                    self.data = self.process_data(self.data)
                    return True

            logger.error("No data files found")
            return False

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def process_data(self, data):
        """Enhanced data processing with better insights"""
        if data is None or data.empty:
            return data

        # Ensure essential columns exist
        if 'Date' not in data.columns and 'Time_Period' in data.columns:
            try:
                data['Date'] = pd.to_datetime(data['Time_Period'], format='%Y %b', errors='coerce')
            except:
                data['Date'] = pd.to_datetime(data['Time_Period'], errors='coerce')

        # Enhanced processing for better insights
        if 'Value' in data.columns:
            data = data.sort_values(['Geography_Code', 'Date'])
            data['Previous_Value'] = data.groupby(['Geography_Code', 'Dataset'])['Value'].shift(1)
            data['Growth_Rate'] = ((data['Value'] - data['Previous_Value']) / data['Previous_Value'] * 100).round(2)
            data['Growth_Rate'] = data['Growth_Rate'].replace([np.inf, -np.inf], np.nan)

            # Add value formatting helpers
            data['Value_Millions'] = (data['Value'] / 1_000_000).round(2)
            data['Value_Thousands'] = (data['Value'] / 1_000).round(1)

        return data

    def calculate_insights(self):
        """Calculate key insights from the data"""
        if self.data is None or self.data.empty:
            return {}

        insights = {}

        # Current total beneficiaries
        latest_data = self.data[self.data['Date'] == self.data['Date'].max()]
        insights['total_beneficiaries'] = latest_data['Value'].sum()

        # Growth trend
        if 'Growth_Rate' in self.data.columns:
            recent_growth = self.data['Growth_Rate'].dropna().tail(12).mean()
            insights['avg_growth_rate'] = recent_growth
            insights['trend_direction'] = 'increasing' if recent_growth > 0 else 'decreasing'

        # Geographic coverage
        insights['geographic_areas'] = self.data['Geography_Code'].nunique()

        # Data freshness
        insights['latest_date'] = self.data['Date'].max()
        insights['data_span'] = f"{self.data['Date'].min().strftime('%Y')} - {self.data['Date'].max().strftime('%Y')}"

        return insights

    def setup_layout(self):
        """Setup enhanced layout with educational content"""
        self.app.layout = html.Div([
            # Accessibility toolbar
            create_accessibility_features(),

            # Educational header
            create_uc_education_module(),

            # Main header with improved accessibility
            html.Div([
                html.Div([
                    html.H1("Universal Credit Analytics Dashboard",
                           className="dashboard-title"),
                    create_contextual_tooltip(
                        "Comprehensive analysis of Universal Credit data across the UK",
                        "This dashboard shows official government statistics about Universal Credit claimants, including trends over time and geographic distribution."
                    )
                ], className="header-content"),

                # Enhanced quick stats with context
                html.Div(id="enhanced-quick-stats", className="quick-stats")
            ], className="header"),

            # Data quality indicator
            html.Div(id="data-quality-indicator"),

            # Enhanced control panel with help
            html.Div([
                html.Div([
                    html.Div(id="data-status", className="data-status"),
                    html.Button("🔄 Refresh Data", id="refresh-btn",
                               className="refresh-button", n_clicks=0),
                ], className="status-controls"),

                html.Div([
                    html.Div([
                        html.Label("Dataset Type:", className="control-label"),
                        create_contextual_tooltip("", "Choose between household counts, individual claimants, or demographic breakdowns"),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[],
                            value=None,
                            className="control-dropdown"
                        )
                    ], className="control-item"),

                    html.Div([
                        html.Label("Time Period:", className="control-label"),
                        create_contextual_tooltip("", "Filter data by time range to focus on recent trends or historical patterns"),
                        dcc.Dropdown(
                            id='time-period-dropdown',
                            options=[
                                {'label': 'Last 12 Months', 'value': '12m'},
                                {'label': 'Last 24 Months', 'value': '24m'},
                                {'label': 'Last 5 Years', 'value': '5y'},
                                {'label': 'All Available Data', 'value': 'all'}
                            ],
                            value='all',
                            className="control-dropdown"
                        )
                    ], className="control-item"),

                    html.Div([
                        html.Label("Analysis Type:", className="control-label"),
                        create_contextual_tooltip("", "View absolute numbers, percentage changes, or growth rates over time"),
                        dcc.Dropdown(
                            id='analysis-type-dropdown',
                            options=[
                                {'label': 'Total Numbers', 'value': 'absolute'},
                                {'label': 'Growth Rates (%)', 'value': 'growth'},
                                {'label': 'Month-on-Month Change', 'value': 'change'}
                            ],
                            value='absolute',
                            className="control-dropdown"
                        )
                    ], className="control-item"),
                ], className="main-controls"),
            ], className="control-panel"),

            # Enhanced tabs with better descriptions
            dcc.Tabs(id="main-tabs", value='dashboard', className="main-tabs", children=[
                dcc.Tab(label='📊 Overview & Insights', value='dashboard', className="tab"),
                dcc.Tab(label='📈 Time Trends', value='timeseries', className="tab"),
                dcc.Tab(label='🔍 Detailed Analytics', value='analytics', className="tab"),
                dcc.Tab(label='📋 Data Explorer', value='explorer', className="tab"),
            ]),

            # Content area
            html.Div(id="tab-content", className="tab-content"),

            # Enhanced footer with methodology
            html.Div([
                html.P([
                    "📊 Data source: ",
                    html.A("DWP Stat-Xplore", href="https://stat-xplore.dwp.gov.uk/", target="_blank"),
                    " | ",
                    html.Span(id="last-updated"),
                    " | Built with accessibility in mind"
                ]),
                html.Details([
                    html.Summary("ℹ️ About this data"),
                    html.P("This dashboard uses official statistics from the Department for Work and Pensions. "
                           "Data is updated monthly and includes all active Universal Credit claims. "
                           "Geographic breakdowns follow ONS standard area classifications.")
                ])
            ], className="footer")

        ], className="main-container")

    def setup_callbacks(self):
        """Setup enhanced callbacks with better user feedback"""

        # Data loading and initialization
        @self.app.callback(
            [Output('data-status', 'children'),
             Output('dataset-dropdown', 'options'),
             Output('dataset-dropdown', 'value'),
             Output('enhanced-quick-stats', 'children'),
             Output('last-updated', 'children'),
             Output('data-quality-indicator', 'children')],
            [Input('refresh-btn', 'n_clicks')],
            prevent_initial_call=False
        )
        def initialize_dashboard(n_clicks):
            force_refresh = n_clicks > 0
            success = self.load_data(force_refresh=force_refresh)

            if success and self.data is not None:
                # Calculate insights
                insights = self.calculate_insights()

                # Dataset options with friendly names
                datasets = self.data['Dataset'].unique() if 'Dataset' in self.data.columns else []
                dataset_options = []
                for ds in datasets:
                    if 'household' in ds.lower():
                        label = "🏠 Households on Universal Credit"
                    elif 'people' in ds.lower():
                        label = "👥 People on Universal Credit"
                    elif 'demographic' in ds.lower():
                        label = "📊 Demographic Breakdown"
                    else:
                        label = ds.replace('_', ' ').title()
                    dataset_options.append({'label': label, 'value': ds})

                # Enhanced quick stats with insights
                quick_stats = html.Div([
                    create_insight_card(
                        "Current Universal Credit Recipients",
                        insights.get('total_beneficiaries', 0),
                        insights.get('avg_growth_rate', 0),
                        f"Based on latest available data from {insights.get('latest_date', 'N/A').strftime('%B %Y') if insights.get('latest_date') else 'N/A'}",
                        "👥"
                    ),
                    create_data_story_narrative(insights)
                ])

                # Data quality indicator
                data_quality = html.Div([
                    html.Div([
                        html.Span("✅ ", className="quality-icon"),
                        html.Span("Official Government Data", className="quality-text high")
                    ], className="quality-indicator"),
                    html.Div([
                        html.Span("🕒 ", className="freshness-icon"),
                        html.Span(f"Data span: {insights.get('data_span', 'N/A')}", className="freshness-text")
                    ], className="freshness-indicator")
                ], className="data-quality-bar")

                return (
                    html.Span("✅ Data loaded successfully", style={'color': self.colors['success']}),
                    dataset_options,
                    datasets[0] if datasets else None,
                    quick_stats,
                    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    data_quality
                )
            else:
                return (
                    html.Span("❌ Failed to load data", style={'color': self.colors['danger']}),
                    [],
                    None,
                    html.Div("No data available", className="alert alert-warning"),
                    "Data unavailable",
                    html.Div()
                )

        # Main content callback
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('dataset-dropdown', 'value'),
             Input('time-period-dropdown', 'value'),
             Input('analysis-type-dropdown', 'value')],
            prevent_initial_call=True
        )
        def update_tab_content(active_tab, selected_dataset, time_period, analysis_type):
            if self.data is None or self.data.empty:
                return html.Div("No data available for analysis.", className="alert alert-warning")

            # Filter data based on selections
            filtered_data = self.data.copy()

            if selected_dataset:
                filtered_data = filtered_data[filtered_data['Dataset'] == selected_dataset]

            # Time filtering
            if time_period != 'all':
                months_back = {'12m': 12, '24m': 24, '5y': 60}[time_period]
                cutoff_date = filtered_data['Date'].max() - pd.DateOffset(months=months_back)
                filtered_data = filtered_data[filtered_data['Date'] >= cutoff_date]

            if active_tab == 'dashboard':
                return self.create_dashboard_tab(filtered_data, analysis_type)
            elif active_tab == 'timeseries':
                return self.create_timeseries_tab(filtered_data, analysis_type)
            elif active_tab == 'analytics':
                return self.create_analytics_tab(filtered_data, analysis_type)
            elif active_tab == 'explorer':
                return self.create_explorer_tab(filtered_data)

    def create_dashboard_tab(self, data, analysis_type):
        """Create enhanced dashboard tab with insights and education"""
        if data.empty:
            return html.Div("No data available for the selected filters.", className="alert alert-info")

        # Key metrics
        latest_value = data[data['Date'] == data['Date'].max()]['Value'].sum()
        avg_growth = data['Growth_Rate'].dropna().tail(12).mean() if 'Growth_Rate' in data.columns else 0

        # Main chart
        fig = px.line(
            data.groupby('Date')['Value'].sum().reset_index(),
            x='Date', y='Value',
            title="Universal Credit Claims Over Time",
            color_discrete_sequence=get_accessible_color_palette()
        )

        fig.update_layout(
            title_font_size=16,
            title_font_color=self.colors['primary'],
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text']
        )

        return html.Div([
            create_enhanced_chart_title(
                "Universal Credit Trends Overview",
                "Shows the total number of Universal Credit recipients over time",
                "This chart displays the cumulative number of people or households receiving Universal Credit payments. Upward trends indicate growing demand for support."
            ),

            dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False}),

            html.Div([
                create_insight_card(
                    "Current Total",
                    latest_value,
                    avg_growth,
                    "Total Universal Credit recipients based on latest available data. This represents real people and families receiving government support.",
                    "📊"
                ),
                create_insight_card(
                    "12-Month Trend",
                    f"{avg_growth:.1f}%",
                    0,
                    "Average monthly growth rate over the past year. Positive values indicate increasing demand for Universal Credit support.",
                    "📈"
                )
            ], style={'display': 'grid', 'grid-template-columns': '1fr 1fr', 'gap': '20px'})
        ])

    def create_timeseries_tab(self, data, analysis_type):
        """Create enhanced time series analysis"""
        if data.empty:
            return html.Div("No data available for time series analysis.", className="alert alert-info")

        # Group by date and dataset
        time_data = data.groupby(['Date', 'Dataset'])['Value'].sum().reset_index()

        fig = px.line(
            time_data, x='Date', y='Value', color='Dataset',
            title="Universal Credit Trends by Dataset Type",
            color_discrete_sequence=get_accessible_color_palette()
        )

        fig.update_layout(
            title_font_size=16,
            title_font_color=self.colors['primary'],
            plot_bgcolor='white',
            paper_bgcolor='white',
            font_color=self.colors['text'],
            legend_title_text="Dataset Type"
        )

        return html.Div([
            create_enhanced_chart_title(
                "Time Series Analysis",
                "Detailed trends showing how Universal Credit usage has changed over time",
                "This analysis helps identify seasonal patterns, policy impacts, and long-term trends in Universal Credit uptake."
            ),
            dcc.Graph(figure=fig, config={'displayModeBar': True, 'displaylogo': False})
        ])

    def create_analytics_tab(self, data, analysis_type):
        """Create advanced analytics tab"""
        if data.empty:
            return html.Div("No analytics available for the selected data.", className="alert alert-info")

        return html.Div([
            html.H3("Advanced Analytics", className="chart-title"),
            html.P("Detailed analytical views will be available here.", className="chart-subtitle"),
            html.Div("Analytics features coming soon...", className="alert alert-info")
        ])

    def create_explorer_tab(self, data):
        """Create data explorer tab"""
        if data.empty:
            return html.Div("No data available to explore.", className="alert alert-info")

        # Sample of data for exploration
        sample_data = data.head(100)

        return html.Div([
            create_enhanced_chart_title(
                "Data Explorer",
                "Browse and export the underlying government data",
                "This table shows a sample of the raw data. Use the controls above to filter the dataset, then download for your own analysis."
            ),

            dash_table.DataTable(
                data=sample_data.to_dict('records'),
                columns=[{"name": i, "id": i} for i in sample_data.columns],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': self.colors['primary'], 'color': 'white'},
                style_data={'backgroundColor': 'white'},
                export_format="csv"
            )
        ])

def main():
    """Run the enhanced dashboard"""
    dashboard = EnhancedUCDashboard()
    dashboard.app.run(debug=True, port=8051)

if __name__ == '__main__':
    main()