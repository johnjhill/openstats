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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedUCDashboard:
    def __init__(self, data_dir="uc_dashboard_data"):
        """
        Advanced Universal Credit Dashboard with comprehensive analytics

        Args:
            data_dir (str): Directory containing the UC data files
        """
        self.data_dir = data_dir
        self.data = None
        self.app = dash.Dash(__name__, assets_folder='assets')
        self.app.title = "Universal Credit Analytics Dashboard"

        # Color scheme
        self.colors = {
            'primary': '#667eea',
            'secondary': '#764ba2',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'info': '#3498db',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }

        self.setup_layout()
        self.setup_callbacks()

    def load_data(self, force_refresh=False):
        """Load or refresh the UC data with enhanced error handling"""
        try:
            # Check for existing data files
            if os.path.exists(self.data_dir) and not force_refresh:
                parquet_files = [f for f in os.listdir(self.data_dir)
                               if f.startswith('uc_combined_') and f.endswith('.parquet')]

                if parquet_files:
                    latest_file = max(parquet_files, key=lambda x: x.split('_')[2])
                    file_path = os.path.join(self.data_dir, latest_file)

                    logger.info(f"Loading existing data from {file_path}")
                    self.data = pd.read_parquet(file_path)

                    # Enhanced data processing
                    self.data = self.process_data(self.data)
                    return True

            # Collect new data
            logger.info("Collecting fresh data from API...")
            api_key = os.getenv('STAT_XPLORE_API_KEY')

            if not api_key:
                logger.error("API key not found in environment variables")
                return False

            saved_files = collect_comprehensive_uc_data(api_key, self.data_dir)

            if 'combined_parquet' in saved_files:
                self.data = pd.read_parquet(saved_files['combined_parquet'])
                self.data = self.process_data(self.data)
                logger.info(f"Loaded fresh data: {len(self.data)} records")
                return True
            else:
                logger.error("Failed to collect data")
                return False

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def process_data(self, data):
        """Enhanced data processing and feature engineering"""
        if data is None or data.empty:
            return data

        # Ensure Date column exists and is properly formatted
        if 'Date' not in data.columns and 'Time_Period' in data.columns:
            try:
                data['Date'] = pd.to_datetime(data['Time_Period'], format='%Y %b', errors='coerce')
            except:
                data['Date'] = pd.to_datetime(data['Time_Period'], errors='coerce')

        # Add time-based features
        if 'Date' in data.columns:
            data['Year'] = data['Date'].dt.year
            data['Month'] = data['Date'].dt.month
            data['Quarter'] = data['Date'].dt.quarter
            data['Month_Name'] = data['Date'].dt.strftime('%B')

        # Add geographic region extraction (simplified)
        if 'Geography_Name' in data.columns:
            # Extract region from geography name (this would need proper geographic data)
            data['Region'] = data['Geography_Name'].str.extract(r'\b(North|South|East|West|Central|London)\b')
            data['Region'] = data['Region'].fillna('Other')

        # Calculate growth rates and trends
        if 'Value' in data.columns and 'Date' in data.columns:
            data = data.sort_values(['Geography_Code', 'Date'])
            data['Previous_Value'] = data.groupby('Geography_Code')['Value'].shift(1)
            data['Growth_Rate'] = ((data['Value'] - data['Previous_Value']) / data['Previous_Value'] * 100).round(2)
            data['Growth_Rate'] = data['Growth_Rate'].replace([np.inf, -np.inf], np.nan)

        return data

    def setup_layout(self):
        """Setup the enhanced dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H1("Universal Credit Analytics Dashboard",
                           className="dashboard-title"),
                    html.P("Comprehensive analysis of Universal Credit data across the UK",
                          className="dashboard-subtitle"),
                ], className="header-content"),

                # Quick stats overlay
                html.Div(id="quick-stats", className="quick-stats")
            ], className="header"),

            # Control panel
            html.Div([
                html.Div([
                    html.Div(id="data-status", className="data-status"),
                    html.Button("Refresh Data", id="refresh-btn",
                               className="refresh-button", n_clicks=0),
                ], className="status-controls"),

                html.Div([
                    html.Div([
                        html.Label("Dataset Type:", className="control-label"),
                        dcc.Dropdown(
                            id='dataset-dropdown',
                            options=[],
                            value=None,
                            className="control-dropdown"
                        )
                    ], className="control-item"),

                    html.Div([
                        html.Label("Geographic Level:", className="control-label"),
                        dcc.Dropdown(
                            id='geo-level-dropdown',
                            options=[
                                {'label': 'LSOA', 'value': 'lsoa'},
                                {'label': 'Local Authority', 'value': 'local_authority'},
                                {'label': 'Ward', 'value': 'ward'},
                                {'label': 'Region', 'value': 'region'}
                            ],
                            value='lsoa',
                            className="control-dropdown"
                        )
                    ], className="control-item"),

                    html.Div([
                        html.Label("Time Period:", className="control-label"),
                        dcc.Dropdown(
                            id='time-period-dropdown',
                            options=[
                                {'label': 'Last 12 Months', 'value': '12m'},
                                {'label': 'Last 24 Months', 'value': '24m'},
                                {'label': 'Last 5 Years', 'value': '5y'},
                                {'label': 'All Time', 'value': 'all'}
                            ],
                            value='all',
                            className="control-dropdown"
                        )
                    ], className="control-item"),

                    html.Div([
                        html.Label("Analysis Type:", className="control-label"),
                        dcc.Dropdown(
                            id='analysis-type-dropdown',
                            options=[
                                {'label': 'Absolute Values', 'value': 'absolute'},
                                {'label': 'Growth Rates', 'value': 'growth'},
                                {'label': 'Percentage Change', 'value': 'pct_change'}
                            ],
                            value='absolute',
                            className="control-dropdown"
                        )
                    ], className="control-item"),
                ], className="main-controls"),
            ], className="control-panel"),

            # Main content with advanced tabs
            dcc.Tabs(id="main-tabs", value='dashboard', className="main-tabs", children=[
                dcc.Tab(label='📊 Dashboard', value='dashboard', className="tab"),
                dcc.Tab(label='🗺️ Geographic', value='geographic', className="tab"),
                dcc.Tab(label='📈 Time Series', value='timeseries', className="tab"),
                dcc.Tab(label='🔍 Analytics', value='analytics', className="tab"),
                dcc.Tab(label='📋 Data Explorer', value='explorer', className="tab"),
            ]),

            # Content area
            html.Div(id="tab-content", className="tab-content"),

            # Footer
            html.Div([
                html.P([
                    "Data source: DWP Stat-Xplore | ",
                    html.Span(id="last-updated"),
                    " | Built with Dash & Plotly"
                ]),
            ], className="footer")

        ], className="main-container")

    def setup_callbacks(self):
        """Setup all dashboard callbacks"""

        # Data loading and initialization
        @self.app.callback(
            [Output('data-status', 'children'),
             Output('dataset-dropdown', 'options'),
             Output('dataset-dropdown', 'value'),
             Output('quick-stats', 'children'),
             Output('last-updated', 'children')],
            [Input('refresh-btn', 'n_clicks')],
            prevent_initial_call=False
        )
        def initialize_dashboard(n_clicks):
            force_refresh = n_clicks > 0
            success = self.load_data(force_refresh=force_refresh)

            if success and self.data is not None:
                # Dataset options
                datasets = self.data['Dataset'].unique() if 'Dataset' in self.data.columns else []
                dataset_options = [{'label': ds.replace('_', ' ').title(), 'value': ds}
                                 for ds in datasets]

                # Quick stats
                total_records = len(self.data)
                total_areas = self.data['Geography_Code'].nunique() if 'Geography_Code' in self.data.columns else 0
                total_value = self.data['Value'].sum() if 'Value' in self.data.columns else 0

                quick_stats = html.Div([
                    html.Div([
                        html.H3(f"{total_value:,.0f}"),
                        html.P("Total UC Count")
                    ], className="quick-stat"),
                    html.Div([
                        html.H3(f"{total_areas:,}"),
                        html.P("Geographic Areas")
                    ], className="quick-stat"),
                    html.Div([
                        html.H3(f"{total_records:,}"),
                        html.P("Data Points")
                    ], className="quick-stat"),
                ])

                status = html.Div([
                    html.Span("✅ ", className="status-icon"),
                    html.Span("Data loaded successfully", className="status-text"),
                ])

                last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                return (status, dataset_options, datasets[0] if datasets else None,
                       quick_stats, last_updated)
            else:
                status = html.Div([
                    html.Span("❌ ", className="status-icon"),
                    html.Span("Failed to load data - check API key", className="status-text"),
                ])

                empty_stats = html.Div("No data available", className="quick-stat")

                return (status, [], None, empty_stats, "Never")

        # Main content callback
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('dataset-dropdown', 'value'),
             Input('geo-level-dropdown', 'value'),
             Input('time-period-dropdown', 'value'),
             Input('analysis-type-dropdown', 'value')]
        )
        def update_content(active_tab, dataset, geo_level, time_period, analysis_type):
            if self.data is None:
                return self.create_no_data_message()

            # Filter data
            filtered_data = self.filter_data(dataset, geo_level, time_period)

            if active_tab == 'dashboard':
                return self.create_dashboard_tab(filtered_data, analysis_type)
            elif active_tab == 'geographic':
                return self.create_geographic_tab(filtered_data, analysis_type)
            elif active_tab == 'timeseries':
                return self.create_timeseries_tab(filtered_data, analysis_type)
            elif active_tab == 'analytics':
                return self.create_analytics_tab(filtered_data, analysis_type)
            elif active_tab == 'explorer':
                return self.create_explorer_tab(filtered_data)

            return html.Div("Content not available")

    def filter_data(self, dataset, geo_level, time_period):
        """Enhanced data filtering"""
        if self.data is None:
            return pd.DataFrame()

        filtered = self.data.copy()

        # Filter by dataset
        if dataset and 'Dataset' in filtered.columns:
            filtered = filtered[filtered['Dataset'] == dataset]

        # Filter by time period
        if time_period != 'all' and 'Date' in filtered.columns:
            end_date = filtered['Date'].max()
            if time_period == '12m':
                start_date = end_date - pd.DateOffset(months=12)
            elif time_period == '24m':
                start_date = end_date - pd.DateOffset(months=24)
            elif time_period == '5y':
                start_date = end_date - pd.DateOffset(years=5)
            else:
                start_date = filtered['Date'].min()

            filtered = filtered[filtered['Date'] >= start_date]

        return filtered

    def create_dashboard_tab(self, data, analysis_type):
        """Create comprehensive dashboard overview"""
        if data.empty:
            return self.create_no_data_message()

        # Key metrics calculation
        total_value = data['Value'].sum() if 'Value' in data.columns else 0
        unique_areas = data['Geography_Code'].nunique() if 'Geography_Code' in data.columns else 0

        # Growth calculation
        if 'Growth_Rate' in data.columns:
            avg_growth = data['Growth_Rate'].mean()
            growth_color = 'success' if avg_growth > 0 else 'danger'
        else:
            avg_growth = 0
            growth_color = 'info'

        # Top performers
        if 'Value' in data.columns and 'Geography_Name' in data.columns:
            top_areas = data.groupby('Geography_Name')['Value'].sum().nlargest(10).reset_index()

            top_chart = px.bar(
                top_areas,
                x='Value',
                y='Geography_Name',
                orientation='h',
                title="Top 10 Areas by Universal Credit Count",
                color='Value',
                color_continuous_scale='viridis'
            )
            top_chart.update_layout(
                height=400,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            top_chart = go.Figure()

        # Trend analysis
        if 'Date' in data.columns and 'Value' in data.columns:
            trend_data = data.groupby('Date')['Value'].sum().reset_index()

            trend_chart = px.line(
                trend_data,
                x='Date',
                y='Value',
                title="Universal Credit Trend Over Time",
                line_shape='spline'
            )
            trend_chart.update_traces(line_color=self.colors['primary'], line_width=3)
            trend_chart.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            trend_chart = go.Figure()

        # Distribution analysis
        if 'Value' in data.columns:
            dist_chart = px.histogram(
                data,
                x='Value',
                title="Distribution of UC Counts",
                nbins=30,
                color_discrete_sequence=[self.colors['secondary']]
            )
            dist_chart.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            dist_chart = go.Figure()

        return html.Div([
            # Metrics row
            html.Div([
                html.Div([
                    html.I(className="fas fa-users metric-icon"),
                    html.H3(f"{total_value:,.0f}"),
                    html.P("Total UC Count")
                ], className="metric-card primary"),

                html.Div([
                    html.I(className="fas fa-map-marker-alt metric-icon"),
                    html.H3(f"{unique_areas:,}"),
                    html.P("Geographic Areas")
                ], className="metric-card secondary"),

                html.Div([
                    html.I(className="fas fa-chart-line metric-icon"),
                    html.H3(f"{avg_growth:+.1f}%"),
                    html.P("Avg Growth Rate")
                ], className=f"metric-card {growth_color}"),

                html.Div([
                    html.I(className="fas fa-calendar metric-icon"),
                    html.H3(f"{data['Date'].nunique() if 'Date' in data.columns else 0}"),
                    html.P("Time Periods")
                ], className="metric-card info"),
            ], className="metrics-grid"),

            # Main charts
            html.Div([
                html.Div([
                    dcc.Graph(figure=trend_chart, className="chart")
                ], className="chart-container-half"),

                html.Div([
                    dcc.Graph(figure=top_chart, className="chart")
                ], className="chart-container-half"),
            ], className="charts-row"),

            # Additional analysis
            html.Div([
                html.Div([
                    dcc.Graph(figure=dist_chart, className="chart")
                ], className="chart-container-full"),
            ], className="charts-row"),
        ])

    def create_geographic_tab(self, data, analysis_type):
        """Create geographic analysis with choropleth mapping"""
        if data.empty:
            return self.create_no_data_message()

        # Geographic summary
        if 'Geography_Name' in data.columns and 'Value' in data.columns:
            geo_summary = data.groupby('Geography_Name').agg({
                'Value': ['sum', 'mean', 'count'],
                'Growth_Rate': 'mean' if 'Growth_Rate' in data.columns else lambda x: 0
            }).round(2)
            geo_summary.columns = ['Total', 'Average', 'Records', 'Avg_Growth']
            geo_summary = geo_summary.reset_index().head(20)

            # Geographic bar chart
            geo_chart = px.bar(
                geo_summary,
                x='Geography_Name',
                y='Total',
                title="Universal Credit by Geographic Area",
                color='Avg_Growth',
                color_continuous_scale='RdYlBu_r'
            )
            geo_chart.update_layout(
                height=500,
                xaxis_tickangle=-45,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )

            # Regional analysis
            if 'Region' in data.columns:
                regional_data = data.groupby('Region')['Value'].sum().reset_index()
                regional_chart = px.pie(
                    regional_data,
                    values='Value',
                    names='Region',
                    title="Regional Distribution of Universal Credit"
                )
                regional_chart.update_layout(height=400)
            else:
                regional_chart = go.Figure()

        else:
            geo_chart = go.Figure()
            regional_chart = go.Figure()
            geo_summary = pd.DataFrame()

        return html.Div([
            html.H2("Geographic Analysis", className="section-title"),

            html.Div([
                html.Div([
                    dcc.Graph(figure=geo_chart)
                ], className="chart-container-full"),
            ], className="charts-row"),

            html.Div([
                html.Div([
                    dcc.Graph(figure=regional_chart)
                ], className="chart-container-half"),

                html.Div([
                    html.H4("Geographic Summary"),
                    dash_table.DataTable(
                        data=geo_summary.to_dict('records') if not geo_summary.empty else [],
                        columns=[{"name": i, "id": i} for i in geo_summary.columns] if not geo_summary.empty else [],
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        style_cell={'textAlign': 'left'},
                        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                    )
                ], className="chart-container-half"),
            ], className="charts-row"),
        ])

    def create_timeseries_tab(self, data, analysis_type):
        """Create detailed time series analysis"""
        if data.empty or 'Date' not in data.columns:
            return self.create_no_data_message()

        # Monthly trend
        monthly_data = data.groupby('Date')['Value'].sum().reset_index()

        # Seasonal analysis
        if 'Month' in data.columns:
            seasonal_data = data.groupby('Month')['Value'].mean().reset_index()
            seasonal_data['Month_Name'] = seasonal_data['Month'].map({
                1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
            })

            seasonal_chart = px.bar(
                seasonal_data,
                x='Month_Name',
                y='Value',
                title="Seasonal Pattern (Average by Month)",
                color='Value',
                color_continuous_scale='viridis'
            )
        else:
            seasonal_chart = go.Figure()

        # Main trend chart with moving average
        if len(monthly_data) > 3:
            monthly_data['MA_3'] = monthly_data['Value'].rolling(window=3).mean()
            monthly_data['MA_6'] = monthly_data['Value'].rolling(window=6).mean()

            trend_chart = go.Figure()
            trend_chart.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['Value'],
                mode='lines+markers',
                name='Actual',
                line=dict(color=self.colors['primary'], width=2)
            ))
            trend_chart.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['MA_3'],
                mode='lines',
                name='3-Month MA',
                line=dict(color=self.colors['secondary'], width=2, dash='dash')
            ))
            trend_chart.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data['MA_6'],
                mode='lines',
                name='6-Month MA',
                line=dict(color=self.colors['info'], width=2, dash='dot')
            ))
            trend_chart.update_layout(
                title="Universal Credit Trend with Moving Averages",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
        else:
            trend_chart = go.Figure()

        return html.Div([
            html.H2("Time Series Analysis", className="section-title"),

            html.Div([
                html.Div([
                    dcc.Graph(figure=trend_chart)
                ], className="chart-container-full"),
            ], className="charts-row"),

            html.Div([
                html.Div([
                    dcc.Graph(figure=seasonal_chart)
                ], className="chart-container-full"),
            ], className="charts-row"),
        ])

    def create_analytics_tab(self, data, analysis_type):
        """Create advanced analytics and insights"""
        return html.Div([
            html.H2("Advanced Analytics", className="section-title"),
            html.Div([
                html.H4("Statistical Analysis"),
                html.P("Advanced analytics features will be implemented here including:"),
                html.Ul([
                    html.Li("Correlation analysis between different datasets"),
                    html.Li("Forecasting and trend prediction"),
                    html.Li("Anomaly detection"),
                    html.Li("Comparative analysis"),
                    html.Li("Statistical summaries and insights")
                ])
            ], className="analytics-placeholder")
        ])

    def create_explorer_tab(self, data):
        """Create data explorer with enhanced filtering"""
        if data.empty:
            return self.create_no_data_message()

        return html.Div([
            html.H2("Data Explorer", className="section-title"),
            html.P(f"Showing {len(data):,} records"),

            dash_table.DataTable(
                data=data.head(1000).to_dict('records'),
                columns=[{"name": i, "id": i, "type": "numeric" if data[i].dtype in ['int64', 'float64'] else "text"}
                        for i in data.columns],
                page_size=25,
                sort_action="native",
                filter_action="native",
                export_action="native",
                export_format="xlsx",
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'Arial, sans-serif'
                },
                style_header={
                    'backgroundColor': self.colors['primary'],
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': 'rgb(248, 248, 248)'
                    }
                ]
            )
        ])

    def create_no_data_message(self):
        """Create a message for when no data is available"""
        return html.Div([
            html.Div([
                html.I(className="fas fa-exclamation-triangle", style={'fontSize': '48px', 'color': '#f39c12'}),
                html.H3("No Data Available"),
                html.P("Please ensure your API key is set in the .env file and click 'Refresh Data'"),
                html.P("API key should be set as: STAT_XPLORE_API_KEY=your_actual_key")
            ], className="no-data-message")
        ])

    def run(self, debug=True, port=8050, host='127.0.0.1'):
        """Run the dashboard"""
        print(f"🚀 Starting Universal Credit Dashboard...")
        print(f"📊 Dashboard will be available at: http://{host}:{port}")
        print(f"🔑 Make sure your API key is set in .env file as: STAT_XPLORE_API_KEY=your_key")

        self.app.run(debug=debug, port=port, host=host)

if __name__ == '__main__':
    dashboard = AdvancedUCDashboard()
    dashboard.run(debug=True)