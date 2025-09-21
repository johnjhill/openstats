import dash
from dash import dcc, html, Input, Output, callback, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

from data_extractor import StatXploreAPI, collect_comprehensive_uc_data

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UCDashboard:
    def __init__(self, data_dir="uc_dashboard_data"):
        """
        Initialize the Universal Credit Dashboard

        Args:
            data_dir (str): Directory containing the UC data files
        """
        self.data_dir = data_dir
        self.data = None
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def load_data(self, force_refresh=False):
        """Load or refresh the UC data"""
        try:
            # Check for existing data files
            if os.path.exists(self.data_dir) and not force_refresh:
                # Look for the most recent combined parquet file
                parquet_files = [f for f in os.listdir(self.data_dir)
                               if f.startswith('uc_combined_') and f.endswith('.parquet')]

                if parquet_files:
                    # Get the most recent file
                    latest_file = max(parquet_files, key=lambda x: x.split('_')[2])
                    file_path = os.path.join(self.data_dir, latest_file)

                    logger.info(f"Loading existing data from {file_path}")
                    self.data = pd.read_parquet(file_path)
                    return True

            # If no data exists or force refresh, collect new data
            logger.info("Collecting fresh data from API...")
            api_key = os.getenv('STAT_XPLORE_API_KEY')

            if not api_key:
                logger.error("API key not found in environment variables")
                return False

            saved_files = collect_comprehensive_uc_data(api_key, self.data_dir)

            if 'combined_parquet' in saved_files:
                self.data = pd.read_parquet(saved_files['combined_parquet'])
                logger.info(f"Loaded fresh data: {len(self.data)} records")
                return True
            else:
                logger.error("Failed to collect data")
                return False

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False

    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Universal Credit Dashboard",
                       className="dashboard-title"),
                html.P("Comprehensive analysis of Universal Credit data across the UK",
                      className="dashboard-subtitle"),
            ], className="header"),

            # Data status indicator
            html.Div([
                html.Div(id="data-status", className="data-status"),
                html.Button("Refresh Data", id="refresh-btn",
                           className="refresh-button", n_clicks=0),
            ], className="controls-row"),

            # Main controls
            html.Div([
                html.Div([
                    html.Label("Dataset Type:"),
                    dcc.Dropdown(
                        id='dataset-dropdown',
                        options=[],
                        value=None,
                        className="dropdown"
                    )
                ], className="control-group"),

                html.Div([
                    html.Label("Geographic Level:"),
                    dcc.Dropdown(
                        id='geo-level-dropdown',
                        options=[
                            {'label': 'LSOA', 'value': 'lsoa'},
                            {'label': 'Local Authority', 'value': 'local_authority'},
                            {'label': 'Ward', 'value': 'ward'},
                            {'label': 'Region', 'value': 'region'}
                        ],
                        value='lsoa',
                        className="dropdown"
                    )
                ], className="control-group"),

                html.Div([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        start_date=None,
                        end_date=None,
                        className="date-picker"
                    )
                ], className="control-group"),
            ], className="controls"),

            # Main content tabs
            dcc.Tabs(id="main-tabs", value='overview', children=[
                dcc.Tab(label='Overview', value='overview'),
                dcc.Tab(label='Geographic Analysis', value='geographic'),
                dcc.Tab(label='Time Series', value='timeseries'),
                dcc.Tab(label='Data Explorer', value='explorer'),
            ], className="tabs"),

            # Content area
            html.Div(id="tab-content", className="tab-content"),

            # Footer
            html.Div([
                html.P("Data source: DWP Stat-Xplore | Last updated: ", id="last-updated"),
            ], className="footer")

        ], className="main-container")

    def setup_callbacks(self):
        """Setup all dashboard callbacks"""

        # Data loading and status
        @self.app.callback(
            [Output('data-status', 'children'),
             Output('dataset-dropdown', 'options'),
             Output('dataset-dropdown', 'value'),
             Output('date-range-picker', 'start_date'),
             Output('date-range-picker', 'end_date'),
             Output('last-updated', 'children')],
            [Input('refresh-btn', 'n_clicks')],
            prevent_initial_call=False
        )
        def update_data_status(n_clicks):
            # Load data (refresh if button clicked)
            force_refresh = n_clicks > 0
            success = self.load_data(force_refresh=force_refresh)

            if success and self.data is not None:
                # Extract available datasets
                datasets = self.data['Dataset'].unique() if 'Dataset' in self.data.columns else []
                dataset_options = [{'label': ds.replace('_', ' ').title(), 'value': ds}
                                 for ds in datasets]

                # Get date range
                if 'Date' in self.data.columns:
                    min_date = self.data['Date'].min()
                    max_date = self.data['Date'].max()
                else:
                    min_date = max_date = None

                status = html.Div([
                    html.Span("✅ Data loaded successfully", className="status-success"),
                    html.Span(f" | {len(self.data):,} records", className="status-info"),
                    html.Span(f" | {self.data['Geography_Code'].nunique():,} areas"
                             if 'Geography_Code' in self.data.columns else "",
                             className="status-info")
                ])

                last_updated = f"Data source: DWP Stat-Xplore | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"

                return (status, dataset_options, datasets[0] if datasets else None,
                       min_date, max_date, last_updated)
            else:
                status = html.Div([
                    html.Span("❌ Failed to load data", className="status-error"),
                    html.Span(" | Check API key in .env file", className="status-info")
                ])

                return (status, [], None, None, None,
                       "Data source: DWP Stat-Xplore | Last updated: Never")

        # Tab content switching
        @self.app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'value'),
             Input('dataset-dropdown', 'value'),
             Input('geo-level-dropdown', 'value'),
             Input('date-range-picker', 'start_date'),
             Input('date-range-picker', 'end_date')]
        )
        def update_tab_content(active_tab, dataset, geo_level, start_date, end_date):
            if self.data is None:
                return html.Div([
                    html.H3("No Data Available"),
                    html.P("Please ensure your API key is set in .env and click 'Refresh Data'")
                ])

            # Filter data based on selections
            filtered_data = self.filter_data(dataset, geo_level, start_date, end_date)

            if active_tab == 'overview':
                return self.create_overview_tab(filtered_data)
            elif active_tab == 'geographic':
                return self.create_geographic_tab(filtered_data)
            elif active_tab == 'timeseries':
                return self.create_timeseries_tab(filtered_data)
            elif active_tab == 'explorer':
                return self.create_explorer_tab(filtered_data)

            return html.Div("Tab content not implemented yet")

    def filter_data(self, dataset, geo_level, start_date, end_date):
        """Filter data based on user selections"""
        if self.data is None:
            return pd.DataFrame()

        filtered = self.data.copy()

        # Filter by dataset
        if dataset and 'Dataset' in filtered.columns:
            filtered = filtered[filtered['Dataset'] == dataset]

        # Filter by date range
        if start_date and end_date and 'Date' in filtered.columns:
            filtered = filtered[
                (filtered['Date'] >= start_date) &
                (filtered['Date'] <= end_date)
            ]

        return filtered

    def create_overview_tab(self, data):
        """Create the overview tab content"""
        if data.empty:
            return html.Div("No data available for selected filters")

        # Key metrics
        total_value = data['Value'].sum() if 'Value' in data.columns else 0
        unique_areas = data['Geography_Code'].nunique() if 'Geography_Code' in data.columns else 0
        date_range = f"{data['Time_Period'].min()} to {data['Time_Period'].max()}" if 'Time_Period' in data.columns else "N/A"

        # Top areas by value
        if 'Value' in data.columns and 'Geography_Name' in data.columns:
            top_areas = data.groupby('Geography_Name')['Value'].sum().nlargest(10).reset_index()
            top_areas_chart = px.bar(
                top_areas,
                x='Value',
                y='Geography_Name',
                orientation='h',
                title="Top 10 Areas by Universal Credit Count"
            )
            top_areas_chart.update_layout(height=400)
        else:
            top_areas_chart = {}

        # Trend over time
        if 'Date' in data.columns and 'Value' in data.columns:
            time_trend = data.groupby('Date')['Value'].sum().reset_index()
            trend_chart = px.line(
                time_trend,
                x='Date',
                y='Value',
                title="Universal Credit Trend Over Time"
            )
            trend_chart.update_layout(height=400)
        else:
            trend_chart = {}

        return html.Div([
            # Key metrics row
            html.Div([
                html.Div([
                    html.H3(f"{total_value:,}"),
                    html.P("Total UC Count")
                ], className="metric-card"),

                html.Div([
                    html.H3(f"{unique_areas:,}"),
                    html.P("Geographic Areas")
                ], className="metric-card"),

                html.Div([
                    html.H3(date_range),
                    html.P("Date Range")
                ], className="metric-card"),
            ], className="metrics-row"),

            # Charts row
            html.Div([
                html.Div([
                    dcc.Graph(figure=top_areas_chart)
                ], className="chart-half"),

                html.Div([
                    dcc.Graph(figure=trend_chart)
                ], className="chart-half"),
            ], className="charts-row")
        ])

    def create_geographic_tab(self, data):
        """Create the geographic analysis tab"""
        return html.Div([
            html.H3("Geographic Analysis"),
            html.P("Geographic mapping and analysis will be implemented here"),
            html.P("This will include choropleth maps and geographic comparisons")
        ])

    def create_timeseries_tab(self, data):
        """Create the time series analysis tab"""
        return html.Div([
            html.H3("Time Series Analysis"),
            html.P("Detailed time series analysis will be implemented here"),
            html.P("This will include seasonal analysis, trends, and forecasting")
        ])

    def create_explorer_tab(self, data):
        """Create the data explorer tab"""
        if data.empty:
            return html.Div("No data available")

        # Show data table
        return html.Div([
            html.H3("Data Explorer"),
            html.P(f"Showing {len(data):,} records"),

            dash_table.DataTable(
                data=data.head(1000).to_dict('records'),
                columns=[{"name": i, "id": i} for i in data.columns],
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
            )
        ])

    def run(self, debug=True, port=8050):
        """Run the dashboard"""
        self.app.run(debug=debug, port=port)

# CSS styling
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

if __name__ == '__main__':
    dashboard = UCDashboard()
    dashboard.run(debug=True)