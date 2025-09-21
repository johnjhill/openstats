"""
UX Components for Universal Credit Dashboard
Enhanced educational and accessibility components
"""

from dash import html, dcc
import plotly.graph_objects as go

def create_uc_education_module():
    """Create an educational module explaining Universal Credit"""
    return html.Div([
        html.Div([
            html.H3("📚 What is Universal Credit?", className="education-title"),
            html.P([
                "Universal Credit (UC) is the UK government's benefit system that helps with living costs ",
                "for people who are on a low income or out of work. It combines six previous benefits into one payment."
            ], className="education-text"),

            html.Div([
                html.H4("Key Information:", className="education-subtitle"),
                html.Ul([
                    html.Li("Introduced in 2013, replacing legacy benefits like Job Seekers Allowance"),
                    html.Li("Paid monthly to help with housing, childcare, and living costs"),
                    html.Li("Amount depends on circumstances like age, housing costs, and income"),
                    html.Li("Currently supports over 6 million households across the UK")
                ], className="education-list")
            ]),

            html.Details([
                html.Summary("🔍 Data Methodology", className="methodology-toggle"),
                html.Div([
                    html.P("This dashboard uses official data from the DWP Stat-Xplore database:"),
                    html.Ul([
                        html.Li("Data updated monthly with official government statistics"),
                        html.Li("Geographic breakdowns available from postcode to national level"),
                        html.Li("Demographic data includes age, gender, and employment status"),
                        html.Li("All figures represent official claimant counts, not estimated populations")
                    ])
                ], className="methodology-content")
            ], className="methodology-section")
        ], className="education-module")
    ], className="education-container")

def create_contextual_tooltip(text, tooltip_text):
    """Create a tooltip for contextual help"""
    return html.Span([
        text,
        html.Span(" ⓘ", className="tooltip-icon"),
        html.Div(tooltip_text, className="tooltip-content")
    ], className="tooltip-container")

def create_insight_card(title, value, change, interpretation, icon="📊"):
    """Create an insight card with clear interpretation"""
    change_class = "positive" if change >= 0 else "negative"
    change_symbol = "+" if change >= 0 else ""

    return html.Div([
        html.Div([
            html.Span(icon, className="insight-icon"),
            html.H4(title, className="insight-title")
        ], className="insight-header"),

        html.Div([
            html.H2(f"{value:,.0f}" if isinstance(value, (int, float)) else str(value),
                   className="insight-value"),
            html.P(f"{change_symbol}{change:.1f}%" if isinstance(change, (int, float)) else str(change),
                  className=f"insight-change {change_class}")
        ], className="insight-stats"),

        html.P(interpretation, className="insight-interpretation")
    ], className="insight-card")

def create_progressive_help_system():
    """Create a progressive help system for new users"""
    return html.Div([
        html.Div([
            html.H4("🚀 Getting Started", className="help-title"),
            html.Div([
                html.Div([
                    html.Span("1", className="step-number"),
                    html.P("Start with the Dashboard tab for key insights")
                ], className="help-step"),
                html.Div([
                    html.Span("2", className="step-number"),
                    html.P("Use filters to explore specific time periods or regions")
                ], className="help-step"),
                html.Div([
                    html.Span("3", className="step-number"),
                    html.P("Visit Geographic tab to see regional patterns")
                ], className="help-step"),
                html.Div([
                    html.Span("4", className="step-number"),
                    html.P("Check Time Series for trends over time")
                ], className="help-step")
            ], className="help-steps")
        ], className="help-content"),

        html.Button("✕", id="close-help", className="close-help-btn")
    ], id="progressive-help", className="progressive-help")

def create_data_story_narrative(data_insights):
    """Create a narrative explanation of the data"""
    return html.Div([
        html.H3("📖 Data Story", className="narrative-title"),
        html.Div([
            html.P(f"Based on the latest data, Universal Credit currently supports approximately "
                   f"{data_insights.get('total_beneficiaries', 'N/A'):,.0f} people across the UK.",
                   className="narrative-text"),

            html.P(f"The trend shows {data_insights.get('trend_direction', 'stable')} patterns over recent months, "
                   f"with {data_insights.get('geographic_focus', 'national')} variations in uptake.",
                   className="narrative-text"),

            html.P("This data helps policymakers understand where support is needed most and "
                   "how the benefit system is performing across different communities.",
                   className="narrative-text")
        ], className="narrative-content")
    ], className="narrative-container")

def create_accessibility_features():
    """Create accessibility enhancement features"""
    return html.Div([
        html.Button("🔍 High Contrast", id="high-contrast-toggle", className="a11y-button"),
        html.Button("🔤 Large Text", id="large-text-toggle", className="a11y-button"),
        html.Button("⏸️ Reduce Motion", id="reduce-motion-toggle", className="a11y-button")
    ], className="accessibility-toolbar")

def create_enhanced_chart_title(title, subtitle, help_text):
    """Create enhanced chart titles with context"""
    return html.Div([
        html.H3(title, className="chart-title"),
        html.P(subtitle, className="chart-subtitle"),
        create_contextual_tooltip("", help_text)
    ], className="enhanced-chart-header")

def create_data_quality_indicator(data_quality_score, last_updated):
    """Create data quality and freshness indicators"""
    quality_class = "high" if data_quality_score >= 0.9 else "medium" if data_quality_score >= 0.7 else "low"

    return html.Div([
        html.Div([
            html.Span("📊", className="quality-icon"),
            html.Span(f"Data Quality: {quality_class.title()}", className=f"quality-text {quality_class}")
        ], className="quality-indicator"),

        html.Div([
            html.Span("🕒", className="freshness-icon"),
            html.Span(f"Last Updated: {last_updated}", className="freshness-text")
        ], className="freshness-indicator")
    ], className="data-quality-bar")

# WCAG compliant color scheme
ACCESSIBLE_COLORS = {
    'primary': '#1f4e79',      # Dark blue (WCAG AA compliant)
    'secondary': '#5a6c7d',    # Medium blue-gray
    'success': '#2d5a27',      # Dark green
    'warning': '#8b4513',      # Dark orange
    'danger': '#8b0000',       # Dark red
    'info': '#2f4f4f',         # Dark slate gray
    'light': '#f8f9fa',        # Light gray
    'dark': '#212529',         # Dark gray
    'accent': '#6c757d',       # Medium gray
    'background': '#ffffff',   # White background
    'text': '#212529',         # Dark text
    'text_secondary': '#6c757d' # Secondary text
}

def get_accessible_color_palette():
    """Return WCAG AA compliant color palette for charts"""
    return [
        '#1f4e79',  # Dark blue
        '#2d5a27',  # Dark green
        '#8b4513',  # Dark orange
        '#5a6c7d',  # Blue-gray
        '#8b0000',  # Dark red
        '#2f4f4f',  # Dark slate
        '#483d8b',  # Dark slate blue
        '#8b4513'   # Saddle brown
    ]