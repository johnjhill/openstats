#!/usr/bin/env python3
"""
Launch script for Universal Credit Dashboard

This script provides an easy way to start the dashboard with options
for data collection and different dashboard modes.
"""

import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_requirements():
    """Check if all required packages are installed"""
    # Package name -> import name mapping
    package_imports = {
        'dash': 'dash',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'python-dotenv': 'dotenv',
        'pyarrow': 'pyarrow'
    }

    missing_packages = []
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True

def check_api_key():
    """Check if API key is configured"""
    api_key = os.getenv('STAT_XPLORE_API_KEY')

    if not api_key:
        print("❌ API key not found!")
        print("\n💡 Please create a .env file with your API key:")
        print("   STAT_XPLORE_API_KEY=your_actual_api_key")
        print("\n🔑 Get your API key from: https://stat-xplore.dwp.gov.uk")
        return False

    print(f"✅ API key configured (ending in ...{api_key[-4:]})")
    return True

def collect_data():
    """Run data collection"""
    print("🔄 Starting data collection...")
    try:
        from data_extractor import main as collect_data
        collect_data()
        print("✅ Data collection completed!")
        return True
    except Exception as e:
        print(f"❌ Data collection failed: {e}")
        return False

def launch_dashboard(advanced=True, port=8050, debug=True):
    """Launch the dashboard"""
    dashboard_type = "Advanced" if advanced else "Basic"
    print(f"🚀 Launching {dashboard_type} Dashboard...")
    print(f"📊 Dashboard will be available at: http://127.0.0.1:{port}")
    print("🔄 Loading dashboard components...")

    try:
        if advanced:
            from dashboard_advanced import AdvancedUCDashboard
            dashboard = AdvancedUCDashboard()
        else:
            from dashboard import UCDashboard
            dashboard = UCDashboard()

        print("✅ Dashboard loaded successfully!")
        print(f"🌐 Open your browser to: http://127.0.0.1:{port}")
        print("🔄 Press Ctrl+C to stop the server")

        dashboard.run(debug=debug, port=port)

    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Universal Credit Dashboard Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dashboard.py                    # Launch advanced dashboard
  python run_dashboard.py --collect         # Collect data then launch
  python run_dashboard.py --basic           # Launch basic dashboard
  python run_dashboard.py --port 8080       # Launch on custom port
  python run_dashboard.py --collect --port 8080  # Collect data and launch on port 8080
        """
    )

    parser.add_argument(
        '--collect',
        action='store_true',
        help='Collect fresh data before launching dashboard'
    )

    parser.add_argument(
        '--basic',
        action='store_true',
        help='Launch basic dashboard instead of advanced'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port to run dashboard on (default: 8050)'
    )

    parser.add_argument(
        '--no-debug',
        action='store_true',
        help='Disable debug mode'
    )

    args = parser.parse_args()

    # Print header
    print("=" * 60)
    print("🏛️  UNIVERSAL CREDIT ANALYTICS DASHBOARD")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check requirements
    print("🔍 Checking system requirements...")
    if not check_requirements():
        sys.exit(1)

    if not check_api_key():
        sys.exit(1)

    print("✅ All requirements satisfied!")
    print()

    # Collect data if requested
    if args.collect:
        if not collect_data():
            print("⚠️  Data collection failed, but continuing with existing data...")
        print()

    # Launch dashboard
    success = launch_dashboard(
        advanced=not args.basic,
        port=args.port,
        debug=not args.no_debug
    )

    if not success:
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
        print("👋 Thank you for using Universal Credit Analytics Dashboard!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)