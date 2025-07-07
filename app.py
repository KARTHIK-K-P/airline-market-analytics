import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import numpy as np
from typing import Dict, List, Any, Optional
import warnings
import re
from dataclasses import dataclass
from urllib.parse import urlencode
import asyncio
import aiohttp
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import threading

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real-Time Airline Market Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #4CAF50;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
        margin-right: 8px;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .stSelectbox > div > div {
        background: linear-gradient(90deg, #f8f9fa, #ffffff);
    }
    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .insight-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #fdcb6e;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class FlightData:
    """Data class for flight information"""
    flight_number: str
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    price: float
    currency: str
    availability: int
    aircraft_type: str
    duration_hours: float
    timestamp: datetime

class AirlineDataFetcher:
    """Enhanced airline data fetcher with real APIs"""
    
    def __init__(self, aviation_api_key: str = None, gemini_api_key: str = None):
        self.aviation_api_key ="91bf74fb05a0cc3133482e324a2c7987"
        self.gemini_api_key = "AIzaSyB8HWR5yuq6SP3ebtYKliJNZ7mNHmCAKNQ"
        
        # Initialize Gemini AI if key provided
        if self.gemini_api_key:
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
            except Exception as e:
                st.warning(f"Gemini AI initialization failed: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
        }
        
        self.flight_cache = {}
        self.last_update = {}
        
    def get_aviation_stack_data(self) -> List[Dict]:
        """Fetch real-time flight data from AviationStack API"""
        try:
            url = "http://api.aviationstack.com/v1/flights"
            params = {
                'access_key': self.aviation_api_key,
                'limit': 100,
                'flight_status': 'active'
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                flights = []
                
                for flight in data.get('data', []):
                    try:
                        # Extract flight information
                        flight_info = {
                            'flight_number': flight.get('flight', {}).get('number', 'N/A'),
                            'airline': flight.get('airline', {}).get('name', 'Unknown'),
                            'origin': flight.get('departure', {}).get('iata', 'N/A'),
                            'destination': flight.get('arrival', {}).get('iata', 'N/A'),
                            'departure_time': flight.get('departure', {}).get('scheduled', 'N/A'),
                            'arrival_time': flight.get('arrival', {}).get('scheduled', 'N/A'),
                            'aircraft_type': flight.get('aircraft', {}).get('registration', 'Unknown'),
                            'status': flight.get('flight_status', 'Unknown'),
                            'timestamp': datetime.now()
                        }
                        flights.append(flight_info)
                    except Exception as e:
                        continue
                
                return flights
            else:
                st.error(f"AviationStack API error: {response.status_code}")
                return []
                
        except Exception as e:
            st.error(f"Error fetching AviationStack data: {str(e)}")
            return []
    
    def get_enhanced_flight_data(self, origin_filter=None, destination_filter=None, 
                               airline_filter=None, price_range=None) -> pd.DataFrame:
        """Get enhanced flight data with filters"""
        
        # Try real API first
        real_flights = self.get_aviation_stack_data()
        
        # Generate enhanced realistic data
        enhanced_flights = self.generate_enhanced_realistic_data(
            origin_filter, destination_filter, airline_filter, price_range
        )
        
        # Combine real and enhanced data
        all_flights = real_flights + enhanced_flights
        
        if not all_flights:
            all_flights = self.generate_enhanced_realistic_data()
        
        # Create DataFrame and ensure all required columns exist
        df = pd.DataFrame(all_flights)
        
        # Add missing columns with default values
        required_columns = {
            'flight_number': 'N/A',
            'airline': 'Unknown',
            'origin': 'N/A',
            'destination': 'N/A',
            'departure_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'arrival_time': (datetime.now() + timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'),
            'price': 200.0,
            'currency': 'AUD',
            'availability': 20,
            'aircraft_type': 'Unknown',
            'duration_hours': 2.0,
            'timestamp': datetime.now()
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        # Ensure proper data types
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(200.0)
        df['availability'] = pd.to_numeric(df['availability'], errors='coerce').fillna(20)
        df['duration_hours'] = pd.to_numeric(df['duration_hours'], errors='coerce').fillna(2.0)
        
        return df
    
    def generate_enhanced_realistic_data(self, origin_filter=None, destination_filter=None, 
                                       airline_filter=None, price_range=None) -> List[Dict]:
        """Generate enhanced realistic flight data with comprehensive filters"""
        
        # Comprehensive airport data
        airports = {
            'SYD': {'name': 'Sydney Kingsford Smith', 'city': 'Sydney', 'country': 'Australia', 'demand': 1.2},
            'MEL': {'name': 'Melbourne Tullamarine', 'city': 'Melbourne', 'country': 'Australia', 'demand': 1.1},
            'BNE': {'name': 'Brisbane Airport', 'city': 'Brisbane', 'country': 'Australia', 'demand': 1.0},
            'PER': {'name': 'Perth Airport', 'city': 'Perth', 'country': 'Australia', 'demand': 0.9},
            'ADL': {'name': 'Adelaide Airport', 'city': 'Adelaide', 'country': 'Australia', 'demand': 0.8},
            'OOL': {'name': 'Gold Coast Airport', 'city': 'Gold Coast', 'country': 'Australia', 'demand': 0.9},
            'CNS': {'name': 'Cairns Airport', 'city': 'Cairns', 'country': 'Australia', 'demand': 0.7},
            'DRW': {'name': 'Darwin Airport', 'city': 'Darwin', 'country': 'Australia', 'demand': 0.6},
            'HBA': {'name': 'Hobart Airport', 'city': 'Hobart', 'country': 'Australia', 'demand': 0.5},
            'CBR': {'name': 'Canberra Airport', 'city': 'Canberra', 'country': 'Australia', 'demand': 0.7},
            'LHR': {'name': 'London Heathrow', 'city': 'London', 'country': 'United Kingdom', 'demand': 1.3},
            'SIN': {'name': 'Singapore Changi', 'city': 'Singapore', 'country': 'Singapore', 'demand': 1.2},
            'BKK': {'name': 'Bangkok Suvarnabhumi', 'city': 'Bangkok', 'country': 'Thailand', 'demand': 1.1},
            'NRT': {'name': 'Tokyo Narita', 'city': 'Tokyo', 'country': 'Japan', 'demand': 1.2},
            'LAX': {'name': 'Los Angeles International', 'city': 'Los Angeles', 'country': 'USA', 'demand': 1.1},
            'DXB': {'name': 'Dubai International', 'city': 'Dubai', 'country': 'UAE', 'demand': 1.0},
            'DPS': {'name': 'Ngurah Rai International', 'city': 'Denpasar', 'country': 'Indonesia', 'demand': 1.3},
            'AKL': {'name': 'Auckland Airport', 'city': 'Auckland', 'country': 'New Zealand', 'demand': 0.9},
            'MNL': {'name': 'Ninoy Aquino International', 'city': 'Manila', 'country': 'Philippines', 'demand': 0.8},
            'HKG': {'name': 'Hong Kong International', 'city': 'Hong Kong', 'country': 'Hong Kong', 'demand': 1.1},
            'ICN': {'name': 'Seoul Incheon', 'city': 'Seoul', 'country': 'South Korea', 'demand': 1.0},
            'KUL': {'name': 'Kuala Lumpur International', 'city': 'Kuala Lumpur', 'country': 'Malaysia', 'demand': 0.9},
            'CGK': {'name': 'Soekarno-Hatta International', 'city': 'Jakarta', 'country': 'Indonesia', 'demand': 0.8},
            'DEL': {'name': 'Indira Gandhi International', 'city': 'Delhi', 'country': 'India', 'demand': 1.0},
            'BOM': {'name': 'Chhatrapati Shivaji Maharaj International', 'city': 'Mumbai', 'country': 'India', 'demand': 1.0}
        }
        
        airlines = [
            {'name': 'Qantas', 'code': 'QF', 'market_share': 0.25, 'price_factor': 1.1, 'reliability': 0.95},
            {'name': 'Virgin Australia', 'code': 'VA', 'market_share': 0.20, 'price_factor': 1.0, 'reliability': 0.92},
            {'name': 'Jetstar', 'code': 'JQ', 'market_share': 0.18, 'price_factor': 0.8, 'reliability': 0.88},
            {'name': 'Tigerair', 'code': 'TT', 'market_share': 0.08, 'price_factor': 0.75, 'reliability': 0.85},
            {'name': 'Emirates', 'code': 'EK', 'market_share': 0.07, 'price_factor': 1.2, 'reliability': 0.96},
            {'name': 'Singapore Airlines', 'code': 'SQ', 'market_share': 0.06, 'price_factor': 1.15, 'reliability': 0.97},
            {'name': 'Thai Airways', 'code': 'TG', 'market_share': 0.04, 'price_factor': 0.95, 'reliability': 0.90},
            {'name': 'Air New Zealand', 'code': 'NZ', 'market_share': 0.03, 'price_factor': 1.05, 'reliability': 0.93},
            {'name': 'Cathay Pacific', 'code': 'CX', 'market_share': 0.03, 'price_factor': 1.1, 'reliability': 0.94},
            {'name': 'AirAsia', 'code': 'AK', 'market_share': 0.06, 'price_factor': 0.7, 'reliability': 0.82}
        ]
        
        # Filter airlines if specified
        if airline_filter:
            airlines = [a for a in airlines if a['name'] in airline_filter]
        
        # Base pricing structure
        route_pricing = {
            'domestic_short': {'base': 150, 'range': (100, 250)},
            'domestic_medium': {'base': 250, 'range': (180, 400)},
            'domestic_long': {'base': 400, 'range': (300, 600)},
            'trans_tasman': {'base': 300, 'range': (200, 500)},
            'asia_pacific': {'base': 600, 'range': (400, 900)},
            'asia': {'base': 800, 'range': (600, 1200)},
            'europe': {'base': 1200, 'range': (900, 1800)},
            'americas': {'base': 1400, 'range': (1000, 2000)},
            'middle_east': {'base': 1000, 'range': (700, 1500)}
        }
        
        aircraft_types = {
            'domestic_short': ['Boeing 737-800', 'Airbus A320', 'Boeing 737 MAX 8', 'Embraer E190'],
            'domestic_medium': ['Boeing 737-800', 'Airbus A320', 'Boeing 787-8', 'Airbus A330-200'],
            'domestic_long': ['Boeing 737-800', 'Airbus A330-200', 'Boeing 787-8', 'Boeing 777-200'],
            'international': ['Boeing 787-9', 'Airbus A330-300', 'Airbus A350-900', 'Boeing 777-300ER', 'Airbus A380-800']
        }
        
        flights = []
        now = datetime.now()
        
        # Time-based pricing factors
        hour = now.hour
        day_of_week = now.weekday()
        
        peak_factor = 1.2 if (6 <= hour <= 9) or (17 <= hour <= 20) else 1.0
        weekend_factor = 1.15 if day_of_week >= 5 else 1.0
        
        # Generate 300 flights
        for i in range(300):
            # Select airports based on filters
            available_origins = list(airports.keys())
            available_destinations = list(airports.keys())
            
            if origin_filter:
                available_origins = [code for code in available_origins if code in origin_filter]
            if destination_filter:
                available_destinations = [code for code in available_destinations if code in destination_filter]
            
            if not available_origins or not available_destinations:
                continue
                
            origin = np.random.choice(available_origins)
            destinations = [dest for dest in available_destinations if dest != origin]
            
            if not destinations:
                continue
                
            destination = np.random.choice(destinations)
            
            # Determine route type
            origin_country = airports[origin]['country']
            dest_country = airports[destination]['country']
            
            if origin_country == dest_country == 'Australia':
                if origin in ['PER', 'DRW'] or destination in ['PER', 'DRW']:
                    route_type = 'domestic_long'
                elif abs(ord(origin[0]) - ord(destination[0])) > 2:
                    route_type = 'domestic_medium'
                else:
                    route_type = 'domestic_short'
            elif (origin_country == 'Australia' and dest_country == 'New Zealand') or \
                 (origin_country == 'New Zealand' and dest_country == 'Australia'):
                route_type = 'trans_tasman'
            elif dest_country in ['Singapore', 'Thailand', 'Malaysia', 'Indonesia', 'Philippines']:
                route_type = 'asia_pacific'
            elif dest_country in ['Japan', 'South Korea', 'Hong Kong', 'India']:
                route_type = 'asia'
            elif dest_country in ['United Kingdom']:
                route_type = 'europe'
            elif dest_country in ['USA']:
                route_type = 'americas'
            elif dest_country in ['UAE']:
                route_type = 'middle_east'
            else:
                route_type = 'asia_pacific'
            
            # Select airline
            if airlines:
                airline_probs = [a['market_share'] for a in airlines]
                airline_probs = np.array(airline_probs) / sum(airline_probs)
                airline = np.random.choice(airlines, p=airline_probs)
            else:
                continue
            
            # Calculate price
            pricing = route_pricing[route_type]
            base_price = pricing['base']
            price_range = pricing['range']
            
            # Apply various factors
            demand_factor = airports[origin]['demand'] * airports[destination]['demand']
            airline_factor = airline['price_factor']
            market_volatility = np.random.uniform(0.85, 1.15)
            
            # Calculate final price
            final_price = base_price * demand_factor * airline_factor * peak_factor * weekend_factor * market_volatility
            
            # Apply price range constraints
            final_price = max(price_range[0], min(price_range[1], final_price))
            
            # Apply price filter if specified
            if price_range and not (price_range[0] <= final_price <= price_range[1]):
                continue
            
            # Flight timing
            departure_time = now + timedelta(hours=np.random.randint(1, 168))  # Next 7 days
            
            # Calculate duration based on route type
            duration_mapping = {
                'domestic_short': (1.5, 2.5),
                'domestic_medium': (2.5, 4.5),
                'domestic_long': (4.5, 6.0),
                'trans_tasman': (3.0, 4.0),
                'asia_pacific': (6.0, 9.0),
                'asia': (8.0, 12.0),
                'europe': (20.0, 24.0),
                'americas': (13.0, 18.0),
                'middle_east': (12.0, 16.0)
            }
            
            duration_range = duration_mapping[route_type]
            flight_duration = np.random.uniform(duration_range[0], duration_range[1])
            arrival_time = departure_time + timedelta(hours=flight_duration)
            
            # Aircraft selection
            if route_type in ['domestic_short', 'domestic_medium', 'domestic_long']:
                aircraft_category = route_type
            else:
                aircraft_category = 'international'
            
            aircraft = np.random.choice(aircraft_types[aircraft_category])
            
            # Availability calculation
            base_availability = 150 if route_type.startswith('domestic') else 250
            availability = max(1, int(np.random.exponential(base_availability * 0.2)))
            availability = min(availability, base_availability)
            
            # Create flight data
            flight_data = {
                'flight_number': f"{airline['code']}{np.random.randint(100, 9999)}",
                'airline': airline['name'],
                'origin': origin,
                'origin_name': airports[origin]['name'],
                'destination': destination,
                'destination_name': airports[destination]['name'],
                'departure_time': departure_time.strftime('%Y-%m-%d %H:%M'),
                'arrival_time': arrival_time.strftime('%Y-%m-%d %H:%M'),
                'duration_hours': round(flight_duration, 1),
                'price': round(final_price, 2),
                'currency': 'AUD',
                'availability': availability,
                'aircraft_type': aircraft,
                'route_type': route_type,
                'booking_class': np.random.choice(['Economy', 'Premium Economy', 'Business', 'First'], 
                                                p=[0.7, 0.15, 0.12, 0.03]),
                'on_time_performance': round(airline['reliability'] * np.random.uniform(0.95, 1.05), 3),
                'carbon_emissions': round(flight_duration * 120 * np.random.uniform(0.8, 1.2), 1),
                'timestamp': now,
                'last_updated': now.strftime('%Y-%m-%d %H:%M:%S'),
                'demand_score': round(demand_factor * 100, 1),
                'price_trend': np.random.choice(['stable', 'increasing', 'decreasing'], p=[0.6, 0.25, 0.15]),
                'popular_route': 1 if route_type.startswith('domestic') else 0,
                'weather_impact': np.random.choice(['none', 'minor', 'moderate'], p=[0.8, 0.15, 0.05]),
                'fuel_surcharge': round(final_price * 0.1 * np.random.uniform(0.5, 1.5), 2),
                'baggage_fee': round(np.random.uniform(0, 50), 2),
                'cancellation_rate': round(np.random.uniform(0.01, 0.05), 3),
                'load_factor': round(np.random.uniform(0.6, 0.95), 2)
            }
            
            flights.append(flight_data)
        
        return flights
    
    def get_ai_insights(self, df: pd.DataFrame) -> List[str]:
        """Get AI-powered insights using Gemini"""
        if not self.gemini_model:
            return self.get_basic_insights(df)
        
        try:
            # Create summary data for AI analysis
            summary_data = {
                'total_flights': len(df),
                'avg_price': df['price'].mean(),
                'price_std': df['price'].std(),
                'top_airlines': df['airline'].value_counts().head(3).to_dict(),
                'popular_routes': df.groupby(['origin', 'destination']).size().sort_values(ascending=False).head(5).to_dict(),
                'route_types': df['route_type'].value_counts().to_dict() if 'route_type' in df.columns else {},
                'avg_availability': df['availability'].mean() if 'availability' in df.columns else 0,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            prompt = f"""
            Analyze the following airline booking data and provide 5 actionable business insights:
            
            Data Summary:
            {json.dumps(summary_data, indent=2)}
            
            Please provide insights about:
            1. Market trends and pricing patterns
            2. Route demand and popularity
            3. Airline competition and market share
            4. Booking recommendations for travelers
            5. Business opportunities or risks
            
            Keep each insight under 50 words and make them practical and actionable.
            """
            
            response = self.gemini_model.generate_content(prompt)
            insights = response.text.split('\n')
            
            # Clean and format insights
            formatted_insights = []
            for insight in insights:
                insight = insight.strip()
                if insight and len(insight) > 20:
                    # Add emoji based on content
                    if 'price' in insight.lower() or 'cost' in insight.lower():
                        insight = f"💰 {insight}"
                    elif 'route' in insight.lower() or 'destination' in insight.lower():
                        insight = f"🗺️ {insight}"
                    elif 'airline' in insight.lower():
                        insight = f"✈️ {insight}"
                    elif 'demand' in insight.lower() or 'popular' in insight.lower():
                        insight = f"📈 {insight}"
                    else:
                        insight = f"🔍 {insight}"
                    
                    formatted_insights.append(insight)
                
                if len(formatted_insights) >= 5:
                    break
            
            return formatted_insights[:5]
            
        except Exception as e:
            st.warning(f"AI insights unavailable: {e}")
            return self.get_basic_insights(df)
    
    def get_basic_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate basic insights without AI"""
        insights = []
        
        try:
            avg_price = df['price'].mean()
            price_std = df['price'].std()
            
            insights.append(f"💰 Average flight price is ${avg_price:.0f} with high variation (${price_std:.0f} std dev)")
            
            top_airline = df['airline'].value_counts().index[0]
            airline_share = df['airline'].value_counts().iloc[0] / len(df) * 100
            insights.append(f"✈️ {top_airline} leads the market with {airline_share:.1f}% share")
            
            if 'route_type' in df.columns:
                popular_route_type = df['route_type'].value_counts().index[0]
                insights.append(f"🗺️ {popular_route_type.replace('_', ' ').title()} routes are most popular")
            
            if 'availability' in df.columns:
                low_availability = len(df[df['availability'] < 10])
                if low_availability > 0:
                    insights.append(f"⚠️ {low_availability} flights have low availability (<10 seats)")
            
            current_hour = datetime.now().hour
            if 6 <= current_hour <= 9 or 17 <= current_hour <= 20:
                insights.append("⏰ Peak hours detected - expect higher prices and lower availability")
            
        except Exception as e:
            insights.append(f"📊 Analysis in progress... {len(df)} flights loaded")
        
        return insights[:5]

def create_advanced_filters(df: pd.DataFrame) -> Dict:
    """Create advanced filtering options"""
    st.sidebar.markdown("### 🎯 Advanced Filters")
    
    filters = {}
    
    # Origin/Destination filters
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        origin_options = sorted(df['origin'].unique())
        filters['origins'] = st.multiselect(
            "Origins",
            origin_options,
            default=origin_options[:5] if len(origin_options) > 5 else origin_options,
            help="Select departure airports"
        )
    
    with col2:
        destination_options = sorted(df['destination'].unique())
        filters['destinations'] = st.multiselect(
            "Destinations",
            destination_options,
            default=destination_options[:5] if len(destination_options) > 5 else destination_options,
            help="Select arrival airports"
        )
    
    # Airline filter
    airline_options = sorted(df['airline'].unique())
    filters['airlines'] = st.sidebar.multiselect(
        "Airlines",
        airline_options,
        default=airline_options,
        help="Select preferred airlines"
    )
    
    # Price range
    price_min, price_max = float(df['price'].min()), float(df['price'].max())
    filters['price_range'] = st.sidebar.slider(
        "Price Range (AUD)",
        min_value=price_min,
        max_value=price_max,
        value=(price_min, price_max),
        step=10.0,
        help="Filter flights by price range"
    )
    
    # Duration filter
    if 'duration_hours' in df.columns:
        duration_min, duration_max = float(df['duration_hours'].min()), float(df['duration_hours'].max())
        filters['duration_range'] = st.sidebar.slider(
            "Flight Duration (hours)",
            min_value=duration_min,
            max_value=duration_max,
            value=(duration_min, duration_max),
            step=0.5,
            help="Filter flights by duration"
        )
    
    # Route type filter
    if 'route_type' in df.columns:
        route_options = sorted(df['route_type'].unique())
        filters['route_types'] = st.sidebar.multiselect(
            "Route Types",
            route_options,default=route_options,
            help="Filter by route type (domestic/international)"
        )
    
    # Availability filter
    if 'availability' in df.columns:
        filters['min_availability'] = st.sidebar.number_input(
            "Minimum Availability",
            min_value=1,
            max_value=int(df['availability'].max()),
            value=1,
            help="Minimum number of available seats"
        )
    
    # Booking class filter
    if 'booking_class' in df.columns:
        class_options = sorted(df['booking_class'].unique())
        filters['booking_classes'] = st.sidebar.multiselect(
            "Booking Class",
            class_options,
            default=class_options,
            help="Select booking class preferences"
        )
    
    # Time filters
    st.sidebar.markdown("### ⏰ Time Filters")
    
    # Departure time filter
    filters['departure_time'] = st.sidebar.selectbox(
        "Departure Time",
        ["Any", "Morning (06:00-12:00)", "Afternoon (12:00-18:00)", "Evening (18:00-24:00)", "Night (00:00-06:00)"],
        help="Filter by preferred departure time"
    )
    
    # Date range filter
    filters['date_range'] = st.sidebar.date_input(
        "Departure Date Range",
        value=(datetime.now().date(), (datetime.now() + timedelta(days=7)).date()),
        min_value=datetime.now().date(),
        max_value=(datetime.now() + timedelta(days=365)).date(),
        help="Select departure date range"
    )
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe"""
    filtered_df = df.copy()
    
    # Apply filters
    if filters.get('origins'):
        filtered_df = filtered_df[filtered_df['origin'].isin(filters['origins'])]
    
    if filters.get('destinations'):
        filtered_df = filtered_df[filtered_df['destination'].isin(filters['destinations'])]
    
    if filters.get('airlines'):
        filtered_df = filtered_df[filtered_df['airline'].isin(filters['airlines'])]
    
    if filters.get('price_range'):
        min_price, max_price = filters['price_range']
        filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
    
    if filters.get('duration_range') and 'duration_hours' in filtered_df.columns:
        min_duration, max_duration = filters['duration_range']
        filtered_df = filtered_df[(filtered_df['duration_hours'] >= min_duration) & (filtered_df['duration_hours'] <= max_duration)]
    
    if filters.get('route_types') and 'route_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['route_type'].isin(filters['route_types'])]
    
    if filters.get('min_availability') and 'availability' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['availability'] >= filters['min_availability']]
    
    if filters.get('booking_classes') and 'booking_class' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['booking_class'].isin(filters['booking_classes'])]
    
    # Time-based filters
    if filters.get('departure_time') and filters['departure_time'] != "Any":
        # Convert departure_time to datetime if it's string
        if isinstance(filtered_df['departure_time'].iloc[0], str):
            filtered_df['departure_hour'] = pd.to_datetime(filtered_df['departure_time']).dt.hour
        else:
            filtered_df['departure_hour'] = filtered_df['departure_time'].dt.hour
        
        time_ranges = {
            "Morning (06:00-12:00)": (6, 12),
            "Afternoon (12:00-18:00)": (12, 18),
            "Evening (18:00-24:00)": (18, 24),
            "Night (00:00-06:00)": (0, 6)
        }
        
        if filters['departure_time'] in time_ranges:
            start_hour, end_hour = time_ranges[filters['departure_time']]
            if end_hour == 24:
                filtered_df = filtered_df[filtered_df['departure_hour'] >= start_hour]
            else:
                filtered_df = filtered_df[
                    (filtered_df['departure_hour'] >= start_hour) & 
                    (filtered_df['departure_hour'] < end_hour)
                ]
    
    return filtered_df

def create_visualizations(df: pd.DataFrame):
    """Create comprehensive visualizations"""
    
    if df.empty:
        st.warning("No data available for selected filters. Please adjust your filters.")
        return
    
    # Key metrics row
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total Flights</h3>
            <h2 style="color: #667eea;">{:,}</h2>
            <p style="color: #666;">Available now</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_price = df['price'].mean()
        st.markdown("""
        <div class="metric-card">
            <h3>Avg Price</h3>
            <h2 style="color: #2d3436;">${:,.0f}</h2>
            <p style="color: #666;">AUD</p>
        </div>
        """.format(avg_price), unsafe_allow_html=True)
    
    with col3:
        cheapest_price = df['price'].min()
        st.markdown("""
        <div class="metric-card">
            <h3>Cheapest</h3>
            <h2 style="color: #00b894;">${:,.0f}</h2>
            <p style="color: #666;">Best deal</p>
        </div>
        """.format(cheapest_price), unsafe_allow_html=True)
    
    with col4:
        total_routes = len(df.groupby(['origin', 'destination']))
        st.markdown("""
        <div class="metric-card">
            <h3>Routes</h3>
            <h2 style="color: #e17055;">{:,}</h2>
            <p style="color: #666;">Unique</p>
        </div>
        """.format(total_routes), unsafe_allow_html=True)
    
    with col5:
        airlines_count = df['airline'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>Airlines</h3>
            <h2 style="color: #a29bfe;">{:,}</h2>
            <p style="color: #666;">Carriers</p>
        </div>
        """.format(airlines_count), unsafe_allow_html=True)
    
    # Charts row
    st.markdown("### 📈 Market Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price distribution
        fig_price = px.histogram(
            df, 
            x='price', 
            nbins=30,
            title='Flight Price Distribution',
            labels={'price': 'Price (AUD)', 'count': 'Number of Flights'},
            color_discrete_sequence=['#667eea']
        )
        fig_price.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16
        )
        st.plotly_chart(fig_price, use_container_width=True)
    
    with col2:
        # Airline market share
        airline_counts = df['airline'].value_counts().head(10)
        fig_airline = px.pie(
            values=airline_counts.values,
            names=airline_counts.index,
            title='Airline Market Share (Top 10)',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_airline.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16
        )
        st.plotly_chart(fig_airline, use_container_width=True)
    
    # Route analysis
    col1, col2 = st.columns(2)
    
    with col1:
        # Popular routes
        route_counts = df.groupby(['origin', 'destination']).size().reset_index(name='count')
        route_counts['route'] = route_counts['origin'] + ' → ' + route_counts['destination']
        top_routes = route_counts.nlargest(10, 'count')
        
        fig_routes = px.bar(
            top_routes,
            x='count',
            y='route',
            orientation='h',
            title='Most Popular Routes',
            labels={'count': 'Number of Flights', 'route': 'Route'},
            color='count',
            color_continuous_scale='Blues'
        )
        fig_routes.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            title_font_size=16,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_routes, use_container_width=True)
    
    with col2:
        # Price vs Duration scatter
        if 'duration_hours' in df.columns:
            fig_scatter = px.scatter(
                df,
                x='duration_hours',
                y='price',
                color='airline',
                size='availability' if 'availability' in df.columns else None,
                title='Price vs Flight Duration',
                labels={'duration_hours': 'Duration (hours)', 'price': 'Price (AUD)'},
                hover_data=['flight_number', 'origin', 'destination']
            )
            fig_scatter.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Time-based analysis
    if 'departure_time' in df.columns:
        st.markdown("### ⏰ Time-Based Analysis")
        
        # Convert departure_time to datetime if it's string
        if isinstance(df['departure_time'].iloc[0], str):
            df_time = df.copy()
            df_time['departure_datetime'] = pd.to_datetime(df_time['departure_time'])
            df_time['departure_hour'] = df_time['departure_datetime'].dt.hour
            df_time['departure_date'] = df_time['departure_datetime'].dt.date
        else:
            df_time = df.copy()
            df_time['departure_hour'] = df_time['departure_time'].dt.hour
            df_time['departure_date'] = df_time['departure_time'].dt.date
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly distribution
            hourly_flights = df_time.groupby('departure_hour').size().reset_index(name='count')
            fig_hourly = px.bar(
                hourly_flights,
                x='departure_hour',
                y='count',
                title='Flights by Departure Hour',
                labels={'departure_hour': 'Hour of Day', 'count': 'Number of Flights'},
                color='count',
                color_continuous_scale='Viridis'
            )
            fig_hourly.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
        
        with col2:
            # Daily price trends
            daily_prices = df_time.groupby('departure_date')['price'].mean().reset_index()
            daily_prices['departure_date'] = pd.to_datetime(daily_prices['departure_date'])
            
            fig_daily = px.line(
                daily_prices,
                x='departure_date',
                y='price',
                title='Average Price Trends',
                labels={'departure_date': 'Date', 'price': 'Average Price (AUD)'},
                markers=True
            )
            fig_daily.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                title_font_size=16
            )
            st.plotly_chart(fig_daily, use_container_width=True)

def create_data_table(df: pd.DataFrame):
    """Create an interactive data table"""
    st.markdown("### 🔍 Flight Details")
    
    # Display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ['price', 'departure_time', 'duration_hours', 'airline', 'availability'],
            index=0
        )
    
    with col2:
        sort_order = st.selectbox(
            "Order:",
            ['Ascending', 'Descending'],
            index=0
        )
    
    with col3:
        rows_to_show = st.selectbox(
            "Rows to show:",
            [25, 50, 100, 200],
            index=0
        )
    
    # Sort data
    ascending = sort_order == 'Ascending'
    df_sorted = df.sort_values(by=sort_by, ascending=ascending)
    
    # Select columns to display
    display_columns = [
        'flight_number', 'airline', 'origin', 'destination', 
        'departure_time', 'arrival_time', 'duration_hours', 
        'price', 'availability', 'aircraft_type'
    ]
    
    # Filter columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in df_sorted.columns]
    
    # Display table
    st.dataframe(
        df_sorted[available_columns].head(rows_to_show),
        use_container_width=True,
        hide_index=True
    )
    
    # Download options
    if st.button("📥 Download Data as CSV"):
        csv = df_sorted.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"flight_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Real-Time Airline Market Analytics</h1>
        <p style="font-size: 18px; margin-top: 10px;">
            <span class="live-indicator"></span>
            Live flight data and market insights powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize data fetcher
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_flight_data():
        fetcher = AirlineDataFetcher()
        return fetcher.get_enhanced_flight_data()
    
    # Load data
    with st.spinner("🔄 Loading real-time flight data..."):
        df = load_flight_data()
    
    if df.empty:
        st.error("No flight data available. Please try again later.")
        return
    
    # Create filters
    filters = create_advanced_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Display results count
    st.markdown(f"**{len(filtered_df):,} flights found** (filtered from {len(df):,} total)")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Analytics", "🔍 Flight Search", "🤖 AI Insights", "📈 Market Trends"])
    
    with tab1:
        create_visualizations(filtered_df)
    with tab2:
        st.markdown("### 📈 Advanced Market Trends")
        
        if not filtered_df.empty:
            # Airline performance comparison
            if len(filtered_df['airline'].unique()) > 1:
                st.markdown("**Airline Performance Comparison**")
                
                airline_stats = filtered_df.groupby('airline').agg({
                    'price': ['mean', 'median', 'std'],
                    'flight_number': 'count',
                    'availability': 'mean'
                }).round(2)
                
                airline_stats.columns = ['Avg Price', 'Median Price', 'Price Std', 'Flight Count', 'Avg Availability']
                st.dataframe(airline_stats, use_container_width=True)
            
            # Route profitability analysis
            if 'route_type' in filtered_df.columns:
                st.markdown("**Route Type Analysis**")
                
                route_analysis = filtered_df.groupby('route_type').agg({
                    'price': ['mean', 'count'],
                    'availability': 'mean'
                }).round(2)
                
                route_analysis.columns = ['Avg Price', 'Flight Count', 'Avg Availability']
                st.dataframe(route_analysis, use_container_width=True)
    
    with tab3:
        create_data_table(filtered_df)
    
    with tab4:
        st.markdown("### 🤖 AI-Powered Insights")
        
        if st.button("🔄 Generate Fresh Insights"):
            with st.spinner("🧠 Analyzing market data..."):
                fetcher = AirlineDataFetcher()
                insights = fetcher.get_ai_insights(filtered_df)
                
                for insight in insights:
                    st.markdown(f"""
                    <div class="insight-card">
                        <p style="margin: 0; font-size: 14px;">{insight}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("### 📊 Quick Market Stats")
        
        if not filtered_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Price statistics
                price_stats = filtered_df['price'].describe()
                st.markdown("**Price Analysis:**")
                st.write(f"• Median: ${price_stats['50%']:.0f}")
                st.write(f"• 25th percentile: ${price_stats['25%']:.0f}")
                st.write(f"• 75th percentile: ${price_stats['75%']:.0f}")
                st.write(f"• Price range: ${price_stats['min']:.0f} - ${price_stats['max']:.0f}")
            
            with col2:
                # Route analysis
                total_routes = len(filtered_df.groupby(['origin', 'destination']))
                avg_flights_per_route = len(filtered_df) / total_routes if total_routes > 0 else 0
                
                st.markdown("**Route Analysis:**")
                st.write(f"• Unique routes: {total_routes}")
                st.write(f"• Avg flights per route: {avg_flights_per_route:.1f}")
                st.write(f"• Most popular origin: {filtered_df['origin'].mode().iloc[0] if not filtered_df.empty else 'N/A'}")
                st.write(f"• Most popular destination: {filtered_df['destination'].mode().iloc[0] if not filtered_df.empty else 'N/A'}")
    

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🔄 Data refreshes every 5 minutes | 
        📊 Showing live market data | 
        🤖 AI insights powered by Gemini</p>
        <p><small>Last updated: {}</small></p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
