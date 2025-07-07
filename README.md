# ✈️ Airline Market Analytics Web App

A real-time, interactive airline market analytics dashboard built with **Streamlit**. It helps users analyze demand trends, pricing patterns, and popular routes using data visualizations and AI-generated insights from **Gemini AI**.

---

## 📊 Features

### 🔍 Advanced Filtering System
- Multi-select filters for **origin**, **destination**, and **airline**
- Price range slider
- Flight duration filter
- Route type (One-way/Round-trip)
- Booking class filter (Economy, Business, etc.)
- Date and time filters
- Minimum availability filter

### 📈 Visual Analytics
- Histogram of ticket price distribution
- Pie chart showing airline market share
- Bar chart of most popular routes
- Scatter plot: Price vs. Flight duration
- Hourly and daily demand trend analysis

### 🤖 AI-Powered Market Insights
- Real-time AI summary using Gemini AI
- Airline performance comparisons
- Trend-based suggestions and alerts
- Fallback to simple insights if API is unavailable

### 📋 Data Table & Export
- Interactive table of filtered results
- Sortable columns
- CSV download option

### 💎 Clean and Responsive UI
- Tabbed layout for better navigation
- Card-style metric displays
- Theme customization via `.streamlit/config.toml`

---

## 🧰 Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| Streamlit | Web app interface |
| Pandas, NumPy | Data processing |
| Plotly, Seaborn, Matplotlib | Data visualizations |
| Google Generative AI | AI-powered insights |
| Requests | API and web scraping support |

---

## 🚀 Getting Started

### 🔧 Prerequisites
- Python 3.9 or later
- Gemini API key (optional, for AI insight)

### 🔌 Installation

1. **Clone this repo**
```bash
git clone https://github.com/kKARTHIK-K-P/airline-market-analytics.git
cd airline-market-analytics
