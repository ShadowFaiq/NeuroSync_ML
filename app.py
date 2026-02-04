import os
# Suppress TensorFlow warnings for faster startup
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from datetime import datetime
import logging
from pathlib import Path
import altair as alt
import plotly.graph_objects as go

# Import only lightweight utility modules at startup
from utils import load_config, ensure_directory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Neuro Sync",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Glow effect for sidebar radio buttons */
    [role="radio"] {
        transition: all 0.3s ease;
    }
    
    [role="radio"][aria-checked="true"] {
        box-shadow: 0 0 10px rgba(100, 200, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_nutrition_tracker():
    """Lazily load the nutrition tracker when needed."""
    logger.info("Loading NutritionTracker...")
    from processing.nutrition import NutritionTracker
    return NutritionTracker()


@st.cache_resource
def get_composite_assessment():
    """Lazily load the composite burnout assessment engine when needed."""
    logger.info("Loading CompositeBurnoutAssessment...")
    from composite_burnout_assessment import CompositeBurnoutAssessment
    return CompositeBurnoutAssessment()


@st.cache_resource
def get_sentiment_journaling():
    """Lazily load the sentiment journaling model when needed."""
    logger.info("Loading SentimentJournaling...")
    from processing.journaling import SentimentJournaling
    return SentimentJournaling()


@st.cache_resource
def get_food_analyzer():
    """Lazily load the food image analyzer when needed."""
    logger.info("Loading FoodImageAnalyzer...")
    from processing.nutrition import FoodImageAnalyzer
    return FoodImageAnalyzer()


@st.cache_resource
def initialize_survey_manager():
    """Initialize survey manager if credentials exist."""
    from integration.survey import SurveyManager
    config = load_config()
    if Path(config['google_sheets_credentials']).exists():
        try:
            survey_manager = SurveyManager(config['google_sheets_credentials'])
            return survey_manager
        except Exception as e:
            logger.warning(f"Survey manager initialization failed: {e}")
            return None
    return None


def main():
    """Main application function."""
    # Header
    st.markdown("""
    <div style='text-align: center'>
        <h1>🧠 Neuro Sync</h1>
        <p><i>A Multimodal Deep Learning Ecosystem for Integrative Mental and Physical Health Monitoring</i></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown("### Navigation")
    app_mode = st.sidebar.radio(
        "Select Module",
        [
            "Dashboard",
            "Sentiment Journaling",
            "Nutrition Tracker",
            "Sleep-Productivity Analysis",
            "Fitness Trends",
            "Burnout Risk",
        ]
    )

    # Load Firebase data when needed
    health_data = None
    if app_mode in ["Sleep-Productivity Analysis", "Fitness Trends", "Burnout Assessment", "Dashboard"]:
        from integration.firebase_manager import get_health_data
        health_data = get_health_data()

    # Display selected module
    if app_mode == "Dashboard":
        show_dashboard(health_data)
    elif app_mode == "Sentiment Journaling":
        sentiment_journaling = get_sentiment_journaling()
        show_sentiment_journaling(sentiment_journaling)
    elif app_mode == "Nutrition Tracker":
        nutrition_tracker = get_nutrition_tracker()
        show_nutrition_tracker(nutrition_tracker)
    elif app_mode == "Sleep-Productivity Analysis":
        show_sleep_productivity_analysis(health_data)
    elif app_mode == "Fitness Trends":
        show_fitness_trends(health_data)
    elif app_mode == "Burnout Risk":
        composite_assessment = get_composite_assessment()
        show_burnout_assessment(composite_assessment)


def show_dashboard(health_data):
    """Display main dashboard with real-time health metrics."""

    # Initialize session state for checklist if not exists
    if "check_survey" not in st.session_state:
        st.session_state.check_survey = False
    if "check_journaling" not in st.session_state:
        st.session_state.check_journaling = False
    if "check_nutrition" not in st.session_state:
        st.session_state.check_nutrition = False
    if "check_burnout" not in st.session_state:
        st.session_state.check_burnout = False
    if "balloons_shown" not in st.session_state:
        st.session_state.balloons_shown = False
    
    # Daily Check-in Survey Section (FIRST)
    st.subheader("✅ Daily Check-in")
    
    st.info("Complete your daily health survey to track your wellness metrics.")
    
    # Display Google Form link button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "📋 Open Neuro Sync Survey",
            "https://docs.google.com/forms/d/e/1FAIpQLSetVuLaE5OEyKZHJmvt_2x_Jq5uwZKIrtg7KNP2WlsQw8jJHg/viewform",
            width='stretch'
        )
    
    # Display instructions in a structured box
    col_left, col_center, col_right = st.columns([0.1, 0.8, 0.1])
    with col_center:
        with st.container(border=True):
            st.subheader("📊 How It Works")
            st.markdown("")
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("**1️⃣**")
            with col2:
                st.markdown("**Spend 2–3 minutes** sharing how you're feeling in our survey.")
            
            st.markdown("")
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("**2️⃣**")
            with col2:
                st.markdown("**Sit back** — we'll handle the rest!")
            
            st.markdown("")
            
            col1, col2 = st.columns([0.1, 0.9])
            with col1:
                st.markdown("**3️⃣**")
            with col2:
                st.markdown("**Get insights** — check out your custom health trends in the Sleep and Burnout tabs.")
    
    st.markdown("")
    
    # Daily Check-in Checklist - Clean Box Layout (SECOND)
    with st.container(border=True):
        st.subheader("✅ Check List")
        
        # Progress bar at top
        completed = sum([
            st.session_state.check_survey,
            st.session_state.check_journaling,
            st.session_state.check_nutrition,
            st.session_state.check_burnout
        ])
        total = 4
        progress = completed / total
        st.progress(progress)
        st.caption(f"**{completed}/{total} tasks completed today**")
        
        st.markdown("")
        
        # Vertical checklist items
        st.checkbox(
            "📝 Daily Survey Completed",
            key="check_survey"
        )
        
        st.checkbox(
            "✍️ Journaling Done",
            key="check_journaling"
        )
        
        st.checkbox(
            "🍎 Nutrition Logged",
            key="check_nutrition"
        )
        
        st.checkbox(
            "⚠️ Burnout Assessment",
            key="check_burnout"
        )
        
        st.markdown("")
        
        # Reset button with callback
        def reset_checklist():
            for key in ["check_survey", "check_journaling", "check_nutrition", "check_burnout", "balloons_shown"]:
                if key in st.session_state:
                    del st.session_state[key]
        
        st.button("🔄 Reset Daily Checklist", use_container_width=True, on_click=reset_checklist)
    
    # Show balloons when all tasks are completed
    if completed == total and not st.session_state.balloons_shown:
        st.balloons()
        st.success("🎉 Congratulations! You've completed all your daily tasks!")
        st.session_state.balloons_shown = True
    
    st.markdown("<br>", unsafe_allow_html=True)


    if health_data is None or health_data.empty:
        st.warning("No health data available yet. Submit entries via the Daily Check-in form.")
        return

    df = health_data.copy()

    # Ensure required columns and numeric types
    for col in ["sleep_hours", "productivity_score", "exercise_frequency", "sentiment_score", "nutrition_score", "health_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        # Normalize to tz-naive to avoid comparisons between tz-aware and tz-naive
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    if not set(["sleep_hours", "productivity_score", "timestamp"]).issubset(df.columns):
        st.error("Missing required columns: sleep_hours, productivity_score, timestamp")
        st.write("Available columns:", list(df.columns))
        return

    df = df.dropna(subset=["sleep_hours", "productivity_score", "timestamp"])
    if df.empty:
        st.warning("No valid data points to visualize yet.")
        return

    latest_ts = df["timestamp"].max()

    # Vibrant multi-metric vibe chart (sleep, productivity, fitness, sentiment, nutrition)
    metric_candidates = [
        ("sleep_hours", "😴 Sleep Hours"),
        ("productivity_score", "⚡ Productivity"),
        ("exercise_frequency", "🏃 Exercise Days/Week"),
        ("sentiment_score", "😊 Sentiment"),
        ("nutrition_score", "🍎 Nutrition"),
        ("health_score", "🍎 Nutrition"),
    ]

    # Try to pull nutrition logs and sentiment history to enrich the chart
    nutrition_df = None
    sentiment_df = None
    try:
        user_id = None
        if "user_id" in df.columns:
            non_null_users = df["user_id"].dropna().astype(str)
            user_id = non_null_users.iloc[0] if not non_null_users.empty else None

        from integration.firebase_manager import get_nutrition_logs, get_sentiment_history
        
        # Pull nutrition logs (try with and without user_id)
        nutrition_df = get_nutrition_logs(user_id=user_id if user_id else None)
        if (nutrition_df is None or nutrition_df.empty) and user_id:
            nutrition_df = get_nutrition_logs(user_id=None)
        if nutrition_df is None or nutrition_df.empty:
            nutrition_df = get_nutrition_logs()  # Try with no parameters
        
        # Pull sentiment history (try with and without user_id)
        sentiment_df = get_sentiment_history(user_id=user_id if user_id else None, days=7)
        if (sentiment_df is None or sentiment_df.empty) and user_id:
            sentiment_df = get_sentiment_history(user_id=None, days=7)
        if sentiment_df is None or sentiment_df.empty:
            sentiment_df = get_sentiment_history(user_id=None, days=7)  # Query all users if no specific user

    except Exception:
        nutrition_df = None
        sentiment_df = None

    chosen = []
    seen_labels = set()
    for col, label in metric_candidates:
        if col in df.columns and label not in seen_labels:
            chosen.append((col, label))
            seen_labels.add(label)

    # Add nutrition score and sentiment from logs if available
    extra_frames = []
    nutrition_added = False
    sentiment_added = False
    
    # Add sentiment history
    if sentiment_df is not None and not sentiment_df.empty:
        if "timestamp" in sentiment_df.columns:
            sentiment_df["timestamp"] = pd.to_datetime(sentiment_df["timestamp"], errors="coerce")
            sentiment_df["timestamp"] = sentiment_df["timestamp"].dt.tz_localize(None)
        
        # More flexible column detection for sentiment
        # Priority: sentiment_compound > sentiment_score > sentiment > compound > score
        sentiment_col = None
        for col_name in ['sentiment_compound', 'sentiment_score', 'sentiment', 'compound']:
            if col_name in sentiment_df.columns:
                sentiment_col = col_name
                break
        
        # Fallback: look for any column with 'sentiment' or 'score' in the name
        if sentiment_col is None:
            sentiment_cols = [col for col in sentiment_df.columns 
                            if any(x in col.lower() for x in ['sentiment', 'compound', 'score']) 
                            and col not in ['timestamp', 'user_id', 'entry_text', 'entry_type']]
            if sentiment_cols:
                sentiment_col = sentiment_cols[0]
        
        if sentiment_col:
            sentiment_df[sentiment_col] = pd.to_numeric(sentiment_df[sentiment_col], errors="coerce")
            # Normalize sentiment score to 0-10 scale if it's in -1 to 1 range
            if sentiment_df[sentiment_col].max() <= 1 and sentiment_df[sentiment_col].min() >= -1:
                sentiment_df[sentiment_col] = (sentiment_df[sentiment_col] + 1) * 5  # Convert -1..1 to 0..10
            
            sentiment_df = sentiment_df.dropna(subset=["timestamp", sentiment_col])
            if not sentiment_df.empty:
                extra_frames.append(
                    sentiment_df[["timestamp", sentiment_col]]
                    .rename(columns={sentiment_col: "value"})
                    .assign(metric_label="😊 Sentiment (Journaling)")
                )
                seen_labels.add("😊 Sentiment (Journaling)")
                sentiment_added = True
    
    # Add nutrition score from logs if available
    if nutrition_df is not None and not nutrition_df.empty:
        if "timestamp" in nutrition_df.columns:
            nutrition_df["timestamp"] = pd.to_datetime(nutrition_df["timestamp"], errors="coerce")
            nutrition_df["timestamp"] = nutrition_df["timestamp"].dt.tz_localize(None)
        
        # More flexible column detection for nutrition
        # Priority: health_score > nutrition_score > score
        nutrition_col = None
        for col_name in ['health_score', 'nutrition_score', 'score']:
            if col_name in nutrition_df.columns:
                nutrition_col = col_name
                break
        
        # Fallback: look for any column with 'health', 'nutrition', or 'score'
        if nutrition_col is None:
            nutrition_cols = [col for col in nutrition_df.columns 
                            if any(x in col.lower() for x in ['health', 'nutrition', 'score'])
                            and col not in ['timestamp', 'user_id', 'food_name', 'health_label']]
            if nutrition_cols:
                nutrition_col = nutrition_cols[0]
        
        if nutrition_col:
            nutrition_df[nutrition_col] = pd.to_numeric(nutrition_df[nutrition_col], errors="coerce")
            nutrition_df = nutrition_df.dropna(subset=["timestamp", nutrition_col])
            if not nutrition_df.empty:
                extra_frames.append(
                    nutrition_df[["timestamp", nutrition_col]]
                    .rename(columns={nutrition_col: "value"})
                    .assign(metric_label="🍎 Nutrition (Tracker)")
                )
                seen_labels.add("🍎 Nutrition (Tracker)")
                nutrition_added = True

    if len(chosen) + len(extra_frames) < 2:
        st.warning("Not enough metrics to plot. Add exercise_frequency, sentiment_score, or nutrition_score to see more lines.")
    else:
        st.subheader("Your Wellness Journey")
        
        plot_cols = [c for c, _ in chosen]
        plot_df = df[["timestamp"] + plot_cols].copy()
        plot_df = plot_df.dropna(subset=["timestamp"])
        plot_df = plot_df.melt("timestamp", var_name="metric", value_name="value")
        label_map = {c: l for c, l in chosen}
        plot_df["metric_label"] = plot_df["metric"].map(label_map)

        # Append sentiment and nutrition frames if any
        if extra_frames:
            extra_df = pd.concat(extra_frames, ignore_index=True)
            combined_df = pd.concat([plot_df, extra_df], ignore_index=True)
        else:
            combined_df = plot_df

        # Clean and sort combined data
        combined_df["value"] = pd.to_numeric(combined_df["value"], errors="coerce")
        combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], errors="coerce").dt.tz_localize(None)
        combined_df = combined_df.dropna(subset=["timestamp", "value"])
        
        if combined_df.empty:
            st.warning("❌ No valid data points to plot yet.")
            return
        
        combined_df = combined_df.sort_values("timestamp")
        
        color_palette = ["#FF6B6B", "#FFD166", "#06D6A0", "#4ECDC4", "#6C63FF", "#F72585", "#FF8FAB", "#FF006E", "#8338EC"]

        # Daily aggregation to avoid tangled lines (one point per day per metric)
        combined_df["date"] = combined_df["timestamp"].dt.date
        daily = combined_df.groupby(["date", "metric_label"], as_index=False)["value"].mean()
        
        if daily.empty:
            st.warning("❌ No aggregated data to plot yet.")
            return

        # Build stacked area chart with Plotly
        color_map = {}
        labels_in_order = list(daily["metric_label"].unique())
        for idx, lbl in enumerate(labels_in_order):
            color_map[lbl] = color_palette[idx % len(color_palette)]

        fig = go.Figure()
        for lbl in labels_in_order:
            subset = daily[daily["metric_label"] == lbl]
            fig.add_trace(
                go.Scatter(
                    x=subset["date"],
                    y=subset["value"],
                    mode="lines+markers",
                    name=lbl,
                    stackgroup="one",
                    line=dict(width=3, color=color_map.get(lbl, "#00FFCC")),
                    marker=dict(size=6),
                    hovertemplate=f"%{{x}}<br>{lbl}: %{{y:.2f}}<extra></extra>",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            height=450,
            xaxis=dict(title="Date", gridcolor="#333"),
            yaxis=dict(title="Value (daily average)", gridcolor="#333"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=80),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show helpful messages for missing data
        if not sentiment_added and not nutrition_added:
            st.caption("💡 Try adding sentiment journal entries and nutrition logs to enrich your wellness graph!")
        elif not sentiment_added:
            st.caption("😊 Add sentiment journal entries to see your mood trends in the graph.")
        elif not nutrition_added:
            st.caption("🍎 Log food entries to see your nutrition insights in the graph.")

    if latest_ts is not None:
        st.caption(f"Last updated: {latest_ts.strftime('%Y-%m-%d %H:%M:%S')}")


def show_sentiment_journaling(sentiment_journaling):
    """Display sentiment journaling module."""
    st.header("📝 Sentiment Journaling")
    
    tab1, tab2, tab3 = st.tabs(["Voice Journal", "Text Journal", "Trends"])
    
    with tab1:
        st.subheader("🎤 Record Voice Entry")
        st.info("Record your thoughts and feelings. We'll analyze the sentiment to track your mental health.")
        
        # Live microphone recording using Streamlit's native audio_input
        st.markdown("### 🎙️ Record from Microphone")
        audio_bytes = st.audio_input("Click to record your voice")
        
        if audio_bytes:
            st.audio(audio_bytes, format='audio/wav')
            
            if st.button("🔍 Analyze Recording", key="analyze_mic"):
                with st.spinner("🎯 Transcribing and analyzing sentiment..."):
                    try:
                        # Save audio bytes to a temporary file
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Process the audio
                        result = sentiment_journaling.process_journal_entry(tmp_path)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        if result:
                            st.success("✅ Analysis Complete!")
                            
                            # Display transcription
                            st.markdown("### 📝 Transcription")
                            st.write(f'*"{result["text"]}"*')
                            
                            st.markdown("---")
                            
                            # Display sentiment analysis
                            st.markdown("### 🎭 Sentiment Analysis")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                emoji = "😊" if result['label'] == 'Positive' else "😔" if result['label'] == 'Negative' else "😐"
                                st.metric("Feeling", f"{emoji} {result['label']}")
                            
                            with col2:
                                st.metric("Compound Score", f"{result['compound']:.3f}")
                            
                            with col3:
                                confidence = max(result['scores']['pos'], result['scores']['neg'], result['scores']['neu'])
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            # Show detailed scores
                            with st.expander("📊 Detailed Scores"):
                                st.write(f"**Positive:** {result['scores']['pos']:.3f}")
                                st.write(f"**Negative:** {result['scores']['neg']:.3f}")
                                st.write(f"**Neutral:** {result['scores']['neu']:.3f}")
                            
                            # Save entry
                            sentiment_journaling.sentiment_history.append(result)
                            st.success("💾 Entry saved to your journal history!")
                        else:
                            st.error("❌ Could not process audio. Please speak clearly and try again.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Tip: Make sure to speak clearly into your microphone.")
        
        st.markdown("---")
        
        # File upload as alternative
        st.markdown("### 📁 Or Upload an Audio File")
        uploaded_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a"], key="upload_audio")
        if uploaded_audio:
            st.audio(uploaded_audio)
            if st.button("🔍 Analyze Uploaded Audio", key="analyze_upload"):
                with st.spinner("🎯 Transcribing and analyzing sentiment..."):
                    try:
                        # Save uploaded file temporarily
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_audio.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_audio.read())
                            tmp_path = tmp_file.name
                        
                        # Process the audio
                        result = sentiment_journaling.process_journal_entry(tmp_path)
                        
                        # Clean up
                        os.unlink(tmp_path)
                        
                        if result:
                            st.success("✅ Analysis Complete!")
                            
                            st.markdown("### 📝 Transcription")
                            st.write(f'*"{result["text"]}"*')
                            
                            st.markdown("---")
                            
                            st.markdown("### 🎭 Sentiment Analysis")
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                emoji = "😊" if result['label'] == 'Positive' else "😔" if result['label'] == 'Negative' else "😐"
                                st.metric("Feeling", f"{emoji} {result['label']}")
                            
                            with col2:
                                st.metric("Compound Score", f"{result['compound']:.3f}")
                            
                            with col3:
                                confidence = max(result['scores']['pos'], result['scores']['neg'], result['scores']['neu'])
                                st.metric("Confidence", f"{confidence:.2%}")
                            
                            with st.expander("📊 Detailed Scores"):
                                st.write(f"**Positive:** {result['scores']['pos']:.3f}")
                                st.write(f"**Negative:** {result['scores']['neg']:.3f}")
                                st.write(f"**Neutral:** {result['scores']['neu']:.3f}")
                            
                            sentiment_journaling.sentiment_history.append(result)
                            st.success("💾 Entry saved to your journal history!")
                        else:
                            st.error("❌ Could not process audio. Please ensure clear audio quality.")
                    
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    
    with tab2:
        st.subheader("✍️ Text Journal Entry")
        st.info("Write your thoughts and we'll analyze your sentiment in real-time.")
        
        journal_text = st.text_area("Write your thoughts here:", height=200, key="journal_text_input")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            analyze_button = st.button("🔍 Analyze", key="analyze_text", width='stretch')
        
        with col1:
            if journal_text:
                st.caption(f"📝 {len(journal_text)} characters")
        
        if analyze_button:
            if journal_text.strip():
                with st.spinner("🎯 Analyzing your sentiment in real-time..."):
                    try:
                        # Analyze sentiment
                        result = sentiment_journaling.analyze_sentiment(journal_text)
                        
                        if result:
                            # Store in session state so it persists across reruns
                            st.session_state['sentiment_result'] = result
                            st.session_state['sentiment_text'] = journal_text
                            st.success("✅ Analysis Complete!")
                        else:
                            st.error("❌ Could not analyze sentiment. Please try again.")
                    
                    except Exception as e:
                        st.error(f"❌ Error analyzing sentiment: {str(e)}")
                        st.info("💡 Tip: Make sure your text is in English and not too long.")
            else:
                st.warning("⚠️ Please enter some text to analyze")
        
        # Display results if they exist in session state
        if 'sentiment_result' in st.session_state and st.session_state['sentiment_result']:
            result = st.session_state['sentiment_result']
            journal_text = st.session_state.get('sentiment_text', journal_text)
            
            # Display sentiment analysis with visual feedback
            st.markdown("### 🎭 Sentiment Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                emoji = "😊" if result['label'] == 'Positive' else "😔" if result['label'] == 'Negative' else "😐"
                st.metric("Feeling", f"{emoji} {result['label']}")
            
            with col2:
                st.metric("Compound Score", f"{result['compound']:.3f}", 
                         delta=f"{'Positive' if result['compound'] > 0 else 'Negative' if result['compound'] < 0 else 'Neutral'}")
            
            with col3:
                confidence = max(result['scores']['pos'], result['scores']['neg'], result['scores']['neu'])
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Show detailed breakdown
            st.markdown("---")
            st.markdown("### 📊 Detailed Sentiment Breakdown")
            
            # Create visualization
            breakdown_data = {
                'Positive': result['scores']['pos'],
                'Neutral': result['scores']['neu'],
                'Negative': result['scores']['neg']
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Score': [breakdown_data['Positive'], breakdown_data['Neutral'], breakdown_data['Negative']]
                })
            
            with col2:
                st.write("**Sentiment Scores:**")
                st.write(f"- 😊 **Positive**: {result['scores']['pos']:.1%}")
                st.write(f"- 😐 **Neutral**: {result['scores']['neu']:.1%}")
                st.write(f"- 😔 **Negative**: {result['scores']['neg']:.1%}")
            
            # Save to Firebase
            st.markdown("---")
            st.write("### 💾 Save Your Entry")
            
            save_button = st.button(
                "Save Entry to Firebase",
                width='stretch',
                key="sentiment_save_btn"
            )
            
            if save_button:
                print("\n" + "="*70)
                print("SAVE BUTTON CLICKED!")
                print("="*70)
                
                from integration.firebase_manager import save_sentiment_entry
                
                # Validate result data
                if result is None:
                    st.error("❌ No sentiment analysis result. Cannot save.")
                    print("ERROR: result is None")
                else:
                    print(f"Result: {result}")
                    print(f"Journal text: {journal_text[:50]}...")
                    
                    # Build entry data
                    try:
                        entry_to_save = {
                            'entry_text': journal_text,
                            'sentiment_label': result['label'],
                            'sentiment_compound': float(result['compound']),
                            'sentiment_scores': {
                                'pos': float(result['scores']['pos']),
                                'neu': float(result['scores']['neu']),
                                'neg': float(result['scores']['neg'])
                            },
                            'entry_type': 'text'
                        }
                        
                        print(f"Entry data prepared: {entry_to_save}")
                        
                        # Save to Firebase
                        success = save_sentiment_entry(entry_to_save, user_id="default_user")
                        
                        print(f"Firebase save result: {success}")
                        print("="*70 + "\n")
                        
                        if success:
                            st.success("✅ Entry saved to Firebase!")
                            st.balloons()
                            
                            # Clear the sentiment history cache to auto-refresh trends
                            from integration.firebase_manager import get_sentiment_history
                            try:
                                get_sentiment_history.clear()
                                print("Cache cleared - trends will auto-update")
                            except Exception as cache_error:
                                print(f"Cache clear warning: {cache_error}")
                            
                            # Clear session state for fresh entry
                            if 'sentiment_result' in st.session_state:
                                del st.session_state['sentiment_result']
                            if 'sentiment_text' in st.session_state:
                                del st.session_state['sentiment_text']
                            
                            # Refresh the app to show updated trends
                            st.rerun()
                        else:
                            st.error("❌ Failed to save. Check terminal for details.")
                            
                    except Exception as e:
                        print(f"❌ Exception: {e}")
                        import traceback
                        traceback.print_exc()
                        st.error(f"❌ Error: {str(e)}")
    
    with tab3:
        st.subheader("📊 Sentiment Trends & History")
        
        # Fetch real data from Firebase
        from integration.firebase_manager import get_sentiment_history
        
        # Try multiple user_ids to find data
        user_ids = ["test_user", "default_user"]
        df_history = None
        
        for user_id in user_ids:
            try:
                # Show only recent mood data (last 7 days) for the trend chart
                df_history = get_sentiment_history(user_id=user_id, days=7)
                if df_history is not None and not df_history.empty:
                    break
            except:
                continue
        
        # If no real data, show message
        if df_history is None or df_history.empty:
            st.info("📝 No sentiment entries yet. Save your first entry in the 'Text Journal' tab to see trends!")
        else:
            # Display stats
            col1, col2, col3, col4 = st.columns(4)
            
            avg_sentiment = df_history['sentiment_compound'].mean()
            max_sentiment = df_history['sentiment_compound'].max()
            min_sentiment = df_history['sentiment_compound'].min()
            total_entries = len(df_history)
            
            with col1:
                emoji = "😊" if avg_sentiment > 0 else "😔" if avg_sentiment < 0 else "😐"
                st.metric("Average Sentiment", f"{emoji} {avg_sentiment:.3f}")
            
            with col2:
                st.metric("Best Day", f"😊 {max_sentiment:.3f}")
            
            with col3:
                st.metric("Toughest Day", f"😔 {min_sentiment:.3f}")
            
            with col4:
                st.metric("Total Entries", f"📝 {total_entries}")
            
            st.markdown("---")
            
            # Plot trend
            st.markdown("### 📈 Sentiment Trend (Last 7 Days)")
            
            fig_data = df_history[['timestamp', 'sentiment_compound']].rename(
                columns={'timestamp': 'Date', 'sentiment_compound': 'Sentiment Score'}
            ).copy()
            fig_data['Date'] = pd.to_datetime(fig_data['Date']).dt.date
            fig_data = fig_data.set_index('Date')
            
            # Create line chart
            st.line_chart(fig_data, width='stretch', height=350)
            
            st.markdown("---")
            
            # Show sentiment distribution
            st.markdown("### 🎭 Sentiment Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Count sentiments
                sentiment_counts = df_history['sentiment_label'].value_counts()
                st.bar_chart(sentiment_counts, width='stretch', height=300)
            
            with col2:
                st.write("**Sentiment Breakdown:**")
                for label, count in sentiment_counts.items():
                    percentage = (count / total_entries) * 100
                    emoji = "😊" if label == 'Positive' else "😔" if label == 'Negative' else "😐"
                    st.write(f"{emoji} **{label}**: {count} entries ({percentage:.1f}%)")


def show_nutrition_tracker(nutrition_tracker):
    """Display nutrition tracking module."""
    st.header("🍎 Nutrition Tracker")
    
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Nutrition Log", "Weekly Summary"])
    
    with tab1:
        st.subheader("📸 Analyze Your Meal")
        st.info("Upload a photo of your meal to get nutritional insights and recommendations.")
        
        uploaded_image = st.file_uploader("Choose a meal image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            col1, col2 = st.columns(2)
            
            with col1:
                image_pil = Image.open(uploaded_image)
                st.image(image_pil, caption="Your Meal", width='stretch')
            
            with col2:
                if st.button("Analyze Meal"):
                    with st.spinner("🔍 Analyzing meal image with AI..."):
                        try:
                            # Use cached analyzer for faster performance
                            analyzer = get_food_analyzer()
                            result = analyzer.analyze_food_image(image_pil)
                            
                            if result['success']:
                                # Store result in session state so it persists across button clicks
                                st.session_state['nutrition_analysis_result'] = result
                                st.success("✅ Analysis Complete!")
                            else:
                                st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                                st.info("💡 Try uploading a clearer image of the food")
                        
                        except Exception as e:
                            st.error(f"❌ Error analyzing image: {str(e)}")
                            st.info("💡 Make sure the food classifier model is trained first")
                            st.code("Run: python train_food_classifier.py", language="bash")
            
            # Display analysis results if available in session state
            if 'nutrition_analysis_result' in st.session_state:
                result = st.session_state['nutrition_analysis_result']
                
                # Main classification
                primary_food = result['primary_food'].upper()
                classification = result['classification'].upper()
                confidence = result['confidence']
                health_emoji = result['health_emoji']
                health_score = result['health_score']
                
                st.write(f"### {health_emoji} {classification} Food")
                st.write(f"**Identified:** {primary_food}")
                st.write(f"**Health Score:** {health_score}/10")
                
                # Nutritional info
                st.write("### 📊 Nutritional Information")
                nutrition = result['nutritional_benefits']
                
                if 'nutrients' in nutrition:
                    st.write(f"**Nutrients:** {', '.join(nutrition['nutrients'])}")
                if 'benefits' in nutrition:
                    st.write(f"**Benefits:** {nutrition['benefits']}")
                if 'concerns' in nutrition:
                    st.warning(f"⚠️ **Concerns:** {nutrition['concerns']}")
                if 'calories_per_serving' in nutrition:
                    st.write(f"**Calories (per serving):** {nutrition['calories_per_serving']}")
                
                # Recommendations
                st.write("### 💡 Dietary Recommendations")
                for rec in result['recommendations']:
                    st.write(rec)
                
                # Store analysis in session state for burnout assessment
                st.session_state['last_nutrition_analysis'] = {
                    'food': primary_food,
                    'classification': classification.lower(),
                    'health_score': health_score,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.info("💾 This data will be used for your Burnout Risk Assessment!")
                
                # Log Meal Button
                st.markdown("---")
                col_log1, col_log2 = st.columns([3, 1])
                
                with col_log2:
                    if st.button("✅ Log Meal", key="log_meal_btn", type="primary"):
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info("🔴 LOG MEAL BUTTON CLICKED!")
                        
                        from integration.firebase_manager import save_nutrition_entry
                        
                        # Prepare nutrition info
                        nutritional_info = {
                            'calories': nutrition.get('calories_per_serving', 'N/A'),
                            'protein': nutrition.get('nutrients', ['Unknown'])[0] if nutrition.get('nutrients') else 'Unknown',
                            'carbs': 'See label',
                            'fats': 'See label'
                        }
                        
                        # Get or create user ID
                        user_id = st.session_state.get('user_id', 'default_user')
                        logger.info(f"🔴 Saving meal for user: {user_id}")
                        
                        # Save to Firebase
                        success = save_nutrition_entry(
                            user_id=user_id,
                            food_name=primary_food,
                            confidence=confidence,
                            nutritional_info=nutritional_info,
                            health_score=health_score,
                            health_label=classification.lower()
                        )
                        
                        if success:
                            # Clear the Streamlit cache for nutrition logs to fetch fresh data
                            from integration.firebase_manager import get_nutrition_logs
                            get_nutrition_logs.clear()
                            st.rerun()
                        else:
                            st.error("❌ Failed to log meal. Check connection.")
    
    with tab2:
        st.subheader("📋 Today's Nutrition Log")
        
        from integration.firebase_manager import get_nutrition_logs
        
        # Get user ID from session
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Fetch nutrition logs
        nutrition_logs = get_nutrition_logs(user_id=user_id)
        
        if nutrition_logs is None or nutrition_logs.empty:
            st.info("📱 No meals logged yet. Upload a food image and click '✅ Log Meal' to get started!")
        else:
            st.success(f"✅ Found {len(nutrition_logs)} logged meals")
            
            # Display as dataframe with key columns
            display_df = nutrition_logs[['timestamp', 'food_name', 'health_label', 'health_score', 'confidence']].copy()
            display_df.columns = ['Time', 'Food', 'Classification', 'Health Score', 'Confidence']
            display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{float(x):.1%}")
            display_df['Health Score'] = display_df['Health Score'].apply(lambda x: f"{int(x)}/10")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Show summary stats
            st.markdown("---")
            st.subheader("📊 Today's Stats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                healthy_count = len(nutrition_logs[nutrition_logs['health_label'] == 'healthy'])
                st.metric("Healthy Meals", healthy_count)
            
            with col2:
                avg_score = nutrition_logs['health_score'].mean()
                st.metric("Avg Health Score", f"{avg_score:.1f}/10")
            
            with col3:
                st.metric("Total Logged", len(nutrition_logs))
    
    with tab3:
        st.subheader("📊 Weekly Summary")
        
        from integration.firebase_manager import get_nutrition_logs
        from datetime import timedelta
        
        # Get user ID
        user_id = st.session_state.get('user_id', 'default_user')
        
        # Fetch logs
        nutrition_logs = get_nutrition_logs(user_id=user_id)
        
        if nutrition_logs is None or nutrition_logs.empty:
            st.info("📊 No meals logged yet. Start logging meals to see your weekly summary!")
        else:
            # Filter last 7 days (make timezone-aware for comparison)
            from datetime import timezone
            seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
            recent_logs = nutrition_logs[nutrition_logs['timestamp'] >= seven_days_ago]
            
            if recent_logs.empty:
                st.info("No meals logged in the last 7 days")
            else:
                st.success(f"✅ {len(recent_logs)} meals logged in the last 7 days")
                
                # Calculate stats
                healthy_count = len(recent_logs[recent_logs['health_label'] == 'healthy'])
                moderate_count = len(recent_logs[recent_logs['health_label'] == 'moderate'])
                junk_count = len(recent_logs[recent_logs['health_label'] == 'junk'])
                total_meals = len(recent_logs)
                avg_health_score = recent_logs['health_score'].mean()
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if total_meals > 0:
                        healthy_pct = (healthy_count / total_meals) * 100
                        st.metric("Healthy Meals", f"{healthy_count}/{total_meals}", f"{healthy_pct:.0f}%")
                    else:
                        st.metric("Healthy Meals", "0/0")
                
                with col2:
                    st.metric("Moderate Meals", moderate_count)
                
                with col3:
                    st.metric("Junk Meals", junk_count)
                
                with col4:
                    st.metric("Avg Health Score", f"{avg_health_score:.1f}/10")
                
                st.markdown("---")
                
                # Show food distribution pie chart
                st.subheader("🍽️ Food Category Distribution")
                
                classification_counts = recent_logs['health_label'].value_counts()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.bar_chart(classification_counts)
                
                with col2:
                    st.write("**Breakdown:**")
                    for label, count in classification_counts.items():
                        emoji = "🥗" if label == 'healthy' else "⚖️" if label == 'moderate' else "🍰"
                        pct = (count / total_meals * 100)
                        st.write(f"{emoji} **{label.capitalize()}**: {count} meals ({pct:.0f}%)")
                
                st.markdown("---")
                
                # Show daily trend
                st.subheader("📈 Daily Meal Count Trend")
                
                recent_logs_copy = recent_logs.copy()
                recent_logs_copy['date'] = pd.to_datetime(recent_logs_copy['timestamp']).dt.date
                daily_counts = recent_logs_copy.groupby('date').size()
                
                st.line_chart(daily_counts)


def show_daily_checkin():
    """Display daily check-in module."""
    st.header("✅ Daily Check-in")
    
    st.info("Complete your daily health survey to track your wellness metrics.")
    
    # Display Google Form link button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.link_button(
            "📋 Open Neuro Sync Survey",
            "https://docs.google.com/forms/d/e/1FAIpQLSetVuLaE5OEyKZHJmvt_2x_Jq5uwZKIrtg7KNP2WlsQw8jJHg/viewform",
            width='stretch'
        )
    
    st.markdown("---")
    
    # Display instructions
    st.subheader("📊 How It Works")
    st.markdown("""
    **Step 1:** Spend 2–3 minutes sharing how you're feeling in our survey.
    
    **Sit back:** We'll handle the rest!
    
    **Get insights:** Check out your custom health trends in the Sleep and Burnout tabs.
    """)


def get_wellness_coach_response(user_message: str, burnout_context: dict, chat_history: list) -> str:
    """
    Get response from AI wellness coach using OpenAI.
    Acts as both personal therapist and dietitian.
    """
    try:
        from openai import OpenAI
        
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            return "⚠️ OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
        
        client = OpenAI(api_key=api_key)
        
        # Build context from burnout assessment
        system_prompt = f"""You are a compassionate wellness coach combining expertise in mental health therapy and nutrition science.
You're speaking to someone with the following health profile:

BURNOUT ASSESSMENT:
- Burnout Score: {burnout_context.get('burnout_score', 50):.1f}/100
- Risk Level: {burnout_context.get('risk_level', 'MODERATE')}
- Stress Level: {burnout_context.get('stress_level', 50):.1f}/100
- Nutrition Health: {burnout_context.get('nutrition_health', 50):.1f}/100
- Sleep Health: {burnout_context.get('sleep_health', 50):.1f}/100
- Mental Health: {burnout_context.get('mental_health', 50):.1f}/100

YOUR ROLE:
1. **As a Therapist**: Help them manage workload, stress, and emotional challenges
2. **As a Dietitian**: Suggest nutrition strategies that boost energy and mood
3. **As a Wellness Coach**: Recommend relaxation activities, coping mechanisms, and lifestyle adjustments

GUIDELINES:
- Be empathetic, warm, and non-judgmental
- Ask clarifying questions about their workload, sleep patterns, eating habits
- Provide practical, actionable advice
- Suggest specific relaxation activities (breathing exercises, meditation, walks, etc.)
- Recommend nutrition strategies connected to their current mood/stress level
- If they mention severe distress or suicidal thoughts, encourage professional help
- Keep responses concise but helpful (2-3 paragraphs max)"""
        
        # Build conversation history for OpenAI
        messages = []
        
        # Add system message first
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add previous messages
        for msg in chat_history[-8:]:  # Last 8 messages for context
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Wellness coach error: {str(e)}")
        return f"I'm here to support you, but I encountered a technical issue. Please check your OpenAI API configuration. Error: {str(e)}"



def calculate_burnout_countdown(burnout_score: float) -> dict:
    """
    Calculate time until critical burnout based on burnout risk assessment.
    
    Simple linear mapping: higher burnout score = less time remaining
    
    Args:
        burnout_score: Overall burnout risk (0-100, higher = worse)
    
    Returns:
        dict with days_remaining, hours_remaining, urgency_level, urgency_color
    """
    
    # Direct mapping: burnout score to days remaining
    # Score 100 = burnout now (0 days)
    # Score 0 = 30 days remaining
    days_remaining = max(0.25, (100 - burnout_score) / 3.33)  # Maps 0-100 to 30-0 days
    hours_remaining = days_remaining * 24
    
    # Urgency based on time remaining
    if hours_remaining <= 24:  # 1 day or less
        urgency = "CRITICAL"
        color = "#FF0000"
    elif days_remaining <= 6:  # 5-6 days
        urgency = "INTERMEDIATE"
        color = "#FF6600"
    elif days_remaining <= 20:  # 12-20 days
        urgency = "MODERATE"
        color = "#FFD700"
    else:  # 20+ days
        urgency = "LOW"
        color = "#00AA00"
    
    total_seconds = int(hours_remaining * 3600)
    
    return {
        "days_remaining": days_remaining,
        "hours_remaining": hours_remaining,
        "seconds_remaining": total_seconds,
        "urgency_level": urgency,
        "urgency_color": color
    }


def show_burnout_assessment(composite_assessment):
    """Display burnout assessment module using the composite engine."""
    st.header("⚠️ Burnout Risk Assessment")

    from integration.firebase_manager import (
        get_health_data,
        get_sentiment_history,
        get_nutrition_history,
    )

    # Fetch primary health data
    health_data = get_health_data()

    st.subheader("Real-Time Burnout Prediction")
    st.write("Based on your latest health data:")

    if health_data is None or health_data.empty:
        st.warning("⚠️ No health data available yet. Please start logging your daily metrics in the 'Daily Check-in' module or submit the Google Form.")
        st.info("Once you submit data, your Burnout Risk will be calculated automatically.")
        return

    try:
        # Use the most recent entry as the anchor for form-derived fields
        latest = health_data.sort_values(by="timestamp", ascending=False).iloc[0]
        user_id = latest.get("user_id", "default_user")

        # Gather supporting logs
        sentiment_df = get_sentiment_history(user_id=user_id, days=7)
        nutrition_df = get_nutrition_history(user_id=user_id, days=7)

        sentiment_logs = sentiment_df.to_dict(orient="records") if sentiment_df is not None else []
        nutrition_logs = nutrition_df.to_dict(orient="records") if nutrition_df is not None else []

        sleep_data = {
            "sleep_hours_per_night": latest.get("sleep_hours"),
            "productivity_score": latest.get("productivity_score"),
        }

        exercise_days_per_week = latest.get("exercise_frequency")
        stress_level = latest.get("stress_level")
        workload_rating = latest.get("workload_rating")
        social_isolation = latest.get("social_isolation")

        with st.spinner("🔮 Analyzing your health data..."):
            result = composite_assessment.calculate_from_user_data(
                user_id=user_id,
                sentiment_logs=sentiment_logs,
                nutrition_logs=nutrition_logs,
                sleep_data=sleep_data,
                exercise_days_per_week=exercise_days_per_week,
                stress_level=stress_level,
                workload_rating=workload_rating,
                social_isolation=social_isolation,
                days_lookback=7,
            )

        if not result:
            st.error("Could not compute burnout risk. Please check your data format.")
            return

        # Top-line metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            risk_level = result.get("risk_level", "?")
            risk_color = "🔴" if risk_level == "SEVERE" or risk_level == "HIGH" else "🟡" if risk_level == "MODERATE" else "🟢"
            st.metric("Risk Level", f"{risk_color} {risk_level}")
        with metric_col2:
            st.metric("Burnout Score", f"{result.get('burnout_risk_score', 0):.1f}")
        with metric_col3:
            st.metric("Confidence", f"{result.get('confidence_pct', 0):.1f}%")

        st.markdown("---")
        
        # Dynamic Burnout Countdown
        burnout_score = result.get('burnout_risk_score', 0)
        risk_level = result.get('risk_level', 'LOW')
        
        # Calculate simple countdown based on burnout score
        countdown_info = calculate_burnout_countdown(burnout_score)
        
        st.subheader("⏱️ Burnout Countdown")
        
        # Create visual timer display with columns
        timer_col1, timer_col2 = st.columns([2, 1])
        
        with timer_col1:
            # Countdown showing time until critical burnout
            total_hours = countdown_info['hours_remaining']
            
            # Convert to DD:HH:MM:SS format
            days = int(total_hours / 24)
            hours = int(total_hours % 24)
            remaining_minutes = int((total_hours - int(total_hours)) * 60)
            remaining_seconds = int(((total_hours - int(total_hours)) * 60 - remaining_minutes) * 60)
            
            countdown_str = f"{days:02d}:{hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
            
            # Get urgency styling
            countdown_color = countdown_info['urgency_color']
            urgency_level = countdown_info['urgency_level']
            
            # Set animation based on urgency
            if urgency_level == "CRITICAL":
                flash_style = "animation: pulse 0.8s infinite;"
            elif urgency_level == "INTERMEDIATE":
                flash_style = "animation: pulse 1.2s infinite;"
            else:
                flash_style = ""
            
            # Calculate total seconds for countdown
            total_seconds = countdown_info['seconds_remaining']
            
            countdown_html = f"""
            <style>
                @keyframes pulse {{
                    0%, 100% {{ opacity: 1; }}
                    50% {{ opacity: 0.5; }}
                }}
            </style>
            <div style="background: linear-gradient(135deg, {countdown_color}22 0%, {countdown_color}44 100%); border: 3px solid {countdown_color}; border-radius: 15px; padding: 20px; text-align: center; margin: 10px 0; {flash_style}">
                <div id="countdown-display" style="color: {countdown_color}; font-size: 64px; font-weight: 900; margin: 10px 0; letter-spacing: 3px; font-family: 'Courier New', monospace;">
                    {countdown_str}
                </div>
            </div>
            
            <script>
                (function() {{
                    let totalSeconds = {total_seconds};
                    const countdownDisplay = document.getElementById('countdown-display');
                    
                    function updateCountdown() {{
                        if (totalSeconds <= 0) {{
                            countdownDisplay.textContent = "00:00:00:00";
                            clearInterval(intervalId);
                            return;
                        }}
                        
                        const days = Math.floor(totalSeconds / (24 * 60 * 60));
                        const hours = Math.floor((totalSeconds % (24 * 60 * 60)) / (60 * 60));
                        const minutes = Math.floor((totalSeconds % (60 * 60)) / 60);
                        const seconds = Math.floor(totalSeconds % 60);
                        
                        const daysStr = String(days).padStart(2, '0');
                        const hoursStr = String(hours).padStart(2, '0');
                        const minutesStr = String(minutes).padStart(2, '0');
                        const secondsStr = String(seconds).padStart(2, '0');
                        
                        countdownDisplay.textContent = daysStr + ':' + hoursStr + ':' + minutesStr + ':' + secondsStr;
                        
                        totalSeconds -= 1;
                    }}
                    
                    updateCountdown();
                    const intervalId = setInterval(updateCountdown, 1000);
                }})();
            </script>
            """
            
            # Use st.components for JavaScript execution
            import streamlit.components.v1 as components
            components.html(countdown_html, height=200)
        
        with timer_col2:
            # Risk severity badge - based on actual burnout risk level
            if risk_level == "SEVERE":
                severity_color = "#FF0000"
                severity_label = "CRITICAL"
            elif risk_level == "HIGH":
                severity_color = "#FF6600"
                severity_label = "HIGH"
            elif risk_level == "MODERATE":
                severity_color = "#FFD700"
                severity_label = "MODERATE"
            else:
                severity_color = "#00AA00"
                severity_label = "LOW"
            
            badge_html = f"""
            <div style="background-color: {severity_color}; color: black; padding: 20px; border-radius: 10px; text-align: center; font-weight: bold; font-size: 16px; margin-top: 50px;">
                SEVERITY<br>
                {severity_label}
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
        
        # Severity levels information
        st.markdown("---")
        
        # ========== DYNAMIC PERSONALIZED RECOMMENDATIONS ==========
        st.subheader("💡 Personalized Action Plan")
        
        # Get component scores for context
        component_scores = result.get('components', {})
        nutrition_health = component_scores.get('nutrition_health', 50.0)
        sleep_health = component_scores.get('sleep_fitness_balance', 50.0)
        stress_score = component_scores.get('stress_form', 50.0)
        
        # Use days remaining from countdown for adaptive recommendations
        days_remaining = countdown_info['days_remaining']
        urgency = countdown_info['urgency_level']
        
        # 🔴 CRITICAL LEVEL
        if urgency == "CRITICAL":
            st.error("### 🔴 STATUS: CRITICAL (≤ 1 day)")
            st.write("**Focus: Emergency Mitigation & Radical Rest**")
            st.markdown("""
            1. **Immediate Offload:** Postpone all non-essential meetings and deadlines for the next 48 hours.
            
            2. **Digital Detox:** Enable "Do Not Disturb" mode on all devices; notify emergency contacts only.
            
            3. **Physical Reset:** Cease all cognitively demanding tasks immediately and prioritize 9+ hours of sleep.
            
            4. **Delegate:** Identify two high-stress responsibilities and hand them over to a teammate or peer.
            """)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚨 Generate Emergency Leave Request", use_container_width=True):
                    st.info("Draft: 'Due to health concerns, I need to take immediate leave for recovery. Will return when cleared by healthcare provider.'")
            with col2:
                if st.button("📞 Crisis Resources", use_container_width=True):
                    st.info("Mental Health Crisis Line: 1-800-273-8255 (24/7)")
        
        # 🟠 INTERMEDIATE LEVEL
        elif urgency == "INTERMEDIATE":
            st.warning("### 🟠 STATUS: INTERMEDIATE (5-6 days)")
            st.write("**Focus: Boundary Setting & Targeted Recovery**")
            st.markdown(f"""
            1. **📋 Workload Review:** Audit your upcoming week; cancel or reschedule at least 20% of your commitments.
            
            2. **Hard Cut-off:** Establish a strict "No-Work Zone" after 6:00 PM to allow your nervous system to downregulate.
            
            3. **Sleep Hygiene:** Your Sleep Analysis shows a deficit (Score: {sleep_health:.0f}/100); implement a "screens-off" policy 60 minutes before bed.
            
            4. **Social Support:** Inform your lead or society head that you are currently at capacity to manage expectations.
            """)
        
        # 🟡 MODERATE LEVEL
        elif urgency == "MODERATE":
            st.info("### 🟡 STATUS: MODERATE (12-20 days)")
            st.write("**Focus: Routine Adjustment & Nutritional Support**")
            st.markdown(f"""
            1. **Nutrition Adjustment:** Your logs show patterns (Score: {nutrition_health:.0f}/100). Switch to complex carbohydrates and increase water intake.
            
            2. **Micro-Break Protocol:** Implement the 50/10 rule—50 minutes of focused work followed by 10 minutes of movement.
            
            3. **Environment Swap:** Change your physical workspace to reduce monotony and boost environmental stimulation.
            
            4. **Energy Audit:** Identify "energy-vampire" tasks and batch them together to reduce frequent context switching.
            """)
        
        # 🟢 LOW LEVEL
        else:
            st.success("### 🟢 STATUS: LOW RISK (20+ days)")
            st.write("**Focus: Optimization & Habit Fortification**")
            st.markdown(f"""
            1. **Skill Building:** This is your peak performance zone. Use this time for deep learning or complex coding projects.
            
            2. **Social Sync:** High energy levels are ideal for networking, leading society meetings, or collaborative brainstorming.
            
            3. **Proactive Planning:** Review your schedule for next month and bake in "rest days" now to maintain this runway.
            
            4. **Consistency Check:** Continue your current sleep ({sleep_health:.0f}/100) and nutrition ({nutrition_health:.0f}/100) habits; they are clearly working for you.
            """)
        
        st.markdown("---")

        st.markdown("---")
        
        # ==================== PERSONAL WELLNESS COACH ====================
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; 
                    border-radius: 10px; 
                    color: white; 
                    margin-bottom: 20px;">
            <h4 style="margin: 0; color: white;">🧘‍♀️ Your AI Wellness Partner</h4>
            <p style="margin: 10px 0 0 0; font-size: 14px;">
                Get personalized support combining <strong>therapeutic guidance</strong> and 
                <strong>nutrition expertise</strong>. Share your thoughts, concerns, or ask for advice.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize session state for chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Extract component scores
        component_scores = result.get('components', {})
        stress_score = component_scores.get('stress_form', 50.0)
        nutrition_health = component_scores.get('nutrition_health', 50.0)
        sleep_health = component_scores.get('sleep_fitness_balance', 50.0)
        mental_health = component_scores.get('sentiment_health', 50.0)
        
        # Context for the chatbot
        burnout_context = {
            "burnout_score": burnout_score,
            "risk_level": risk_level,
            "stress_level": stress_score,
            "nutrition_health": nutrition_health,
            "sleep_health": sleep_health,
            "mental_health": mental_health
        }
        
        # Use Streamlit container for proper boxed structure
        with st.container():
            st.markdown("""
            <style>
            .stContainer > div {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                padding: 25px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Display chat history with custom styling
            if len(st.session_state.chat_history) > 0:
                # Header with clear button
                header_col1, header_col2 = st.columns([4, 1])
                with header_col1:
                    st.markdown("**💬 Conversation:**")
                with header_col2:
                    if st.button("🔄 Clear", use_container_width=True, key="clear_chat"):
                        st.session_state.chat_history = []
                        st.rerun()
                
                # Create scrollable chat area
                with st.container():
                    for i, msg in enumerate(st.session_state.chat_history):
                        if msg["role"] == "user":
                            st.chat_message("user", avatar="👤").write(msg["content"])
                        else:
                            st.chat_message("assistant", avatar="🤖").write(msg["content"])
            else:
                st.info("👋 **Welcome!** Start a conversation! Ask me anything about your mental health, stress management, nutrition, or lifestyle.")
            
            # Chat input
            user_input = st.chat_input("💭 Type your message here...")
            
            if user_input:
                # Add user message to history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Get response from chatbot
                with st.spinner("🤔 Your wellness coach is thinking..."):
                    response = get_wellness_coach_response(user_input, burnout_context, st.session_state.chat_history)
                
                # Add assistant response to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # Rerun to display new messages
                st.rerun()
            
            # Quick action buttons
            st.markdown("---")
            st.markdown("**🎯 Quick Prompts:**")
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            
            with btn_col1:
                if st.button("💪 Manage Stress", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user", 
                        "content": "I'm feeling stressed. Can you help me with some stress management techniques?"
                    })
                    with st.spinner("🤔 Your wellness coach is thinking..."):
                        response = get_wellness_coach_response(
                            "I'm feeling stressed. Can you help me with some stress management techniques?",
                            burnout_context,
                            st.session_state.chat_history
                        )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
            
            with btn_col2:
                if st.button("🥗 Meal Suggestions", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "Can you suggest some healthy meals to boost my energy?"
                    })
                    with st.spinner("🤔 Your wellness coach is thinking..."):
                        response = get_wellness_coach_response(
                            "Can you suggest some healthy meals to boost my energy?",
                            burnout_context,
                            st.session_state.chat_history
                        )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
            
            with btn_col3:
                if st.button("😌 Relaxation Tips", use_container_width=True):
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": "What are some relaxation activities I can do to unwind?"
                    })
                    with st.spinner("🤔 Your wellness coach is thinking..."):
                        response = get_wellness_coach_response(
                            "What are some relaxation activities I can do to unwind?",
                            burnout_context,
                            st.session_state.chat_history
                        )
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
        
        # Quick breathing exercise
        st.markdown("---")
        st.subheader("🫁 Quick Breathing Exercise")
        breathing_col1, breathing_col2 = st.columns([2, 1])
        with breathing_col1:
            st.markdown("""
            Take a moment to calm your mind with this simple breathing exercise:
            1. **Breathe In** - Slowly inhale through your nose for 4 counts
            2. **Hold** - Hold your breath for 4 counts
            3. **Breathe Out** - Exhale slowly through your mouth for 4 counts
            
            Repeat this 3-5 times. Feel your stress melt away.
            """)
        with breathing_col2:
            if st.button("🫁 Start Exercise", use_container_width=True):
                # Create a placeholder for the animation
                placeholder = st.empty()
                
                import time
                for cycle in range(5):
                    # Breathe In - Circle expands
                    for size in range(50, 200, 5):
                        with placeholder.container():
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="font-size: 24px; font-weight: bold; color: #667eea; margin-bottom: 20px;">
                                    Cycle {cycle + 1}/5 - Breathe In
                                </div>
                                <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                                    <div style="width: {size}px; height: {size}px; border-radius: 50%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); box-shadow: 0 0 30px rgba(102, 126, 234, 0.6); transition: all 0.1s ease;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        time.sleep(0.05)
                    
                    # Hold - Circle stays at max
                    with placeholder.container():
                        st.markdown("""
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #FF9800; margin-bottom: 20px;">
                                Hold Your Breath
                            </div>
                            <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                                <div style="width: 200px; height: 200px; border-radius: 50%; background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%); box-shadow: 0 0 30px rgba(255, 152, 0, 0.6);"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    time.sleep(1)
                    
                    # Breathe Out - Circle contracts
                    for size in range(200, 50, -5):
                        with placeholder.container():
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <div style="font-size: 24px; font-weight: bold; color: #4CAF50; margin-bottom: 20px;">
                                    Breathe Out
                                </div>
                                <div style="display: flex; justify-content: center; align-items: center; height: 300px;">
                                    <div style="width: {size}px; height: {size}px; border-radius: 50%; background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%); box-shadow: 0 0 30px rgba(76, 175, 80, 0.6); transition: all 0.1s ease;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        time.sleep(0.05)
                
                # Completion
                placeholder.empty()
                st.success("✅ Well done! You've completed the breathing exercise. Feel better?")

    except Exception as e:
        st.error(f"Error analyzing burnout risk: {str(e)}")


def show_analytics():
    """Display analytics and insights."""
    st.header("📊 Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Correlations", "Predictions"])
    
    with tab1:
        st.subheader("Health Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Days Tracked", 25)
        with col2:
            st.metric("Avg Sleep", "7.3 hrs")
        with col3:
            st.metric("Productivity Avg", "76%")
        with col4:
            st.metric("Nutrition Score", "78%")
        
        st.markdown("---")
        
        # Time series chart
        dates = pd.date_range('2024-01-01', periods=25, freq='D')
        metrics = pd.DataFrame({
            'Date': dates,
            'Sleep Hours': np.random.normal(7, 1, 25),
            'Productivity': np.random.normal(75, 10, 25),
        })
        
        st.line_chart(metrics.set_index('Date'))
    
    with tab2:
        st.subheader("Sleep-Productivity Correlation")
        st.info("Analyzing the relationship between sleep quality and daily productivity.")
        
        # Dummy correlation data
        sleep_data = np.random.normal(7, 1, 100)
        productivity_data = sleep_data * 10 + np.random.normal(0, 5, 100)
        
        df_corr = pd.DataFrame({
            'Sleep (hours)': sleep_data,
            'Productivity (%)': productivity_data
        })
        
        st.scatter_chart(df_corr)
        st.success("Correlation: 0.72 (Strong Positive)")
    
    with tab3:
        st.subheader("Burnout Risk Trend")
        
        dates = pd.date_range('2024-01-01', periods=25, freq='D')
        burnout_scores = np.random.normal(35, 15, 25)
        
        df_burnout = pd.DataFrame({
            'Date': dates,
            'Burnout Score': burnout_scores
        })
        
        st.area_chart(df_burnout.set_index('Date'))


def show_sleep_productivity_analysis(df):
    """Display sleep and productivity correlation analysis."""
    st.header("😴 Sleep-Productivity Analysis")
    
    st.info("Analyze the relationship between your sleep duration and daily productivity.")
    
    # Use health_logs data (your actual form submissions)
    if df is None or df.empty:
        st.warning("No health data available yet. Please submit the Google Form to log your daily metrics.")
        return
    
    corr_df = df.copy()

    # Ensure numeric types for plotting
    for col in ["sleep_hours", "productivity_score"]:
        if col in corr_df.columns:
            corr_df[col] = pd.to_numeric(corr_df[col], errors="coerce")

    # Check required columns
    required_cols = ['sleep_hours', 'productivity_score']
    missing_cols = [col for col in required_cols if col not in corr_df.columns]
    
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        st.write("Available columns:", list(corr_df.columns))
        return

    # Display scatter chart
    st.subheader("📊 Sleep vs Productivity Correlation")
    
    scatter_data = corr_df[['sleep_hours', 'productivity_score']].dropna()
    
    if scatter_data.empty:
        st.warning("No valid sleep and productivity data to display.")
        st.write(f"Data rows with sleep_hours: {corr_df['sleep_hours'].notna().sum()}")
        st.write(f"Data rows with productivity_score: {corr_df['productivity_score'].notna().sum()}")
        return

    # Create an enhanced Altair scatter plot with gradient and trend line
    import altair as alt
    
    # Main scatter plot with color gradient based on productivity
    scatter = alt.Chart(scatter_data.reset_index(drop=True)).mark_circle(
        size=200,
        opacity=0.8
    ).encode(
        x=alt.X('sleep_hours:Q', 
                title='Sleep Hours', 
                scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True)),
        y=alt.Y('productivity_score:Q', 
                title='Productivity Score', 
                scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True)),
        color=alt.Color('productivity_score:Q', 
                        scale=alt.Scale(scheme='viridis'),
                        legend=alt.Legend(title='Productivity')),
        tooltip=[
            alt.Tooltip('sleep_hours:Q', title='Sleep Hours', format='.1f'),
            alt.Tooltip('productivity_score:Q', title='Productivity', format='.1f')
        ]
    ).properties(
        width=700,
        height=450,
        title={
            "text": "Sleep vs Productivity Relationship",
            "fontSize": 18,
            "fontWeight": "bold",
            "anchor": "middle"
        }
    ).interactive()
    
    # Add trend line with confidence band
    try:
        z = np.polyfit(scatter_data['sleep_hours'].dropna(), scatter_data['productivity_score'].dropna(), 1)
        p = np.poly1d(z)
        trend_x = np.linspace(scatter_data['sleep_hours'].min(), scatter_data['sleep_hours'].max(), 100)
        trend_y = p(trend_x)
        trend_df = pd.DataFrame({'sleep_hours': trend_x, 'productivity_score': trend_y})
        
        trend_line = alt.Chart(trend_df).mark_line(
            color='#FF6B35',
            size=4,
            opacity=0.9
        ).encode(
            x='sleep_hours:Q',
            y='productivity_score:Q'
        )
        
        # Combine charts
        final_chart = scatter + trend_line
    except:
        final_chart = scatter
    
    st.altair_chart(final_chart, use_container_width=True)
    
    # Metrics dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    avg_sleep = scatter_data['sleep_hours'].mean()
    avg_productivity = scatter_data['productivity_score'].mean()
    correlation = scatter_data['sleep_hours'].corr(scatter_data['productivity_score'])
    
    with col1:
        st.metric("💤 Avg Sleep", f"{avg_sleep:.1f} hrs")
    with col2:
        st.metric("⚡ Avg Productivity", f"{avg_productivity:.1f}/10")
    with col3:
        st.metric("🔗 Correlation", f"{correlation:.2f}")
    with col4:
        st.metric("📊 Data Points", len(scatter_data))
    
    # Insights
    st.markdown("---")
    st.subheader("💡 Insights")
    
    if correlation > 0.5:
        st.success("✅ Strong positive correlation: Better sleep significantly improves your productivity!")
    elif correlation > 0.2:
        st.info("📈 Moderate positive correlation: More sleep tends to boost your productivity.")
    elif correlation > -0.2:
        st.warning("⚠️ Weak correlation: Sleep may not be the main factor affecting your productivity.")
    else:
        st.error("📉 Negative correlation detected: Review your sleep quality and work habits.")
    
    # Additional insight based on average sleep
    if avg_sleep < 6:
        st.warning("⚠️ You're averaging less than 6 hours of sleep. Consider increasing sleep duration.")
    elif avg_sleep > 9:
        st.info("💤 You're getting plenty of sleep. Ensure it's quality sleep for best results.")
    else:
        st.success("✅ Your sleep duration is in a healthy range (6-9 hours).")


def show_fitness_trends(df):
    """Display fitness and exercise trends."""
    st.header("🏃 Fitness Trends")
    
    st.info("Track your fitness progress over time with exercise and activity metrics.")
    
    if df is None or df.empty:
        st.warning("No fitness data available. Start logging your workouts in the 'Daily Check-in' module.")
        return
    
    # Check for exercise frequency column
    if 'exercise_frequency' not in df.columns:
        st.warning("Exercise data not found in your logs.")
        return
    
    # Prepare data
    fitness_df = df[['timestamp', 'exercise_frequency']].copy()
    fitness_df['exercise_frequency'] = pd.to_numeric(fitness_df['exercise_frequency'], errors='coerce')
    fitness_df = fitness_df.dropna().sort_values('timestamp')
    
    if fitness_df.empty:
        st.warning("No valid exercise data to display.")
        return
    
    # Create interactive Altair chart
    import altair as alt
    
    # Line chart with area fill
    base = alt.Chart(fitness_df.reset_index(drop=True)).encode(
        x=alt.X('timestamp:T', title='Date', axis=alt.Axis(format='%b %d')),
        y=alt.Y('exercise_frequency:Q', title='Exercise Days/Week', scale=alt.Scale(zero=True))
    )
    
    line = base.mark_line(color='#FF6B35', size=3, point=True)
    area = base.mark_area(color='#FF6B35', opacity=0.3)
    
    chart = (line + area).properties(
        width=700,
        height=400,
        title='Exercise Frequency Over Time'
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📈 Avg Exercise Days/Week", f"{fitness_df['exercise_frequency'].mean():.1f}")
    with col2:
        st.metric("🔥 Max Streak", f"{fitness_df['exercise_frequency'].max():.0f} days")
    with col3:
        st.metric("📊 Total Entries", len(fitness_df))
    with col4:
        # Calculate trend
        if len(fitness_df) >= 2:
            recent_avg = fitness_df.tail(5)['exercise_frequency'].mean()
            older_avg = fitness_df.head(5)['exercise_frequency'].mean()
            trend = recent_avg - older_avg
            st.metric("📉 Trend", f"{trend:+.1f} days", delta=f"{trend:+.1f}")
        else:
            st.metric("📉 Trend", "N/A")
    
    # Insights
    avg_exercise = fitness_df['exercise_frequency'].mean()
    if avg_exercise >= 5:
        st.success("💪 Excellent! You're maintaining a strong exercise routine!")
    elif avg_exercise >= 3:
        st.info("👍 Good work! You're staying active regularly.")
    else:
        st.warning("⚠️ Consider increasing your exercise frequency for better health.")


if __name__ == "__main__":
    main()
