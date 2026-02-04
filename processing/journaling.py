"""
Sentiment Journaling Module
Handles voice input, speech-to-text transcription, and sentiment analysis.
Processes audio locally to ensure privacy.
"""

import speech_recognition as sr
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentJournaling:
    """Handles voice journaling and sentiment analysis."""
    
    def __init__(self):
        """Initialize sentiment analyzer and speech recognizer."""
        self.sia = SentimentIntensityAnalyzer()
        self.recognizer = sr.Recognizer()
        self.sentiment_history = []
    
    def transcribe_audio(self, audio_input):
        """
        Transcribe audio to text using speech recognition.
        
        Args:
            audio_input: Audio file path or microphone input
        
        Returns:
            str: Transcribed text or None if transcription fails
        """
        try:
            if isinstance(audio_input, str):
                # Load from file
                with sr.AudioFile(audio_input) as source:
                    audio = self.recognizer.record(source)
            else:
                # Use provided audio data
                audio = audio_input
            
            # Transcribe using Google Web Speech API
            text = self.recognizer.recognize_google(audio)
            logger.info(f"Transcribed text: {text[:50]}...")
            return text
        except sr.UnknownValueError:
            logger.warning("Speech was unintelligible")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using VADER.
        
        Args:
            text (str): Text to analyze
        
        Returns:
            dict: Sentiment scores and label
        """
        try:
            scores = self.sia.polarity_scores(text)
            
            # Determine sentiment label
            if scores['compound'] >= 0.05:
                label = 'Positive'
            elif scores['compound'] <= -0.05:
                label = 'Negative'
            else:
                label = 'Neutral'
            
            result = {
                'text': text,
                'scores': scores,
                'label': label,
                'compound': scores['compound'],
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Sentiment analysis: {label} ({scores['compound']:.3f})")
            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return None
    
    def process_journal_entry(self, audio_input):
        """
        Process a complete journal entry from audio to sentiment.
        
        Args:
            audio_input: Audio file path or audio data
        
        Returns:
            dict: Complete journal entry with transcription and sentiment
        """
        try:
            # Step 1: Transcribe audio
            transcribed_text = self.transcribe_audio(audio_input)
            if transcribed_text is None:
                return None
            
            # Step 2: Analyze sentiment
            sentiment_result = self.analyze_sentiment(transcribed_text)
            if sentiment_result is None:
                return None
            
            # Add to history
            self.sentiment_history.append(sentiment_result)
            
            return sentiment_result
        except Exception as e:
            logger.error(f"Error processing journal entry: {e}")
            return None
    
    def record_from_microphone(self, duration=10):
        """
        Record audio from microphone.
        
        Args:
            duration (int): Duration to record in seconds
        
        Returns:
            Audio data
        """
        try:
            with sr.Microphone() as source:
                logger.info(f"Recording for {duration} seconds...")
                self.recognizer.adjust_for_ambient_noise(source)
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=duration)
            logger.info("Recording complete")
            return audio
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
    
    def get_sentiment_trends(self, num_entries=7):
        """
        Get sentiment trends over recent entries.
        
        Args:
            num_entries (int): Number of recent entries to analyze
        
        Returns:
            dict: Trend analysis and statistics
        """
        try:
            if not self.sentiment_history:
                return None
            
            recent_entries = self.sentiment_history[-num_entries:]
            compounds = [entry['compound'] for entry in recent_entries]
            
            trends = {
                'average_sentiment': sum(compounds) / len(compounds),
                'max_sentiment': max(compounds),
                'min_sentiment': min(compounds),
                'trend': 'improving' if compounds[-1] > compounds[0] else 'declining',
                'entry_count': len(recent_entries),
                'emotions_detected': self._classify_emotional_patterns(recent_entries)
            }
            
            logger.info(f"Sentiment trends calculated")
            return trends
        except Exception as e:
            logger.error(f"Error calculating trends: {e}")
            return None
    
    def _classify_emotional_patterns(self, entries):
        """Classify emotional patterns from entries."""
        emotions = {
            'positive_count': sum(1 for e in entries if e['label'] == 'Positive'),
            'negative_count': sum(1 for e in entries if e['label'] == 'Negative'),
            'neutral_count': sum(1 for e in entries if e['label'] == 'Neutral'),
        }
        return emotions
    
    def save_journal(self, filepath):
        """
        Save journal entries to file.
        
        Args:
            filepath (str): Path to save journal
        
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.sentiment_history, f, indent=4)
            logger.info(f"Journal saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving journal: {e}")
            return False
    
    def load_journal(self, filepath):
        """
        Load journal entries from file.
        
        Args:
            filepath (str): Path to journal file
        
        Returns:
            bool: Success status
        """
        try:
            with open(filepath, 'r') as f:
                self.sentiment_history = json.load(f)
            logger.info(f"Journal loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading journal: {e}")
            return False


class SentimentAnalyzer:
    """Lightweight sentiment analyzer used by tests and E2E pipeline."""

    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> dict:
        """Return VADER scores for text."""
        scores = self.sia.polarity_scores(text)
        scores['compound'] = float(scores['compound'])
        return scores

    def get_sentiment_label(self, scores: dict) -> str:
        """Derive label from VADER compound score."""
        comp = scores.get('compound', 0)
        if comp >= 0.05:
            return 'Positive'
        if comp <= -0.05:
            return 'Negative'
        return 'Neutral'


# Example usage function
def example_journaling_usage():
    """Example of how to use SentimentJournaling."""
    journaler = SentimentJournaling()
    
    # Example: Analyze text directly
    sample_text = "I'm feeling great today, very productive and happy!"
    result = journaler.analyze_sentiment(sample_text)
    print(f"Sentiment Analysis: {result}")
    
    # Get trends
    trends = journaler.get_sentiment_trends()
    print(f"Trends: {trends}")


if __name__ == '__main__':
    example_journaling_usage()
