"""
Daily Check-in Survey Module
Handles integration with Google Forms and Google Sheets for data collection.
Collects sleep consistency, productivity metrics, and activity levels.
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class SurveyManager:
    """Manages Google Forms and Sheets integration for daily check-ins."""
    
    def __init__(self, credentials_path):
        """
        Initialize SurveyManager with Google Sheets credentials.
        
        Args:
            credentials_path (str): Path to service_account.json
        """
        self.credentials_path = credentials_path
        self.client = None
        self.sheet = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate with Google Sheets API."""
        try:
            scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive'
            ]
            credentials = ServiceAccountCredentials.from_json_keyfile_name(
                self.credentials_path, scope
            )
            self.client = gspread.authorize(credentials)
            logger.info("Successfully authenticated with Google Sheets")
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise
    
    def open_sheet(self, sheet_id, worksheet_name="Form Responses 1"):
        """
        Open a specific Google Sheet by ID and worksheet name.
        
        Args:
            sheet_id (str): Google Sheet ID
            worksheet_name (str): Name of the worksheet
        
        Returns:
            Worksheet object or None
        """
        try:
            spreadsheet = self.client.open_by_key(sheet_id)
            self.sheet = spreadsheet.worksheet(worksheet_name)
            logger.info(f"Opened worksheet: {worksheet_name}")
            return self.sheet
        except Exception as e:
            logger.error(f"Failed to open sheet: {e}")
            return None
    
    def get_latest_responses(self, limit=10):
        """
        Retrieve the latest survey responses from Google Sheet.
        
        Args:
            limit (int): Number of latest responses to retrieve
        
        Returns:
            DataFrame with survey responses
        """
        try:
            all_records = self.sheet.get_all_records()
            df = pd.DataFrame(all_records)
            
            # Sort by timestamp or index (assuming last rows are most recent)
            df = df.tail(limit)
            logger.info(f"Retrieved {len(df)} survey responses")
            return df
        except Exception as e:
            logger.error(f"Error retrieving responses: {e}")
            return None
    
    def parse_survey_data(self, df):
        """
        Parse survey data into structured format.
        
        Args:
            df (DataFrame): Raw survey responses
        
        Returns:
            dict: Parsed survey data with metrics
        """
        try:
            parsed_data = {
                'sleep_consistency': self._extract_sleep_data(df),
                'productivity_metrics': self._extract_productivity_data(df),
                'activity_levels': self._extract_activity_data(df),
                'burnout_indicators': self._extract_burnout_data(df),
            }
            return parsed_data
        except Exception as e:
            logger.error(f"Error parsing survey data: {e}")
            return None
    
    def _extract_sleep_data(self, df):
        """Extract sleep-related data from survey."""
        # Placeholder: adjust column names based on actual Google Form
        sleep_cols = [col for col in df.columns if 'sleep' in col.lower()]
        if sleep_cols:
            return df[sleep_cols].to_dict('records')
        return None
    
    def _extract_productivity_data(self, df):
        """Extract productivity-related data from survey."""
        # Placeholder: adjust column names based on actual Google Form
        productivity_cols = [col for col in df.columns if 'productivity' in col.lower() or 'task' in col.lower()]
        if productivity_cols:
            return df[productivity_cols].to_dict('records')
        return None
    
    def _extract_activity_data(self, df):
        """Extract activity/fitness-related data from survey."""
        # Placeholder: adjust column names based on actual Google Form
        activity_cols = [col for col in df.columns if 'activity' in col.lower() or 'fitness' in col.lower() or 'exercise' in col.lower()]
        if activity_cols:
            return df[activity_cols].to_dict('records')
        return None
    
    def _extract_burnout_data(self, df):
        """Extract burnout assessment data from survey."""
        # Placeholder: adjust column names based on actual Google Form
        burnout_cols = [col for col in df.columns if 'burnout' in col.lower() or 'stress' in col.lower() or 'workload' in col.lower()]
        if burnout_cols:
            return df[burnout_cols].to_dict('records')
        return None
    
    def add_response(self, response_data):
        """
        Add a new response to the Google Sheet.
        
        Args:
            response_data (dict): Response data to add
        
        Returns:
            bool: Success status
        """
        try:
            row_values = [str(v) for v in response_data.values()]
            self.sheet.append_row(row_values)
            logger.info("Response added to sheet")
            return True
        except Exception as e:
            logger.error(f"Error adding response: {e}")
            return False


# Example usage function
def example_survey_usage():
    """Example of how to use SurveyManager."""
    survey = SurveyManager('service_account.json')
    survey.open_sheet('your-sheet-id', 'Form Responses 1')
    responses = survey.get_latest_responses(limit=5)
    parsed_data = survey.parse_survey_data(responses)
    print(parsed_data)


if __name__ == '__main__':
    example_survey_usage()
