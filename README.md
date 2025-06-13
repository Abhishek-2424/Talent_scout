# TalentScout AI Hiring Assistant

An intelligent chatbot that helps with the initial screening of technical candidates for TalentScout recruitment agency.

## Features

- **Interactive Chat Interface**: Clean and intuitive UI built with Streamlit
- **Candidate Information Collection**: Gathers essential details like name, contact info, and experience
- **Tech Stack Analysis**: Identifies and analyzes the candidate's technical skills
- **Automated Technical Questions**: Generates relevant technical questions based on the candidate's tech stack
- **Context-Aware Conversations**: Maintains conversation context for a natural flow
- **Responsive Design**: Works on both desktop and mobile devices

## Prerequisites

- Python 3.9 or higher
- Google API key with access to Gemini
- pip (Python package manager)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Talent_scout
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
   
   To get a Google API key with Gemini access:
   1. Go to [Google AI Studio](https://makersuite.google.com/)
   2. Sign in with your Google account
   3. Click on "Get API Key" and follow the instructions
   4. Copy the API key and add it to your `.env` file

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Access the application**
   Open your web browser and navigate to `http://localhost:8501`

## Usage

1. **Start the Interview**
   - Fill in your basic information
   - List your tech stack and desired positions
   - Click "Start Interview"

2. **Chat with the AI**
   - Answer the AI's questions about your experience and skills
   - The AI will ask follow-up questions based on your responses

3. **Technical Assessment**
   - Answer the generated technical questions
   - These are tailored to your specified tech stack

4. **Complete the Interview**
   - The AI will thank you for your time
   - You can start a new interview if needed

## Project Structure

- `main.py`: Main application file with Streamlit UI and OpenAI integration
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (not included in version control)
- `README.md`: This file

## Customization

You can customize the application by:

1. Modifying the system prompt in `main.py` to change the AI's behavior
2. Adding more fields to the candidate information form
3. Adjusting the technical question generation logic
4. Customizing the UI theme and styling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for the web framework
- [Google AI](https://ai.google.dev/) for the Gemini language model API
- [Python](https://www.python.org/) for making it all possible