# TalentScout AI Interview Assistant

An advanced AI-powered interview platform that automates the technical screening process for hiring managers and recruiters. The system conducts interactive interviews, evaluates candidate responses, and generates comprehensive evaluation reports.

## ✨ Features

- **Interactive Interview Interface**: Modern, user-friendly interface built with Streamlit
- **Automated Technical Assessments**: Generates relevant questions based on candidate's tech stack
- **AI-Powered Evaluation**: Uses Google's Gemini AI to evaluate responses and provide feedback
- **Comprehensive Reporting**: Generates detailed PDF reports with candidate evaluation
- **Session Management**: Tracks interview progress and maintains conversation history
- **Security & Privacy**: Implements data encryption and secure session handling
- **Responsive Design**: Works seamlessly across desktop and mobile devices

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Google API key with access to Gemini
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/talent_scout.git
   cd talent_scout
   ```

2. **Set up a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root with your configuration:
   ```
   # Required
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Optional (with default values shown)
   DEBUG=True
   ENCRYPTION_KEY=your_encryption_key_here
   ```

   To get a Google API key:
   1. Visit [Google AI Studio](https://makersuite.google.com/)
   2. Sign in with your Google account
   3. Navigate to API Keys section
   4. Create a new API key and copy it to your `.env` file

## 🏃‍♂️ Running the Application

1. Start the Streamlit server:
   ```bash
   streamlit run main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Follow the on-screen instructions to start the interview process

## 🛠️ Project Structure

```
talent_scout/
├── data/                   # Data storage
│   ├── candidates.json     # Candidate information
│   └── consents.json      # Consent records
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── src/                   # Source code
    ├── main.py            # Main application
    ├── interview_logic.py # Interview flow logic
    ├── report_generator.py# PDF report generation
    ├── security_utils.py  # Security utilities
    └── consent_manager.py # Consent management
```

## 🔒 Security & Privacy

- All sensitive data is encrypted at rest
- Candidate data is stored securely with proper access controls
- Audit logging for all system activities
- Configurable data retention policies

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to contribute to this project.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For support or questions, please contact support@talent-scout.ai

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