# Configure Streamlit page first - this must be the first Streamlit command
import streamlit as st

# Debug mode - set to False in production
DEBUG = True

st.set_page_config(
    page_title="TalentScout AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Security configuration
SECURITY_CONFIG = {
    'require_consent': True,  # Set to False to disable consent requirement
    'encrypt_sensitive_data': True,  # Encrypt candidate data
    'audit_logging': True  # Enable audit logging
}

import os
import re
import time
import json
import uuid
import logging
import hashlib
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Callable

import google.generativeai as genai
from dotenv import load_dotenv

# Import security components
from security_utils import security_manager, data_manager
from consent_manager import consent_manager

# Import security and data management
from security_utils import security_manager, data_manager, SecurityManager, DataManager
from consent_manager import consent_manager, ConsentManager

# Import interview logic
from interview_logic import (
    InterviewSession, InterviewStage, Question,
    generate_question_bank, check_conversation_end,
    generate_follow_up_question, generate_evaluation,
    generate_interview_report, FALLBACK_RESPONSES
)

# Import report generation
from report_generator import generate_pdf_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
class SecurityConfig:
    """Application security configuration."""
    SESSION_TIMEOUT = timedelta(minutes=30)  # Session timeout
    MAX_LOGIN_ATTEMPTS = 5  # Maximum failed login attempts
    PASSWORD_MIN_LENGTH = 12  # Minimum password length
    PASSWORD_COMPLEXITY = {
        'min_length': 12,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_digits': True,
        'require_special': True,
        'special_chars': '!@#$%^&*()_+-=[]{}|;:,.<>?'
    }

# Security utilities
def hash_sensitive_data(data: str, salt: Optional[bytes] = None) -> tuple[str, bytes]:
    """Hash sensitive data with salt."""
    if salt is None:
        salt = os.urandom(16)  # 128-bit salt
    
    if not data:
        return "", salt
        
    # Use PBKDF2 with 100,000 iterations
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        data.encode('utf-8'),
        salt,
        100000
    )
    return hashed.hex(), salt

def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    """Validate password meets complexity requirements."""
    errors = []
    
    if len(password) < SecurityConfig.PASSWORD_COMPLEXITY['min_length']:
        errors.append(f"Password must be at least {SecurityConfig.PASSWORD_COMPLEXITY['min_length']} characters long")
    
    if SecurityConfig.PASSWORD_COMPLEXITY['require_uppercase'] and not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter")
    
    if SecurityConfig.PASSWORD_COMPLEXITY['require_lowercase'] and not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter")
    
    if SecurityConfig.PASSWORD_COMPLEXITY['require_digits'] and not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit")
    
    if SecurityConfig.PASSWORD_COMPLEXITY['require_special'] and \
       not any(c in SecurityConfig.PASSWORD_COMPLEXITY['special_chars'] for c in password):
        errors.append(f"Password must contain at least one special character: {SecurityConfig.PASSWORD_COMPLEXITY['special_chars']}")
    
    return len(errors) == 0, errors

def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for a route."""
    def wrapper(*args, **kwargs):
        if not st.session_state.get('authenticated', False):
            st.warning("Please log in to access this page.")
            show_login_form()
            return
        return func(*args, **kwargs)
    return wrapper

def audit_log(action: str, resource: str, status: str = "success", details: Optional[Dict] = None) -> None:
    """Log security-relevant events."""
    try:
        security_manager.log_audit_event(
            action=action,
            resource=resource,
            status=status,
            details=details or {}
        )
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")

# Initialize security components
def initialize_security() -> bool:
    """Initialize security components and configurations."""
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize security manager
        global security_manager, data_manager, consent_manager
        security_manager = SecurityManager()
        data_manager = DataManager()
        consent_manager = ConsentManager()
        
        # Log security initialization
        audit_log("security_init", "application", "success")
        logger.info("Security components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize security components: {e}")
        return False

# Load environment variables
load_dotenv()

# Initialize security components
security_manager.log_audit_event("app_start", "application", "info", {"message": "Application started"})

# Authentication functions
def show_login_form() -> None:
    """Display the login form."""
    st.title("🔒 Login to TalentScout AI")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        remember_me = st.checkbox("Remember me")
        
        if st.form_submit_button("Login"):
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.last_activity = datetime.now(timezone.utc)
                
                # Set session cookie if remember me is checked
                if remember_me:
                    st.session_state.persist = True
                
                audit_log("login_success", f"user:{username}", "success")
                st.rerun()
            else:
                audit_log("login_failed", f"user:{username}", "failed", {"reason": "invalid_credentials"})
                st.error("Invalid username or password")
    
    # Add password reset link
    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <a href='#reset-password' style='color: #4CAF50; text-decoration: none;'>Forgot password?</a>
    </div>
    """, unsafe_allow_html=True)

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password."""
    try:
        # In a real application, you would verify against a database
        # This is a simplified example
        if not username or not password:
            return False
            
        # Check for brute force attempts
        failed_attempts = st.session_state.get('failed_attempts', 0)
        if failed_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            lockout_time = st.session_state.get('lockout_until')
            if lockout_time and datetime.utcnow() < lockout_time:
                remaining = (lockout_time - datetime.utcnow()).seconds // 60 + 1
                st.error(f"Account locked. Try again in {remaining} minutes.")
                return False
            else:
                # Reset lockout if time has passed
                st.session_state.failed_attempts = 0
                if 'lockout_until' in st.session_state:
                    del st.session_state.lockout_until
        
        # TODO: Replace with actual user authentication logic
        # For demo purposes, using a simple check
        if username == os.getenv('ADMIN_USERNAME') and password == os.getenv('ADMIN_PASSWORD'):
            return True
            
        # Log failed attempt
        st.session_state.failed_attempts = failed_attempts + 1
        
        # Lock account after max attempts
        if st.session_state.failed_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            st.session_state.lockout_until = datetime.utcnow() + timedelta(minutes=15)
            st.error("Too many failed attempts. Account locked for 15 minutes.")
            audit_log("account_locked", f"user:{username}", "security", {
                "reason": "too_many_failed_attempts",
                "attempts": st.session_state.failed_attempts
            })
        
        return False
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return False

def logout() -> None:
    """Log out the current user."""
    if 'username' in st.session_state:
        username = st.session_state.username
        audit_log("logout", f"user:{username}", "success")
    
    # Clear session state
    for key in list(st.session_state.keys()):
        if key != 'persist':  # Keep persist flag for remember me functionality
            del st.session_state[key]
    
    st.rerun()

def check_session_timeout() -> None:
    """Check if the current session has timed out."""
    if 'last_activity' in st.session_state:
        last_activity = st.session_state.last_activity
        if isinstance(last_activity, str):
            last_activity = datetime.fromisoformat(last_activity)
            
        if datetime.now(timezone.utc) - last_activity > SecurityConfig.SESSION_TIMEOUT:
            if 'username' in st.session_state:
                username = st.session_state.username
                audit_log("session_timeout", f"user:{username}", "security")
            logout()
            st.warning("Your session has timed out. Please log in again.")
            st.stop()
    
    # Update last activity time with timezone-aware datetime
    st.session_state.last_activity = datetime.now(timezone.utc)

# Initialize security components
initialize_security()

# Check session timeout on each page load
check_session_timeout()

# Conversation ending keywords
CONVERSATION_END_KEYWORDS = {
    'goodbye', 'bye', 'exit', 'quit', 'end', 'finish',
    'that\'s all', 'i\'m done', 'no more questions',
    'thank you', 'thanks', 'appreciate it'
}

# Fallback responses
FALLBACK_RESPONSES = [
    "I'm here to help with your interview. Could you please rephrase your question?",
    "I want to make sure I understand. Could you provide more details?",
    "I'm focused on your technical interview. Let's get back to the questions.",
    "I'm not sure I follow. Could you try asking that differently?"
]

def add_to_history(role: str, content: str, is_typing: bool = False) -> None:
    """Add a message to the conversation history with optional typing indicator"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Check for conversation end keywords
    if role == "user" and check_conversation_end(content):
        content = "I'd like to end the interview now."
    
    st.session_state.conversation_history.append({
        "role": role, 
        "content": content,
        "is_typing": is_typing,
        "timestamp": datetime.now().isoformat()
    })

def should_end_conversation(message: str) -> bool:
    """Check if the message indicates the conversation should end"""
    return any(keyword in message.lower() for keyword in CONVERSATION_END_KEYWORDS)

def get_fallback_response() -> str:
    """Get a random fallback response, ensuring we don't repeat the same one"""
    if 'last_fallback' not in st.session_state:
        st.session_state.last_fallback = -1
    
    available = [i for i in range(len(FALLBACK_RESPONSES)) if i != st.session_state.last_fallback]
    if not available:
        available = list(range(len(FALLBACK_RESPONSES)))
    
    idx = available[0]  # Just get the first available for now
    st.session_state.last_fallback = idx
    return FALLBACK_RESPONSES[idx]

def initialize_ai():
    """Initialize the AI model and configuration"""
    # First try to get from Streamlit secrets (for production)
    if 'GOOGLE_API_KEY' in st.secrets:
        api_key = st.secrets['GOOGLE_API_KEY']
    else:
        # Fall back to environment variables (for local development)
        load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error("""
            Error: GOOGLE_API_KEY not found.\n\n
            For local development, create a .env file with GOOGLE_API_KEY.\n
            For production, add GOOGLE_API_KEY to Streamlit Cloud secrets.
        """)
        st.stop()

    api_key = str(api_key).strip('\"\'')

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.95,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        return model, generation_config
    except Exception as e:
        st.error(f"Failed to initialize AI: {str(e)}")
        st.stop()

# Initialize AI and session state
model, generation_config = None, None

try:
    model, generation_config = initialize_ai()
    logger.info("AI model and configuration initialized successfully")
except Exception as e:
    logger.error(f"Error initializing AI: {str(e)}", exc_info=True)
    st.error("Error initializing AI. Please check the logs for more details.")
    st.stop()

def generate_questions(candidate_info: dict) -> List[dict]:
    """Generate interview questions based on candidate info"""
    try:
        if not candidate_info:
            logger.warning("No candidate info provided for question generation")
            return []
            
        # Extract tech stack and experience
        tech_stack = candidate_info.get('tech_stack', '')
        if isinstance(tech_stack, str):
            tech_stack = [t.strip() for t in tech_stack.split(',') if t.strip()]
            
        years_exp = int(candidate_info.get('experience', 1))
        
        # Generate questions using the question bank
        questions = generate_question_bank(tech_stack, years_exp)
        
        # Convert Question objects to dictionaries for serialization
        return [{
            'text': q.text,
            'category': q.category,
            'difficulty': q.difficulty,
            'weight': q.weight
        } for q in questions]
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        return []

# Initialize session state if it doesn't exist
if 'session' not in st.session_state:
    st.session_state.session = InterviewSession()
    st.session_state.current_question_idx = 0
    st.session_state.conversation_history = []
    st.session_state.last_fallback = -1
    st.session_state.interview_started = False
    st.session_state.current_stage = InterviewStage.GREETING

# Get the current session
session = st.session_state.session

def evaluate_response(question: Question, answer: str, max_retries: int = 2) -> Dict[str, Any]:
    """
    Evaluate a candidate's response to a question using AI with rate limit handling.
    
    Args:
        question: The question that was asked
        answer: The candidate's answer
        max_retries: Maximum number of retry attempts for rate limits
        
    Returns:
        Dict containing 'score' (0-100) and 'feedback' (str)
    """
    if not answer.strip():
        return {'score': 0, 'feedback': 'No answer provided.'}
    
    def calculate_fallback_score(text: str, question_type: str) -> int:
        """Calculate a fallback score based on answer length and question type"""
        base_score = min(100, len(text) * 2)  # Base score on length (max 100)
        
        # Adjust based on question type
        if question_type.lower() in ['behavioral', 'problem_solving']:
            # Look for structured responses (bullets, numbers, paragraphs)
            structure_indicators = sum(1 for c in ['. ', '\n', '•', '* ', '1.'] if c in text)
            base_score = min(100, base_score + (structure_indicators * 5))
            
        return max(10, min(100, base_score))  # Ensure score is between 10-100
    
    def generate_feedback(score: int, question: Question) -> str:
        """Generate meaningful feedback based on the score"""
        if score >= 80:
            return ("Excellent response! Your answer demonstrates strong understanding of the topic. "
                   "You provided clear, detailed information that directly addresses the question.")
        elif score >= 60:
            return ("Good response. You've covered the main points, but could benefit from "
                   "more specific examples or deeper technical details.")
        elif score >= 40:
            return ("Fair response. You've addressed the question but your answer could be "
                   "more comprehensive. Consider providing more details or examples.")
        else:
            return ("Your response needs improvement. Try to provide more specific details, "
                   "examples, or clarify your thoughts further.")
    
    # First, try to use the AI evaluation
    for attempt in range(max_retries + 1):
        try:
            # Prepare the prompt for the AI
            prompt = """
            Evaluate the following interview question and answer. Provide a score from 0-100 and detailed feedback.
            
            Question: {question.text}
            Category: {question.category}
            Difficulty: {question.difficulty}
            
            Candidate's Answer: {answer}
            
            Please provide:
            1. A score from 0-100 based on accuracy, completeness, and relevance
            2. Detailed feedback on what was good and what could be improved
            
            Format your response as a JSON object with these keys: 'score' and 'feedback'
            """.format(question=question, answer=answer)
            
            # Call the AI model
            response = model.generate_content(prompt)
            
            # Parse the response
            try:
                import json
                result = json.loads(response.text)
                
                # Validate the response
                if not isinstance(result, dict) or 'score' not in result or 'feedback' not in result:
                    raise ValueError("Invalid response format from AI")
                
                # Ensure score is within 0-100 range
                score = max(0, min(100, float(result['score'])))
                
                return {
                    'score': score,
                    'feedback': result['feedback'],
                    'source': 'ai'
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # If JSON parsing fails, use fallback
                raise Exception(f"Failed to parse AI response: {str(e)}")
                
        except Exception as e:
            if '429' in str(e) and attempt < max_retries:
                # Wait before retrying (exponential backoff)
                import time
                wait_time = (2 ** attempt) + 1  # 2s, 4s, 8s, etc.
                time.sleep(wait_time)
                continue
                
            # If we've exhausted retries or it's a different error, use fallback
            if attempt == max_retries or '429' not in str(e):
                # Calculate fallback score
                score = calculate_fallback_score(answer, question.category)
                return {
                    'score': score,
                    'feedback': generate_feedback(score, question),
                    'source': 'fallback',
                    'error': str(e)
                }
    
    # This should never be reached, but just in case
    score = 50  # Default score if all else fails
    return {
        'score': score,
        'feedback': generate_feedback(score, question),
        'source': 'fallback',
        'error': 'Unexpected error in evaluation'
    }

def handle_answer_submission(answer: str) -> None:
    """Handle submission of an answer"""
    if not answer.strip():
        st.warning("Please provide an answer before submitting.")
        return
    
    current_question = session.questions[st.session_state.current_question_idx]
    current_question.answer = answer
    
    # Evaluate the response
    evaluation = evaluate_response(current_question, answer)
    current_question.score = evaluation['score']
    current_question.feedback = evaluation['feedback']
    
    # Add to conversation history
    add_to_history("user", answer)
    add_to_history("assistant", f"Thank you for your answer. Here's some feedback: {evaluation['feedback']}")

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_phone(phone: str) -> bool:
    """Validate phone number format (supports international numbers)"""
    # Remove all non-digit characters except leading +
    cleaned = re.sub(r'[^\d+]', '', phone)
    # Check if it's a valid international number (with country code)
    return len(cleaned) >= 10 and (cleaned.startswith('+') or cleaned.startswith('1') or len(cleaned) == 10)

def process_candidate_info(form_data: dict, consent_given: bool) -> None:
    """Process the candidate information form submission.
    
    Args:
        form_data: Dictionary containing the form data
        consent_given: Whether the user gave consent
    """
    try:
        # Generate a unique candidate ID
        candidate_id = str(uuid.uuid4())
        
        # Create a clean copy of form data with metadata
        candidate_info = {
            **form_data,
            'candidate_id': candidate_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'consent_given': consent_given
        }
        
        # Save data with security manager if encryption is enabled
        if SECURITY_CONFIG['encrypt_sensitive_data']:
            if not data_manager.save_candidate_data(candidate_info):
                raise Exception("Failed to save candidate data")
        
        # Record consent if required
        if SECURITY_CONFIG['require_consent'] and consent_given:
            try:
                consent_manager.record_consent(
                    user_id=candidate_id,
                    consent_type="data_processing",
                    granted=True,
                    purpose="interview_processing"
                )
            except Exception as consent_error:
                logger.error(f"Error recording consent: {consent_error}", exc_info=True)
                # Continue even if consent recording fails
        
        # Initialize interview session
        interview_session = InterviewSession()
        interview_session.candidate_info = candidate_info
        interview_session.current_stage = InterviewStage.TECHNICAL_ASSESSMENT  # Start with technical assessment
        
        # Initialize questions list if it doesn't exist
        if not hasattr(interview_session, 'questions'):
            interview_session.questions = []
        
        # Update session state atomically
        st.session_state.update({
            'session': interview_session,
            'candidate_info': candidate_info,
            'conversation_history': [],
            'current_question_idx': 0,
            'interview_started': True,
            'current_stage': InterviewStage.TECHNICAL_ASSESSMENT,  # Set the stage to technical assessment
            'last_activity': datetime.now(timezone.utc).isoformat(),
            'show_greeting': False  # Ensure we don't show greeting page again
        })
        
        # Generate questions for the interview based on candidate info
        try:
            # First convert tech_stack to a list if it's a string
            tech_stack = form_data.get('tech_stack', '')
            if isinstance(tech_stack, str):
                tech_stack = [t.strip() for t in tech_stack.split(',') if t.strip()]
                
            # Generate questions using the new function
            questions_data = generate_questions(form_data)
            
            if questions_data:
                # Convert question dictionaries back to Question objects
                questions = []
                for q_data in questions_data:
                    q = Question(
                        text=q_data['text'],
                        category=q_data['category'],
                        difficulty=q_data.get('difficulty', 'medium'),
                        weight=q_data.get('weight', 1.0)
                    )
                    questions.append(q)
                
                interview_session.questions = questions
                interview_session.candidate_info = candidate_info
                st.session_state.session = interview_session
                
                logger.info(f"Generated {len(questions)} questions for the interview")
            else:
                raise Exception("No questions were generated")
                
        except Exception as e:
            error_msg = f"Error generating questions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error("Failed to generate interview questions. Please try again.")
            
            # Log the error to audit log if available
            if 'audit_logging' in SECURITY_CONFIG and SECURITY_CONFIG['audit_logging']:
                security_manager.log_audit_event(
                    "question_generation_failed",
                    f"candidate:{candidate_id}",
                    "error",
                    {
                        'error': str(e),
                        'candidate_email': candidate_info.get('email', 'unknown')
                    }
                )
            return
        
        # Log interview start
        if SECURITY_CONFIG['audit_logging']:
            security_manager.log_audit_event(
                "interview_started", 
                f"candidate:{candidate_id}", 
                "info",
                {
                    'candidate_email': candidate_info.get('email', 'unknown'),
                    'position': candidate_info.get('position', 'unspecified'),
                    'stage': 'TECHNICAL_ASSESSMENT'
                }
            )
        
        logger.info(f"Interview started for candidate: {candidate_info.get('email')}")
        
        # Force a rerun to update the UI
        st.rerun()
            
    except Exception as e:
        error_msg = f"Error processing candidate info: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Log the error to audit log if available
        if 'audit_logging' in SECURITY_CONFIG and SECURITY_CONFIG['audit_logging']:
            security_manager.log_audit_event(
                "candidate_processing_error",
                f"candidate:{candidate_id if 'candidate_id' in locals() else 'unknown'}",
                "error",
                {
                    'error': str(e),
                    'candidate_email': form_data.get('email', 'unknown')
                }
            )
        
        # Re-raise the exception to be handled by the caller
        raise

def show_greeting_page() -> None:
    """Display the greeting and candidate info form"""
    st.title("👋 Welcome to TalentScout AI")
    st.markdown("""
    Thank you for participating in our technical screening process. 
    This AI-powered interview will help us understand your skills and experience better.
    """)
    
    # Ensure form data is initialized
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {
            'name': '',
            'email': '',
            'phone': '',
            'position': '',
            'experience': 0,
            'tech_stack': '',
            'consent_given': False
        }
    
    # Create form with a unique key
    with st.form(key="candidate_form"):
        st.subheader("Candidate Information")
        
        # Form fields with current values from session state
        name = st.text_input("Full Name*", value=st.session_state.form_data['name'])
        email = st.text_input("Email*", value=st.session_state.form_data['email'])
        phone = st.text_input("Phone Number", value=st.session_state.form_data['phone'])
        
        # Position selection
        position_options = ["", "Software Engineer", "AI Engineer", "Data Scientist", 
                          "DevOps Engineer", "Frontend Developer", "Backend Developer"]
        
        # Get current position index safely
        current_position = st.session_state.form_data.get('position', '')
        position_index = position_options.index(current_position) if current_position in position_options else 0
        
        position = st.selectbox(
            "Position Applying For*", 
            position_options,
            index=position_index
        )
        
        experience = st.number_input(
            "Years of Experience*", 
            min_value=0, 
            max_value=50, 
            step=1,
            value=st.session_state.form_data.get('experience', 0)
        )
        
        tech_stack = st.text_area(
            "Technologies You're Comfortable With (comma-separated)*",
            value=st.session_state.form_data.get('tech_stack', '')
        )
        
        # Consent checkbox - only show if required by config
        consent_given = st.checkbox(
            "I consent to the processing of my data for interview purposes*",
            value=st.session_state.form_data.get('consent_given', False),
            disabled=not SECURITY_CONFIG['require_consent']
        )
        
        # Form submission button
        submitted = st.form_submit_button("Start Interview")
        
        if submitted:
            # Validate required fields first
            if not all([name.strip(), email.strip(), position, tech_stack.strip()]):
                st.error("Please fill in all required fields (marked with *)")
                st.stop()
                
            if not is_valid_email(email):
                st.error("Please enter a valid email address")
                st.stop()
                
            if phone and not is_valid_phone(phone):
                st.error("Please enter a valid phone number")
                st.stop()
                
            if SECURITY_CONFIG['require_consent'] and not consent_given:
                st.error("You must give consent to proceed with the interview")
                st.stop()
            
            try:
                # Create form data dictionary
                form_data = {
                    'name': name.strip(),
                    'email': email.strip(),
                    'phone': phone.strip() if phone else '',
                    'position': position,
                    'experience': experience,
                    'tech_stack': tech_stack.strip(),
                    'consent_given': consent_given
                }
                
                # Process the form submission
                process_candidate_info(form_data, consent_given)
                
                # Update session state to move to the interview stage
                st.session_state.form_data = form_data
                st.session_state.interview_started = True
                
                # Force a rerun to update the UI
                st.rerun()
                
            except Exception as e:
                st.error(f"An error occurred while processing your information. Please try again.")
                logger.error(f"Error in show_greeting_page: {str(e)}", exc_info=True)
                # Don't return here - let the form be shown again
                
            if SECURITY_CONFIG['require_consent'] and not consent_given:
                st.error("You must give consent to proceed with the interview")
                return
                
            # If we get here, validation passed
            form_data = {
                'name': name,
                'email': email,
                'phone': phone,
                'position': position,
                'experience': experience,
                'tech_stack': tech_stack
            }
            
            # Store form data in session state
            st.session_state.form_data = form_data
            
            # Process the form submission
            try:
                process_candidate_info(form_data, consent_given)
                st.rerun()  # Rerun to update the UI with the new state
            except Exception as e:
                st.error(f"Failed to process your information: {str(e)}")
                logger.error(f"Error processing candidate info: {str(e)}", exc_info=True)
            
            # Generate a unique candidate ID
            candidate_id = str(uuid.uuid4())
            
            # Prepare candidate data with timestamp
            candidate_data = {
                'candidate_id': candidate_id,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'name': name,
                'email': email,
                'phone': phone,
                'position': position,
                'experience': experience,
                'tech_stack': tech_stack
            }
            
            # Save data with security manager if encryption is enabled
            try:
                if SECURITY_CONFIG['encrypt_sensitive_data']:
                    if not data_manager.save_candidate_data(candidate_data):
                        raise Exception("Failed to save candidate data")
                    
                if SECURITY_CONFIG['audit_logging']:
                    security_manager.log_audit_event(
                        "candidate_data_collected", 
                        f"candidate:{candidate_id}", 
                        "success",
                        {"position": position, "experience": experience}
                    )
                
                # Record consent if required
                if SECURITY_CONFIG['require_consent']:
                    try:
                        consent_manager.record_consent(
                            user_id=candidate_id,
                            consent_type="data_processing",
                            granted=True,
                            purpose="interview_processing"
                        )
                    except Exception as consent_error:
                        logger.error(f"Error recording consent: {consent_error}", exc_info=True)
                        # Continue even if consent recording fails
                
                # Initialize interview session
                try:
                    logger.info("Initializing interview session...")
                    
                    # Create a copy of candidate data to avoid reference issues
                    candidate_info = candidate_data.copy()
                    
                    # Debug: Log candidate info
                    logger.info(f"Candidate info: {candidate_info}")
                    
                    # Create interview session
                    try:
                        interview_session = InterviewSession()
                        logger.info("InterviewSession created successfully")
                        interview_session.current_stage = InterviewStage.GREETING
                        logger.info(f"Set current_stage to {InterviewStage.GREETING}")
                        
                        # Store in session state - use st.session_state.update for atomic updates
                        st.session_state.update({
                            'candidate_info': candidate_info,
                            'session': interview_session,
                            'conversation_history': [],
                            'interview_started': True
                        })
                        
                        # Log interview start
                        if SECURITY_CONFIG['audit_logging']:
                            security_manager.log_audit_event(
                                "interview_started", 
                                f"candidate:{candidate_id}", 
                                "info"
                            )
                        
                        logger.info("Session initialization complete, rerunning...")
                        st.rerun()
                        
                    except Exception as create_error:
                        logger.error(f"Error creating InterviewSession: {str(create_error)}", exc_info=True)
                        raise Exception(f"Failed to create interview session: {str(create_error)}")
                    
                except Exception as session_error:
                    error_msg = f"Session initialization error: {str(session_error)}"
                    logger.error(error_msg, exc_info=True)
                    st.error("Failed to initialize interview session. Please check the logs for details.")
                    if SECURITY_CONFIG['audit_logging']:
                        security_manager.log_audit_event(
                            "session_init_failed",
                            f"candidate:{candidate_id}",
                            "error",
                            {"error": str(session_error), "traceback": traceback.format_exc()}
                        )
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Candidate data processing failed: {error_msg}", exc_info=True)
                if SECURITY_CONFIG['audit_logging']:
                    security_manager.log_audit_event(
                        "candidate_data_save_failed", 
                        f"candidate:{candidate_id}", 
                        "error", 
                        {"error": str(e), "traceback": traceback.format_exc()}
                    )
                st.error(f"An error occurred: {error_msg}. Please check the logs for more details.")

def initialize_session_state():
    """Initialize all required session state variables if they don't exist."""
    defaults = {
        'session': None,  # Will be initialized when needed
        'conversation_history': [],
        'current_question_idx': 0,
        'awaiting_followup': False,
        'followup_count': 0,
        'candidate_info': {},
        'interview_started': False,
        'current_stage': InterviewStage.GREETING,
        'last_activity': datetime.now(timezone.utc).isoformat(),
        'form_data': {
            'name': '',
            'email': '',
            'phone': '',
            'position': '',
            'experience': 0,
            'tech_stack': ''
        },
        'initialized': True
    }
    
    # Only set defaults if they don't exist
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Log initialization if audit logging is enabled
    if SECURITY_CONFIG['audit_logging'] and not st.session_state.get('_logged_initialization', False):
        security_manager.log_audit_event(
            'session_initialized',
            'application',
            'info',
            {'timestamp': datetime.now(timezone.utc).isoformat()}
        )
        st.session_state._logged_initialization = True

def show_interview_page() -> None:
    """Display the interview interface with enhanced conversation handling"""
    st.title("🔍 Technical Interview")
    
    # Initialize session state
    initialize_session_state()
    
    # Get the current session from session state
    session = st.session_state.session
    
    # Display conversation history with improved formatting
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            # Add typing indicator for AI messages
            if msg["role"] == "assistant" and msg.get("is_typing", False):
                with st.spinner("AI is thinking..."):
                    time.sleep(0.5)  # Simulate typing
            st.markdown(msg["content"])
    
    # Check if all questions are answered
    if st.session_state.current_question_idx >= len(session.questions):
        # All questions answered, prepare to show evaluation
        session.current_stage = InterviewStage.CLOSING
        session.end_time = datetime.now()
        session.calculate_overall_score()
        st.session_state.current_page = "evaluation"
        st.rerun()
    
    # Handle current question
    current_question = session.questions[st.session_state.current_question_idx]
    
    # Check if we're in follow-up mode
    if st.session_state.awaiting_followup and st.session_state.followup_count < 2:
        question_text = current_question.text  # Keep the same question for follow-up
    else:
        # Move to next question
        st.session_state.awaiting_followup = False
        st.session_state.followup_count = 0
        question_text = f"Question {st.session_state.current_question_idx + 1}: {current_question.text}"
        
    # Add question to history if not already there
    if not hasattr(current_question, '_added_to_history'):
        # Clean up any previous questions from history
        st.session_state.conversation_history = [
            msg for msg in st.session_state.get('conversation_history', [])
            if not (msg['role'] == 'assistant' and 
                   (f'Question {st.session_state.current_question_idx + 1}:' in msg['content'] or 
                    'Next question:' in msg['content']))
        ]
        add_to_history("assistant", question_text, is_typing=True)
        current_question._added_to_history = True
        st.rerun()
    
    # Always show answer input form, even in follow-up mode
    with st.form(key='answer_form'):
        answer = st.text_area("Your answer:", 
                           key=f"answer_{st.session_state.current_question_idx}",
                           placeholder="Type your answer here...")
        
        # Create three columns for the buttons
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            submitted = st.form_submit_button("✅ Submit Answer")
        with col2:
            skip_clicked = st.form_submit_button("⏭️ Skip Question")
        with col3:
            if st.form_submit_button("❌ End Interview"):
                st.session_state.current_page = "evaluation"
                st.rerun()
                
        # Handle skip question
        if skip_clicked:
            add_to_history("user", "[Skipped]")
            st.session_state.current_question_idx += 1
            
            # If that was the last question, go to evaluation
            if st.session_state.current_question_idx >= len(session.questions):
                end_msg = """
                That's all the questions! Thank you for your responses.
                I'll now evaluate your answers and prepare a report.
                """
                add_to_history("assistant", end_msg)
                session.current_stage = InterviewStage.CLOSING
                st.session_state.current_page = "evaluation"
            
            st.rerun()
        
        if submitted:
            if not answer.strip():
                st.warning("Please provide an answer before submitting.")
            else:
                # Save the answer
                current_question.answer = answer
                
                # Add user's answer to history
                add_to_history("user", answer)
                
                # Show typing indicator while evaluating
                with st.spinner("Evaluating your answer..."):
                    # Evaluate the answer
                    evaluation = evaluate_response(current_question, answer)
                    current_question.score = evaluation['score']
                    current_question.feedback = evaluation['feedback']
                    
                    # Decide on follow-up or next question
                    if evaluation['score'] < 70 and st.session_state.followup_count < 2:
                        # Ask follow-up question
                        follow_up = generate_follow_up_question(current_question, evaluation)
                        add_to_history("assistant", follow_up, is_typing=True)
                        st.session_state.awaiting_followup = True
                        st.session_state.followup_count += 1
                    else:
                        # Provide feedback
                        feedback_msg = f"Feedback: {evaluation['feedback']}"
                        if st.session_state.followup_count > 0:
                            feedback_msg = "Thank you for the additional information. " + feedback_msg
                        add_to_history("assistant", feedback_msg, is_typing=True)
                        
                        # Reset follow-up state
                        st.session_state.awaiting_followup = False
                        st.session_state.followup_count = 0
                        
                        # Move to next question
                        st.session_state.current_question_idx += 1
                        
                        # If no more questions, go to evaluation
                        if st.session_state.current_question_idx >= len(session.questions):
                            st.session_state.current_page = "evaluation"
                            session.current_stage = InterviewStage.CLOSING
                
                # Rerun to show next question or evaluation
                st.rerun()
            
    # The evaluation page will be shown on the next render due to the page change

def show_evaluation_page() -> None:
    """Display the evaluation report with conversation history"""
    st.title("📊 Interview Evaluation")
    
    # Get the current session
    session = st.session_state.session
    
    # Show conversation history
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.subheader("Conversation History")
        for msg in st.session_state.conversation_history:
            with st.chat_message(msg['role']):
                st.write(msg['content'])
    
    # Show evaluation summary
    st.markdown("---")
    st.subheader("Evaluation Summary")
    
    # Calculate scores
    answered_questions = [q for q in session.questions if q.answer]
    if answered_questions:
        avg_score = sum(q.score for q in answered_questions) / len(answered_questions)
        st.metric("Average Score", f"{avg_score:.1f}/100")
        
        # Show scores by category
        categories = {}
        for q in answered_questions:
            if q.category not in categories:
                categories[q.category] = []
            categories[q.category].append(q.score)
        
        st.subheader("Scores by Category")
        for category, scores in categories.items():
            avg = sum(scores) / len(scores)
            st.progress(avg / 100, text=f"{category}: {avg:.1f}/100")
    
    # Add download button for PDF report
    st.markdown("---")
    
    # Generate the PDF report
    try:
        pdf_data = generate_pdf_report(session, st.session_state.conversation_history)
        
        # Ensure we have valid PDF data
        if pdf_data and len(pdf_data) > 0:
            st.download_button(
                label="📄 Download Full Report",
                data=pdf_data,
                file_name=f"talent_scout_report_{session.candidate_info.get('name', 'candidate').replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        else:
            st.error("Failed to generate PDF report. Please try again.")
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
    
    # Add a button to restart the interview
    if st.button("🔄 Start New Interview"):
        st.session_state.clear()
        st.rerun()

def display_sidebar() -> None:
    """Display the sidebar with app description and navigation"""
    with st.sidebar:
        # App logo and title
        st.markdown("<h1 style='text-align: center;'>🤖 TalentScout AI</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        # App description
        st.markdown("""
        **TalentScout AI** is an intelligent interview assistant that helps you:
        
        - Conduct structured technical interviews
        - Evaluate candidate responses in real-time
        - Generate comprehensive interview reports
        - Streamline your hiring process
        
        Simply follow the interview flow and provide your feedback.
        """)
        
        st.markdown("---")
        
        # Navigation help
        st.markdown("### Navigation")
        st.markdown("1. Fill in candidate details")
        st.markdown("2. Answer interview questions")
        st.markdown("3. Review the evaluation report")
        
        # Add some space at the bottom
        st.markdown("\n\n\n\n\n")
        st.caption("© 2025 TalentScout AI - Making Hiring Smarter")

@require_auth
def protected_main():
    """Protected main application logic that requires authentication"""
    # Initialize session state if it doesn't exist
    if 'session' not in st.session_state:
        st.session_state.session = InterviewSession()
    
    # Get the current session
    session = st.session_state.session
    
    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Initialize question tracking if it doesn't exist
    if 'current_question_idx' not in st.session_state:
        st.session_state.current_question_idx = 0
        session.current_question_index = 0
    
    # Initialize follow-up state if it doesn't exist
    if 'awaiting_followup' not in st.session_state:
        st.session_state.awaiting_followup = False
    if 'followup_count' not in st.session_state:
        st.session_state.followup_count = 0
        
    # Add logout button to sidebar
    with st.sidebar:
        if st.button("🚪 Logout"):
            logout()
        
        # Display user info
        if 'username' in st.session_state:
            st.markdown(f"""
                <div style='margin-top: 20px; padding: 10px; background-color: #1a1a1a; border-radius: 5px;'>
                    <p style='margin: 0; color: #4CAF50; font-weight: bold;'>
                        👤 {st.session_state.username}
                    </p>
                    <p style='margin: 5px 0 0 0; font-size: 0.8em; color: #aaa;'>
                        Last active: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}
                    </p>
                </div>
            """, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Initialize session state variables
    initialize_session_state()
    
    # Log application start
    if SECURITY_CONFIG['audit_logging']:
        security_manager.log_audit_event(
            "app_start", 
            "application", 
            "info", 
            {"message": "Application started"}
        )
    
    # Initialize AI components
    global model, generation_config
    if model is None or generation_config is None:
        model, generation_config = initialize_ai()
    
    # Display sidebar
    display_sidebar()
    
    # Debug: Print session state for troubleshooting
    if DEBUG:
        st.sidebar.json(st.session_state.to_dict())
    
    # Check if interview has started
    interview_started = st.session_state.get('interview_started', False)
    current_stage = st.session_state.get('current_stage', InterviewStage.GREETING)
    current_page = st.session_state.get('current_page', 'greeting' if not interview_started else 'interview')
    
    # Debug output
    if DEBUG:
        st.sidebar.subheader("Debug Info")
        st.sidebar.json({
            'interview_started': interview_started,
            'current_stage': str(current_stage),
            'current_page': current_page,
            'question_idx': st.session_state.get('current_question_idx', 0)
        })
    
    # Route to the appropriate page based on current_page
    if current_page == 'greeting' or not interview_started:
        show_greeting_page()
    elif current_page == 'interview':
        show_interview_page()
    elif current_page == 'evaluation':
        show_evaluation_page()
    else:
        st.error("Invalid page state. Please refresh the page to start over.")
        # Reset to initial state if invalid
        st.session_state.update({
            'current_stage': InterviewStage.GREETING,
            'current_page': 'greeting',
            'interview_started': False
        })
        st.rerun()

if __name__ == "__main__":
    main()