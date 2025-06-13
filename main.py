import os
import re
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from interview_logic import (
    InterviewSession, InterviewStage, Question, 
    generate_question_bank, check_conversation_end,
    generate_follow_up_question
)
from report_generator import generate_pdf_report

# Load environment variables
load_dotenv()

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

# Configure page
st.set_page_config(
    page_title="TalentScout AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
except Exception as e:
    st.error(f"Error initializing AI: {str(e)}")
    st.stop()

if 'session' not in st.session_state:
    st.session_state.session = InterviewSession()
    st.session_state.current_question_idx = 0
    st.session_state.conversation_history = []
    st.session_state.last_fallback = -1

session = st.session_state.session

def generate_questions() -> None:
    """Generate interview questions based on candidate info"""
    if not session.questions:
        session.questions = generate_question_bank(
            session.candidate_info.get('tech_stack', []),
            session.candidate_info.get('years_of_experience', 1)
        )

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

def show_greeting_page() -> None:
    """Display the greeting and candidate info form"""
    st.title("👋 Welcome to TalentScout AI")
    st.markdown("""
    Thank you for participating in our technical screening process. 
    This AI-powered interview will help us understand your skills and experience better.
    """)
    
    with st.form("candidate_info"):
        st.subheader("Candidate Information")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name*", key="name_input", 
                               value=session.candidate_info.get('name', ''))
            email = st.text_input("Email*", key="email_input",
                               value=session.candidate_info.get('email', ''))
            phone = st.text_input("Phone*", key="phone_input",
                               value=session.candidate_info.get('phone', ''))
            location = st.text_input("Current Location*", key="location_input",
                                   value=session.candidate_info.get('location', ''))
        
        with col2:
            position = st.text_input("Position Applied For*", key="position_input",
                                  value=session.candidate_info.get('position', ''))
            years_exp = st.number_input("Years of Experience*", key="exp_input",
                                      min_value=0, max_value=50, step=1,
                                      value=session.candidate_info.get('years_of_experience', 0))
        
        # Tech stack input with tags
        st.subheader("Technical Skills")
        st.markdown("Enter the technologies you're proficient in (comma-separated)")
        tech_input = st.text_input("Tech Stack*", 
                                 value=", ".join(session.candidate_info.get('tech_stack', [])),
                                 label_visibility="collapsed")
        
        # Form validation
        submitted = st.form_submit_button("Start Interview")
        if submitted:
            errors = []
            
            if not name.strip():
                errors.append("Name is required")
            if not email or not is_valid_email(email):
                errors.append("Please enter a valid email address")
            if not phone or not is_valid_phone(phone):
                errors.append("Please enter a valid phone number")
            if not position.strip():
                errors.append("Position is required")
            
            tech_stack = [t.strip() for t in tech_input.split(",") if t.strip()]
            if not tech_stack:
                errors.append("Please enter at least one technical skill")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Save candidate info
                session.candidate_info = {
                    'name': name,
                    'email': email,
                    'phone': phone,
                    'position': position,
                    'years_of_experience': years_exp,
                    'tech_stack': tech_stack,
                    'start_time': datetime.now().isoformat()
                }
                
                # Generate questions
                generate_questions()
                
                # Move to technical assessment
                session.current_stage = InterviewStage.TECHNICAL_ASSESSMENT
                st.rerun()

def show_interview_page() -> None:
    """Display the interview interface with enhanced conversation handling"""
    st.title("🔍 Technical Interview")
    
    # Get the current session from session state
    session = st.session_state.session
    
    # Initialize conversation history if not exists
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
        
    # Initialize follow-up state if not exists
    if 'awaiting_followup' not in st.session_state:
        st.session_state.awaiting_followup = False
    if 'followup_count' not in st.session_state:
        st.session_state.followup_count = 0
    
    # Display conversation history with improved formatting
    for msg in st.session_state.conversation_history:
        with st.chat_message(msg["role"]):
            # Add typing indicator for AI messages
            if msg["role"] == "assistant" and msg.get("is_typing", False):
                with st.spinner("AI is thinking..."):
                    time.sleep(0.5)  # Simulate typing
            st.markdown(msg["content"])
    
    # Handle conversation flow
    if st.session_state.current_question_idx < len(session.questions):
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
        
        # Answer input with improved handling
        with st.form(key='answer_form'):
            answer = st.text_area("Your answer:", 
                               key=f"answer_{st.session_state.current_question_idx}",
                               placeholder="Type your answer here...")
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("✅ Submit Answer")
            with col2:
                if st.form_submit_button("❌ End Interview"):
                    st.session_state.current_page = "evaluation"
                    st.rerun()
            
            if submitted and answer.strip():
                # Check for conversation end
                if check_conversation_end(answer):
                    st.session_state.current_page = "evaluation"
                    st.rerun()
                
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
                        # Provide feedback and move to next question
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
                
                st.rerun()
            elif not answer.strip():
                st.warning("Please provide an answer before submitting.")
            
            with col2:
                if st.form_submit_button("Skip Question"):
                    if st.session_state.current_question_idx < len(session.questions) - 1:
                        add_to_history("user", "[Skipped]")
                        st.session_state.current_question_idx += 1
                        # The next question will be added by the main loop on the next render
                        st.rerun()
                    else:
                        add_to_history("user", "[Skipped]")
                        end_msg = """
                        That's all the questions! Thank you for your responses.
                        I'll now evaluate your answers and prepare a report.
                        """
                        add_to_history("assistant", end_msg)
                        session.current_stage = InterviewStage.CLOSING
                        session.end_time = datetime.now()
                        session.calculate_overall_score()
                        st.rerun()
    else:
        # No more questions, move to evaluation
        session.current_stage = InterviewStage.CLOSING
        session.end_time = datetime.now()
        session.calculate_overall_score()
        st.rerun()

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

def main():
    """Main application entry point"""
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
    
    # Display sidebar
    display_sidebar()
    
    # Route to the appropriate page based on interview stage
    if session.current_stage == InterviewStage.GREETING:
        show_greeting_page()
    elif session.current_stage in [InterviewStage.TECHNICAL_ASSESSMENT, InterviewStage.BEHAVIORAL_ASSESSMENT]:
        show_interview_page()
    elif session.current_stage == InterviewStage.CLOSING:
        show_evaluation_page()
    else:
        st.error("Invalid interview stage. Please refresh the page to start over.")

if __name__ == "__main__":
    main()
