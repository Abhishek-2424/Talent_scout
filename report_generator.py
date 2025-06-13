"""Utility functions for generating interview reports in various formats."""
from typing import Dict, List, Any
from datetime import datetime
from fpdf import FPDF
import streamlit as st
from interview_logic import InterviewSession

class PDF(FPDF):
    """Custom PDF generator for interview reports"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_auto_page_break(auto=True, margin=15)
        self.set_margins(15, 15, 15)  # Left, Top, Right margins
        
        # Try to use Arial if available, otherwise use default font
        try:
            # Try common Windows paths first
            font_paths = [
                'c:/windows/fonts/arial.ttf',
                'c:/winnt/fonts/arial.ttf',
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',  # Common Linux path
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'  # Another common Linux font
            ]
            
            # Try to find the first available font
            for font_path in font_paths:
                try:
                    self.add_font('CustomFont', '', font_path, uni=True)
                    # If we get here, font was added successfully
                    self.set_font('CustomFont', '', 12)
                    return
                except (RuntimeError, OSError):
                    continue
            
            # If we get here, no custom fonts were found - use default
            self.set_font('Arial', '', 12)
            
        except Exception as e:
            # Fall back to default font if any error occurs
            self.set_font('Arial', '', 12)
    
    def header(self):
        # Try to use bold font if available, otherwise use regular
        try:
            self.set_font('CustomFont', 'B', 15)
        except:
            self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'TalentScout AI - Interview Report', 0, 1, 'C')
        self.ln(5)
        
    def footer(self):
        self.set_y(-15)
        # Try to use italic font if available, otherwise use regular
        try:
            self.set_font('CustomFont', 'I', 8)
        except:
            self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def safe_cell(self, w, h=0, txt='', border=0, ln=0, align='', fill=False, link=''):
        """A safer version of cell that handles encoding"""
        try:
            self.cell(w, h, txt, border, ln, align, fill, link)
        except Exception as e:
            # Fallback for problematic characters
            safe_txt = txt.encode('latin-1', 'replace').decode('latin-1')
            self.cell(w, h, safe_txt, border, ln, align, fill, link)

def generate_pdf_report(session: InterviewSession, conversation_history: List[Dict[str, str]]) -> bytes:
    """Generate a PDF report of the interview session.
    
    Args:
        session: The interview session containing candidate info and evaluation
        conversation_history: List of conversation messages
        
    Returns:
        bytes: PDF file content as bytes
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Set font for the document
    pdf.set_font('Arial', '', 12)
    
    # Add candidate information
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Candidate Information', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    candidate_info = session.candidate_info
    pdf.cell(0, 10, f"Name: {candidate_info.get('name', 'Not provided')}", 0, 1)
    pdf.cell(0, 10, f"Email: {candidate_info.get('email', 'Not provided')}", 0, 1)
    pdf.cell(0, 10, f"Phone: {candidate_info.get('phone', 'Not provided')}", 0, 1)
    pdf.cell(0, 10, f"Position: {candidate_info.get('position', 'Not specified')}", 0, 1)
    pdf.cell(0, 10, f"Experience: {candidate_info.get('years_of_experience', 0)} years", 0, 1)
    pdf.cell(0, 10, f"Tech Stack: {candidate_info.get('tech_stack', ['None specified'])}", 0, 1)
    
    # Add interview details
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Interview Details', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    start_time = session.start_time.strftime('%Y-%m-%d %H:%M:%S') if session.start_time else 'N/A'
    end_time = session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'N/A'
    duration = str(session.end_time - session.start_time).split('.')[0] if session.start_time and session.end_time else 'N/A'
    
    pdf.cell(0, 10, f"Start Time: {start_time}", 0, 1)
    pdf.cell(0, 10, f"End Time: {end_time}", 0, 1)
    pdf.cell(0, 10, f"Duration: {duration}", 0, 1)
    
    # Add conversation history
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Conversation History', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    for msg in conversation_history:
        role = 'Candidate' if msg['role'] == 'user' else 'Interviewer'
        # Handle multi-line content
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, f"{role}:", 0, 1)
        pdf.set_font('Arial', '', 12)
        
        # Process content to handle special characters and long lines
        content = msg['content'].encode('latin-1', 'replace').decode('latin-1')
        
        # Split long text into chunks that fit within page width
        max_width = 180  # Adjust based on your page width and margins
        lines = []
        
        for line in content.split('\n'):
            if not line.strip():
                lines.append('')
                continue
                
            # Split long lines into chunks
            while len(line) > 0:
                # Find the last space within the max width
                if pdf.get_string_width(line) <= max_width:
                    lines.append(line)
                    break
                
                # Binary search for the right split point
                low = 0
                high = len(line)
                while low < high:
                    mid = (low + high) // 2
                    if pdf.get_string_width(line[:mid]) <= max_width:
                        low = mid + 1
                    else:
                        high = mid
                
                # If we can't find a good split point, force a break
                if low <= 1:  # No space found within max_width
                    split_at = max_width
                    while split_at > 0 and not line[split_at-1].isspace():
                        split_at -= 1
                    if split_at == 0:  # No space found, force break
                        split_at = max_width
                else:
                    split_at = max(1, low - 1)
                    
                lines.append(line[:split_at].strip())
                line = line[split_at:].strip()
        
        # Add the processed lines to the PDF
        for line in lines:
            if line:  # Only add non-empty lines
                pdf.cell(0, 6, line, 0, 1)
            else:
                pdf.ln(6)  # Add some space for empty lines
        pdf.ln(2)  # Add a small space after each message
    
    # Add evaluation summary
    if hasattr(session, 'overall_score') and session.overall_score is not None:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Evaluation Summary', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        
        pdf.cell(0, 10, f"Overall Score: {session.overall_score:.1f}/100", 0, 1)
        
        # Add feedback for each question
        for i, q in enumerate(session.questions, 1):
            if hasattr(q, 'feedback') and q.feedback:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f"Question {i}: {q.text}", 0, 1)
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 10, f"Feedback: {q.feedback}")
                pdf.ln(2)
    
    # Add final notes
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Final Notes', 0, 1, 'L')
    pdf.set_font('Arial', '', 12)
    
    notes = """
    This report was automatically generated by TalentScout AI.
    The evaluation is based on the candidate's responses during the interview.
    
    For any questions or concerns, please contact our support team.
    """
    pdf.multi_cell(0, 10, notes.strip())
    
    # Generate PDF and return as bytes
    try:
        # Use a temporary file to ensure proper binary handling
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            pdf.output(tmp_file.name, 'F')
            
        # Read the file as binary
        with open(tmp_file.name, 'rb') as f:
            pdf_data = f.read()
            
        # Clean up the temporary file
        try:
            os.unlink(tmp_file.name)
        except:
            pass
            
        return pdf_data
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        # Return empty bytes if there's an error
        return b''