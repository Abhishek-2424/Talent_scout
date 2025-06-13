from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import random

class InterviewStage(Enum):
    GREETING = auto()
    TECHNICAL_ASSESSMENT = auto()
    BEHAVIORAL_ASSESSMENT = auto()
    CODING_EXERCISE = auto()
    CLOSING = auto()

@dataclass
class Question:
    text: str
    category: str
    difficulty: str = "medium"
    weight: float = 1.0
    answer: str = ""
    score: float = 0.0
    feedback: str = ""
    evaluation_notes: str = ""

@dataclass
class InterviewSession:
    candidate_info: dict = field(default_factory=dict)
    questions: List[Question] = field(default_factory=list)
    current_stage: InterviewStage = InterviewStage.GREETING
    current_question_index: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    overall_score: float = 0.0
    evaluation_notes: str = ""
    
    def add_question(self, question: Question):
        self.questions.append(question)
    
    def submit_answer(self, answer: str):
        if self.current_question_index < len(self.questions):
            self.questions[self.current_question_index].answer = answer
            self.current_question_index += 1
    
    def evaluate_response(self, question_index: int, evaluation: dict):
        if 0 <= question_index < len(self.questions):
            question = self.questions[question_index]
            question.score = evaluation.get('score', 0)
            question.feedback = evaluation.get('feedback', '')
            question.evaluation_notes = evaluation.get('notes', '')
    
    def calculate_overall_score(self) -> float:
        if not self.questions:
            return 0.0
            
        total_weight = sum(q.weight for q in self.questions if q.answer)
        if total_weight == 0:
            return 0.0
            
        # Calculate weighted average without multiplying by 100 since scores are already 0-100
        weighted_sum = sum(q.score * q.weight for q in self.questions if q.answer)
        self.overall_score = weighted_sum / total_weight
        return self.overall_score
    
    def to_dict(self) -> dict:
        return {
            'candidate_info': self.candidate_info,
            'questions': [{
                'text': q.text,
                'category': q.category,
                'difficulty': q.difficulty,
                'answer': q.answer,
                'score': q.score,
                'feedback': q.feedback,
                'evaluation_notes': q.evaluation_notes
            } for q in self.questions],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'overall_score': self.overall_score,
            'evaluation_notes': self.evaluation_notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'InterviewSession':
        session = cls()
        session.candidate_info = data.get('candidate_info', {})
        session.questions = [
            Question(
                text=q['text'],
                category=q['category'],
                difficulty=q.get('difficulty', 'medium'),
                answer=q.get('answer', ''),
                score=q.get('score', 0),
                feedback=q.get('feedback', ''),
                evaluation_notes=q.get('evaluation_notes', '')
            ) for q in data.get('questions', [])
        ]
        session.current_stage = InterviewStage[data.get('current_stage', 'GREETING')]
        session.current_question_index = data.get('current_question_index', 0)
        session.start_time = datetime.fromisoformat(data['start_time'])
        session.end_time = datetime.fromisoformat(data['end_time']) if data.get('end_time') else None
        session.overall_score = data.get('overall_score', 0.0)
        session.evaluation_notes = data.get('evaluation_notes', '')
        return session

# Conversation handling constants
FALLBACK_RESPONSES = [
    "I'm not sure I understand. Could you rephrase your answer?",
    "I'm here to help with the technical interview. Could you clarify your response?",
    "Let's focus on the technical questions. Could you answer the current question?",
    "I'm not sure I follow. Could you provide more details?",
    "Could you elaborate on that? I'd like to understand your experience better."
]

CONVERSATION_END_KEYWORDS = [
    "goodbye", "exit", "end", "that's all", "finish", "done", "complete",
    "i'm done", "no more questions", "that's it"
]

def check_conversation_end(message: str) -> bool:
    """Check if the message indicates the conversation should end"""
    if not message:
        return False
    return any(keyword in message.lower() for keyword in CONVERSATION_END_KEYWORDS)

def get_fallback_response() -> str:
    """Get a random fallback response"""
    import random
    return random.choice(FALLBACK_RESPONSES)

def generate_follow_up_question(question: 'Question', evaluation: dict) -> str:
    """Generate a follow-up question based on the previous answer and evaluation"""
    tech = question.tech if hasattr(question, 'tech') else 'the technology'
    
    follow_ups = [
        f"I noticed your answer could use more detail. Could you elaborate on how you would approach {tech} in a production environment?",
        f"That's an interesting perspective. Could you provide a specific example of how you've used {tech} in a real-world scenario?",
        f"Let's dive deeper. What are some challenges you've faced while working with {tech}, and how did you overcome them?",
        f"I'd like to understand your experience better. Could you compare {tech} with similar technologies you've worked with?",
        f"To better understand your approach, could you walk me through how you would optimize a solution using {tech}?"
    ]
    
    # Choose a follow-up based on the evaluation score
    score = evaluation.get('score', 0)
    if score < 30:
        # For very low scores, ask for more basic clarification
        return f"Let's break this down. Could you explain your understanding of {tech} in simpler terms?"
    elif score < 60:
        # For moderate scores, ask for examples
        return f"Could you provide a specific example of how you've used {tech} in a project?"
    else:
        # For nearly passing scores, ask for more depth
        return f"That's a good start. Could you elaborate more on how you would implement this in a real-world scenario?"
    
    # Fallback to random follow-up
    import random
    return random.choice(follow_ups)

def generate_question_bank(tech_stack: List[str], years_exp: int) -> List[Question]:
    """Generate a bank of questions based on tech stack and experience"""
    questions = []
    
    # Enhanced question templates with follow-up potential
    question_templates = {
        'conceptual': [
            ("Explain the core concepts of {tech}.", 'beginner'),
            ("How does {tech} handle {feature}?", 'intermediate'),
            ("Compare {tech} with alternative technologies in the same domain.", 'advanced')
        ],
        'practical': [
            ("Describe a project where you used {tech} to solve a problem.", 'intermediate'),
            ("How would you optimize a slow {tech} application?", 'advanced'),
            ("What are the security best practices when working with {tech}?", 'advanced')
        ],
        'problem_solving': [
            ("How would you troubleshoot a memory leak in {tech}?", 'advanced'),
            ("Describe how you would scale a {tech} application.", 'intermediate'),
            ("What metrics would you monitor in a production {tech} application?", 'intermediate')
        ]
    }
    
    # Generate 3-5 questions per technology
    for tech in tech_stack:
        base_difficulty = 'beginner' if years_exp < 2 else 'intermediate' if years_exp < 5 else 'advanced'
        
        # Select questions based on difficulty
        selected_questions = []
        for q_type, templates in question_templates.items():
            for template, difficulty in templates:
                if (base_difficulty == 'beginner' and difficulty == 'beginner') or \
                   (base_difficulty == 'intermediate' and difficulty in ['beginner', 'intermediate']) or \
                   (base_difficulty == 'advanced'):
                    selected_questions.append((template, difficulty))
        
        # Randomly select 3-5 questions per tech
        selected = random.sample(selected_questions, min(len(selected_questions), random.randint(3, 5)))
        
        for template, difficulty in selected:
            # Replace placeholders
            question_text = template.format(
                tech=tech,
                feature=random.choice(['error handling', 'data processing', 'performance optimization'])
            )
            
            questions.append(Question(
                text=question_text,
                category=tech,
                difficulty=difficulty,
                weight=1.5 if difficulty == 'advanced' else 1.0
            ))
    
    # Add behavioral questions (2-3)
    behavioral_questions = [
        ("Describe a challenging project you worked on and how you handled it.", "Problem Solving"),
        ("How do you handle tight deadlines and pressure?", "Time Management"),
        ("Tell me about a time you had to learn a new technology quickly.", "Adaptability"),
        ("Describe a situation where you had to work with a difficult team member.", "Teamwork"),
        ("Describe your experience with code reviews.", "Teamwork")
    ]
    
    selected_behavioral = random.sample(behavioral_questions, min(len(behavioral_questions), 3))
    for text, category in selected_behavioral:
        questions.append(Question(
            text=text,
            category=category,
            difficulty='medium',
            weight=1.0
        ))
    
    return questions

def generate_evaluation(question: Question, answer: str) -> dict:
    """Generate evaluation for a given question and answer"""
    # This would typically call an LLM for evaluation
    # For now, return a mock evaluation
    return {
        'score': min(len(answer.split()) / 20 * 100, 100),  # Simple length-based scoring
        'feedback': f"Good response. Consider providing more specific examples for better scoring.",
        'notes': f"Candidate demonstrated understanding of {question.category}."
    }

def generate_interview_report(session: InterviewSession) -> str:
    """Generate a comprehensive interview report"""
    report = [
        "# Interview Evaluation Report\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Candidate:** {session.candidate_info.get('full_name', 'N/A')}",
        f"**Position:** {', '.join(session.candidate_info.get('desired_positions', ['N/A']))}",
        f"**Tech Stack:** {', '.join(session.candidate_info.get('tech_stack', ['N/A']))}",
        f"**Experience:** {session.candidate_info.get('years_of_experience', 0)} years\n",
        f"## Overall Score: {session.overall_score:.1f}/100\n"
    ]
    
    # Scores by category
    category_scores = {}
    for q in session.questions:
        if q.answer:  # Only include answered questions
            if q.category not in category_scores:
                category_scores[q.category] = []
            category_scores[q.category].append(q.score)
    
    report.append("## Performance by Category\n")
    for category, scores in category_scores.items():
        avg_score = sum(scores) / len(scores)
        report.append(f"- **{category}:** {avg_score:.1f}/100")
    
    # Detailed responses
    report.append("\n## Detailed Responses\n")
    for i, q in enumerate(session.questions, 1):
        if q.answer:
            report.extend([
                f"### Question {i}: {q.text}",
                f"**Answer:** {q.answer}",
                f"**Score:** {q.score:.1f}/100",
                f"**Feedback:** {q.feedback}\n"
            ])
    
    # Summary and recommendations
    report.extend([
        "\n## Summary",
        session.evaluation_notes or "No summary available.",
        "\n## Next Steps",
        "1. Review the candidate's technical responses",
        "2. Consider scheduling a follow-up interview for top candidates",
        "3. Provide feedback regardless of the outcome"
    ])
    
    return "\n".join(report)