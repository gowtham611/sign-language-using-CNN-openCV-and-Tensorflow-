import streamlit as st
from streamlit_option_menu import option_menu
import base64
from PIL import Image
import time
import numpy as np
import cv2
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model
from random import choice, shuffle
from collections import deque
import mediapipe as mp

# Initialize MediaPipe Hands
gesture_classes = [
    'dad', 'good morning', 'hello', 'help', 'i', 'love you', 'me', 'mom',
    'need', 'no', 'pineapple', 'sorry', 'want', 'yes', 'your'
]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

class GestureAI:
    def __init__(self):
        # Comprehensive Knowledge Base with 10 detailed Q&A pairs
        self.knowledge_base = {
            "qa_pairs": [
                {
                    "question": "What is sign language?",
                    "answer": "Sign language is a complete, natural language that uses visual-gestural communication through hand shapes, facial expressions, and body language. It's not universal - each country has its own sign language with unique grammar and syntax."
                },
                {
                    "question": "How many hand shapes are there in sign language?",
                    "answer": "In American Sign Language (ASL), there are approximately 40-60 basic hand shapes called 'cheremes'. These hand shapes are fundamental building blocks, similar to how phonemes work in spoken languages."
                },
                {
                    "question": "What is the difference between ASL and other sign languages?",
                    "answer": "Each sign language is unique to its country or region. For example, ASL is different from British Sign Language (BSL) or Australian Sign Language (Auslan). They have distinct grammatical structures, vocabulary, and regional variations."
                },
                {
                    "question": "How do deaf people communicate internationally?",
                    "answer": "Deaf individuals use various methods for international communication, including International Sign (a pidgin sign language), visual gesture communication, writing, and increasingly, technology like translation apps and video interpretation services."
                },
                {
                    "question": "What is deaf culture?",
                    "answer": "Deaf culture is a rich, vibrant community with its own unique identity, values, and social norms. It celebrates visual communication, linguistic heritage, and emphasizes community bonds beyond hearing ability."
                },
                {
                    "question": "How can I start learning sign language?",
                    "answer": "Begin by learning the manual alphabet (fingerspelling), practice basic vocabulary, watch sign language videos, take online courses, engage with deaf community events, and use language learning apps. Consistency and immersion are key."
                },
                {
                    "question": "Are facial expressions important in sign language?",
                    "answer": "Absolutely! Facial expressions are crucial in sign language. They convey grammatical information, emotional tone, and can completely change the meaning of a sign. They're as important as hand movements."
                },
                {
                    "question": "How fast can people communicate in sign language?",
                    "answer": "Experienced sign language users can communicate as quickly as spoken language speakers, typically around 150-250 words per minute. The visual nature of sign language allows for rapid, nuanced communication."
                },
                {
                    "question": "Can sign language be written?",
                    "answer": "While sign languages are primarily visual, there are notation systems like SignWriting that can represent signs in written form. However, most deaf communities use the written language of their country."
                },
                {
                    "question": "What is the history of sign language?",
                    "answer": "Sign language has existed as long as human communication. The first formal sign language education began in the 18th century in France with the work of Abbé Charles-Michel de l'Épée, who established the first public school for the deaf."
                }
            ]
        }

    def generate_response(self, query):
        """
        Advanced response generation with semantic matching
        """
        query = query.lower().strip()
        
        # Semantic matching algorithm
        best_match = None
        max_match_score = 0
        
        for qa_pair in self.knowledge_base['qa_pairs']:
            # Calculate match score
            match_score = self._calculate_match_score(query, qa_pair['question'].lower())
            
            if match_score > max_match_score:
                max_match_score = match_score
                best_match = qa_pair
        
        # Return best matching response or a fallback
        if best_match and max_match_score > 0.3:
            return best_match['answer']
        else:
            return "I'm Gesture AI. While I couldn't find an exact match for your query, I'm always learning. Could you rephrase or ask about sign language basics?"

    def _calculate_match_score(self, query, question):
        """
        Calculate semantic matching score between query and question
        """
        query_words = set(query.split())
        question_words = set(question.split())
        
        # Calculate Jaccard similarity
        intersection = len(query_words.intersection(question_words))
        union = len(query_words.union(question_words))
        
        return intersection / union if union > 0 else 0

def gesture_ai_chat_interface():
    st.markdown("""
    <style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        background: linear-gradient(135deg, #1E1E1E, #2C3E50);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        border-bottom: 2px solid #34495E;
        padding-bottom: 10px;
    }
    .chat-body {
        height: 500px;
        overflow-y: auto;
        padding: 15px;
        background: #2C3E50;
        border-radius: 10px;
        display: flex;
        flex-direction: column;
        scrollbar-width: thin;
        scrollbar-color: #3498DB #2C3E50;
    }
    .message {
        margin-bottom: 15px;
        padding: 10px;
        border-radius: 10px;
        max-width: 80%;
        position: relative;
        animation: fadeIn 0.3s ease;
    }
    .ai-message {
        background: #34495E;
        color: #ECF0F1;
        align-self: flex-start;
    }
    .ai-message::before {
        content: '🤟';
        position: absolute;
        top: -10px;
        left: -10px;
        background: #3498DB;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
    }
    .user-message {
        background: #2980B9;
        color: white;
        align-self: flex-end;
        margin-left: auto;
    }
    .chat-input {
        display: flex;
        gap: 10px;
        margin-top: 20px;
    }
    .chat-input input {
        flex-grow: 1;
        padding: 10px;
        background: #34495E;
        border: 1px solid #2C3E50;
        border-radius: 10px;
        color: #ECF0F1;
        transition: all 0.3s ease;
    }
    .chat-input input:focus {
        outline: none;
        border-color: #3498DB;
        box-shadow: 0 0 10px rgba(52, 152, 219, 0.5);
    }
    .chat-input button {
        background: #3498DB;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .chat-input button:hover {
        background: #2980B9;
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    .typing-indicator {
        display: none;
        align-self: flex-start;
        background: #34495E;
        color: #ECF0F1;
        padding: 5px 10px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes typingDots {
        0%, 20% { opacity: 0; }
        50% { opacity: 1; }
        80%, 100% { opacity: 0; }
    }
    .typing-indicator span {
        animation: typingDots 1.4s infinite;
        display: inline-block;
        margin-left: 4px;
    }
    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }
    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }
    </style>
    """, unsafe_allow_html=True)

    # JavaScript for advanced chat interaction
    st.markdown("""
    <script>
    document.addEventListener('DOMContentLoaded', (event) => {
    const chatBody = document.getElementById('chat-body');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');
    const typingIndicator = document.getElementById('typing-indicator');

    function addMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        messageDiv.innerText = message;
        chatBody.appendChild(messageDiv);
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function showTypingIndicator() {
        typingIndicator.style.display = 'flex';
        chatBody.scrollTop = chatBody.scrollHeight;
    }

    function hideTypingIndicator() {
        typingIndicator.style.display = 'none';
    }

    sendButton.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message) {
            // Add user message
            addMessage(message, 'user-message');
            
            // Show typing indicator
            showTypingIndicator();

            // Send message to Python backend for processing
            fetch('/process_chat_message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                addMessage(data.response, 'ai-message');
            })
            .catch(error => {
                hideTypingIndicator();
                addMessage('Sorry, an error occurred. Please try again.', 'ai-message');
                console.error('Error:', error);
            });

            // Clear input
            chatInput.value = '';
        }
    });

    // Enter key support
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendButton.click();
        }
    });
});

    """, unsafe_allow_html=True)

    # Initialize AI
    gesture_ai = GestureAI()

    # Chat Container
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 style="color: #3498DB;">🤟 Gesture AI Assistant</h2>
            <div style="color: #ECF0F1;">Context: Sign Language Learning</div>
        </div>
        <div id="chat-body" class="chat-body">
            <div id="typing-indicator" class="typing-indicator">
                Gesture AI is thinking
                <span>.</span>
                <span>.</span>
                <span>.</span>
            </div>
        </div>
        <div class="chat-input">
            <input type="text" id="chat-input" placeholder="Ask me anything about sign language...">
            <button id="send-button">Send</button>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_pricing_modal():
    """Display the PRO version pricing and features modal."""
    with st.container():
        st.markdown('<div class="pricing-modal">', unsafe_allow_html=True)
        
        # Header
        st.markdown("<h2 style='text-align: center; color: #00FF9D;'>PRO Version Features</h2>", unsafe_allow_html=True)
        
        # Price
        st.markdown('<div class="price-tag">$9.99/month</div>', unsafe_allow_html=True)
        
        # Features in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-list">
                <div class="feature-item">Advanced Gesture Recognition</div>
                <div class="feature-item">Personalized Learning Path</div>
                <div class="feature-item">Premium Practice Games</div>
                <div class="feature-item">Progress Tracking</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-list">
                <div class="feature-item">Advanced AI Tutor</div>
                <div class="feature-item">Video Analysis Tools</div>
                <div class="feature-item">Community Features</div>
                <div class="feature-item">Priority Support</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.button("Subscribe Now", type="primary")
            st.button("Try Free for 7 Days")
        
        st.markdown('</div>', unsafe_allow_html=True)

def preprocess_frame(frame):
    """
    Preprocess the frame using MediaPipe Hands to extract hand landmarks.
    Reshapes the landmarks to match the model's expected input shape (None, 21, 3, 1).
    """
    try:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Get landmarks of the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract 3D coordinates of 21 landmarks
            landmarks_array = []
            for landmark in hand_landmarks.landmark:
                landmarks_array.extend([landmark.x, landmark.y, landmark.z])
            
            # Convert to numpy array and reshape to (None, 21, 3, 1)
            landmarks_array = np.array(landmarks_array, dtype=np.float32)
            landmarks_array = landmarks_array.reshape(1, 21, 3, 1)
            
            return landmarks_array, results.multi_hand_landmarks
        else:
            return None, None
            
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None, None

def decode_prediction(prediction):
    """
    Convert model prediction to gesture label.
    """
    try:
        # Get prediction index and confidence
        pred_index = np.argmax(prediction[0])
        confidence = prediction[0][pred_index]
        
        return gesture_classes[pred_index], confidence
    except Exception as e:
        st.error(f"Error in decoding prediction: {str(e)}")
        return "Unknown", 0.0

def draw_landmarks(frame, hand_landmarks):
    """
    Draw hand landmarks on the frame for visualization.
    """
    if hand_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        for hand_lms in hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lms,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
    
    return frame

def initialize_speed_sign_game():
    """Initialize a new speed sign game session."""
    signs = [
        ('hello', 'Wave your hand in greeting'),
        ('thank you', 'Touch your chin with your fingertips and move forward'),
        ('love you', 'Show ILY hand sign'),
        ('yes', 'Nod your fist up and down'),
        ('no', 'Shake your index finger side to side'),
        ('help', 'One hand resting on the other, both palms up'),
        ('want', 'Hands in pulling motion towards chest'),
        ('more', 'Fingers together, tap fingertips multiple times')
    ]
    shuffle(signs)
    return signs

def initialize_speed_gesture_game():
    """Initialize a new speed gesture game session."""
    gestures = gesture_classes.copy()
    shuffle(gestures)
    return gestures

def calculate_score(start_time, end_time, correct_answers, total_questions):
    """Calculate game score based on time and accuracy."""
    time_taken = end_time - start_time
    accuracy = correct_answers / total_questions
    base_score = 1000
    time_penalty = time_taken * 10
    score = int((base_score * accuracy) - time_penalty)
    return max(0, score)

# Configure the app with dark theme
st.set_page_config(
    page_title="Gesture Friend",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
# Custom CSS with removed duplicates and simplified styles
# Custom CSS with clean styles
st.markdown("""
    <style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .big-font {
        font-size: 50px !important;
        font-weight: bold;
        color: #00FF9D;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
    .info-font {
        font-size: 20px !important;
        color: #FFFFFF;
        text-align: center;
    }
    .feature-card {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .pro-badge {
        background-color: #FFD700;
        color: #000000;
        padding: 5px 10px;
        border-radius: 15px;
        font-weight: bold;
    }
    .pricing-modal {
        background-color: #2D2D2D;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .price-tag {
        font-size: 40px;
        font-weight: bold;
        color: #00FF9D;
        text-align: center;
        margin: 15px 0;
    }
    .feature-list {
        list-style: none;
        padding: 0;
    }
    .feature-item {
        padding: 10px 0;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .feature-item:before {
        content: "✓";
        color: #00FF9D;
        font-weight: bold;
    }
    /* Custom button styling */
    div[data-testid="stButton"] button {
        background-color: #00FF9D;
        color: #1E1E1E;
        border-radius: 20px;
        padding: 15px 40px;
        font-size: 20px;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        margin: 20px 0;
    }
    div[data-testid="stButton"] button:hover {
        background-color: #00CC7A;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Welcome'
if 'is_pro' not in st.session_state:
    st.session_state.is_pro = False

# Welcome Page
# Welcome Page
# Welcome Page
if st.session_state.page == 'Welcome':
    # Enhanced CSS with video background and image effects
    st.markdown("""
        <style>
        /* Existing styles */
        .hero-section {
            text-align: center;
            padding: 3rem 0;
            background: linear-gradient(135deg, rgba(30,30,30,0.9) 0%, rgba(45,45,45,0.9) 100%);
            border-radius: 15px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
            min-height: 400px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .video-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.3;
            z-index: 0;
        }
        
        .hero-content {
            position: relative;
            z-index: 1;
            padding: 2rem;
        }
        
        .feature-card {
            background: rgba(45,45,45,0.9);
            border-radius: 15px;
            padding: 2rem;
            margin: 1rem 0;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0,255,157,0.2);
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: #00FF9D;
            box-shadow: 0 8px 20px rgba(0,255,157,0.2);
        }
        
        .feature-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        
        .testimonial-section {
            background-image: url('https://images.unsplash.com/photo-1516397281156-ca07cf9746fc?auto=format&fit=crop&w=1920');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            padding: 4rem 2rem;
            position: relative;
            border-radius: 15px;
            margin: 2rem 0;
        }
        
        .testimonial-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(30,30,30,0.85);
            backdrop-filter: blur(5px);
        }
        
        .news-section {
            background-image: url('https://images.unsplash.com/photo-1516321497487-e288fb19713f?auto=format&fit=crop&w=1920');
            background-size: cover;
            background-position: center;
            padding: 2rem;
            border-radius: 15px;
            position: relative;
        }
        
        .news-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(30,30,30,0.8);
            backdrop-filter: blur(5px);
            border-radius: 15px;
        }
        
        .quick-links-section {
            background-image: url('https://images.unsplash.com/photo-1517245386807-bb43f82c33c4?auto=format&fit=crop&w=1920');
            background-size: cover;
            background-position: center;
            padding: 3rem 2rem;
            border-radius: 15px;
            position: relative;
            margin-top: 2rem;
        }
        
        .demo-video-container {
            position: relative;
            padding-top: 56.25%;
            margin: 1rem 0;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .demo-video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border: none;
        }
        
        /* Animation keyframes */
        @keyframes float {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        
        .floating-element {
            animation: float 3s ease-in-out infinite;
        }
        </style>
    """, unsafe_allow_html=True)

    # Hero Section with Video Background
    st.markdown("""
        <div class="hero-section">
            <video class="video-background" autoplay loop muted playsinline>
                <source src="https://example.com/sign-language-hero.mp4" type="video/mp4">
            </video>
            <div class="hero-content">
                <h1 class="big-font floating-element">Welcome to Gesture Friend</h1>
                <p class="info-font">Your AI-powered companion for mastering sign language</p>
                <p style="color: #888; margin-top: 1rem;">Join over 10,000 learners worldwide</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Quick Demo Videos Section
    # Demo Videos Section
    # Demo Videos Section with more compact layout
    st.markdown("<h2 style='margin-top: 2rem; margin-bottom: 1rem; color: #00FF9D; font-size: 1.8rem;'>See Gesture Friend in Action</h2>", unsafe_allow_html=True)
    
    # Adjust column ratio for a more compact layout
    video_col1, video_col2 = st.columns([0.8, 1.2])
    
    with video_col1:
        st.markdown("""
            <div class="compact-video-container">
        """, unsafe_allow_html=True)
        # Load and display the local video
        video_file = open('B:/mini_project/Imagine_a_world_where_V1.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
        video_file.close()
        st.markdown("""
            </div>
            <div class="video-caption">
                <h4>Interactive Learning</h4>
                <div class="video-controls">
                    <button class="control-button" onclick="toggleGuide('guide')">📖 Guide</button>
                    <button class="control-button" onclick="toggleGuide('tips')">💡 Tips</button>
                </div>
                <div class="guide-content" id="guide-content">
                    <div class="guide-item">
                        <span>🎯 Start with basic hand positions</span>
                    </div>
                    <div class="guide-item">
                        <span>✨ Practice fingerspelling daily</span>
                    </div>
                    <div class="guide-item">
                        <span>📱 Use camera for feedback</span>
                    </div>
                </div>
                <div class="guide-content" id="tips-content">
                    <div class="guide-item">
                        <span>💪 Practice in front of mirror</span>
                    </div>
                    <div class="guide-item">
                        <span>🎯 Focus on accuracy over speed</span>
                    </div>
                    <div class="guide-item">
                        <span>🔄 Record and review yourself</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    with video_col2:
        st.markdown("""
            <div class="feature-card enhanced">
                <h3>Why Choose Interactive Learning?</h3>
                <div class="compact-feature-list">
                    <div class="feature-item">
                        <span class="feature-icon">🎯</span>
                        <div class="feature-content">
                            <p>Real-time feedback and AI assistance</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📈</span>
                        <div class="feature-content">
                            <p>Track progress with detailed analytics</p>
                        </div>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🎮</span>
                        <div class="feature-content">
                            <p>Learn through interactive games</p>
                        </div>
                    </div>
                </div>
                <div class="action-buttons">
                    <button class="start-button">Try Now</button>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Updated CSS including guide content styling
    st.markdown("""
        <style>
        /* Existing styles */
        .compact-video-container {
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }
        
        .stVideo {
            width: 100%;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border: 1px solid rgba(0,255,157,0.2);
            transition: all 0.3s ease;
        }
        
        [data-testid="stVideo"] {
            max-height: 280px !important;
        }
        
        .video-caption {
            text-align: center;
            padding: 0.5rem;
        }
        
        .video-caption h4 {
            color: #00FF9D;
            margin: 0.5rem 0;
            font-size: 1rem;
        }
        
        .video-controls {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 0.5rem;
        }
        
        .control-button {
            background: rgba(0,255,157,0.1);
            border: 1px solid rgba(0,255,157,0.3);
            color: #00FF9D;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .control-button:hover {
            background: rgba(0,255,157,0.2);
            transform: translateY(-2px);
        }
        
        /* Guide Content Styling */
        .guide-content {
            display: none;
            background: rgba(45,45,45,0.95);
            border-radius: 10px;
            margin-top: 0.5rem;
            padding: 0.5rem;
            border: 1px solid rgba(0,255,157,0.2);
        }
        
        .guide-item {
            padding: 0.4rem;
            font-size: 0.8rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.3s ease;
            border-radius: 5px;
        }
        
        .guide-item:hover {
            background: rgba(0,255,157,0.1);
            transform: translateX(5px);
        }
        
        /* Feature Card Styles */
        .feature-card.enhanced {
            padding: 1.2rem;
            background: rgba(45,45,45,0.95);
            border-radius: 10px;
            height: calc(100% - 2.4rem);
        }
        
        .feature-card.enhanced h3 {
            color: #00FF9D;
            font-size: 1.2rem;
            margin-bottom: 1rem;
        }
        
        .compact-feature-list {
            margin: 1rem 0;
        }
        
        .feature-item {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            padding: 0.6rem;
            border-radius: 8px;
            transition: all 0.3s ease;
            margin-bottom: 0.5rem;
        }
        
        .feature-item:hover {
            background: rgba(0,255,157,0.1);
            transform: translateX(5px);
        }
        
        .feature-icon {
            font-size: 1.1rem;
            min-width: 30px;
            height: 30px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0,255,157,0.1);
            border-radius: 50%;
        }
        
        .feature-content p {
            color: #fff;
            margin: 0;
            font-size: 0.9rem;
        }
        
        .action-buttons {
            display: flex;
            justify-content: center;
            margin-top: 1rem;
        }
        
        .start-button {
            background: linear-gradient(45deg, #00FF9D, #00FFB3);
            color: #1E1E1E;
            padding: 0.6rem 1.2rem;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .start-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,255,157,0.2);
        }
        </style>

        <script>
            function toggleGuide(type) {
                const guideContent = document.getElementById('guide-content');
                const tipsContent = document.getElementById('tips-content');
                
                if (type === 'guide') {
                    guideContent.style.display = guideContent.style.display === 'none' ? 'block' : 'none';
                    tipsContent.style.display = 'none';
                } else {
                    tipsContent.style.display = tipsContent.style.display === 'none' ? 'block' : 'none';
                    guideContent.style.display = 'none';
                }
            }
        </script>
    """, unsafe_allow_html=True)


    # Features Section with Images
    st.markdown("<h2 style='margin-top: 3rem;'>Features & Benefits</h2>", unsafe_allow_html=True)
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
            <div class="feature-card">
                <img src="https://images.unsplash.com/photo-1531538606174-0f90ff5dce83?auto=format&fit=crop&w=800" 
                     class="feature-image" alt="Beginner features">
                <h3>🎯 For Beginners</h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%;"></div>
                </div>
                <ul>
                    <li>Real-time gesture recognition</li>
                    <li>Interactive step-by-step tutorials</li>
                    <li>Basic practice exercises</li>
                    <li>Progress tracking</li>
                    <li>Community support</li>
                    <li>Mobile compatibility</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with feature_col2:
        st.markdown("""
            <div class="feature-card">
                <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f?auto=format&fit=crop&w=800" 
                     class="feature-image" alt="Pro features">
                <h3>⭐ Pro Features <span class="pro-badge">PRO</span></h3>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%;"></div>
                </div>
                <ul>
                    <li>Advanced gesture recognition with 99% accuracy</li>
                    <li>Personalized learning path with AI</li>
                    <li>Premium practice games & exercises</li>
                    <li>Advanced progress analytics</li>
                    <li>One-on-one tutoring sessions</li>
                    <li>Certification preparation</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Testimonials Section with Background Image
    st.markdown("""
        <div class="testimonial-section">
            <div class="testimonial-overlay"></div>
            <div style="position: relative; z-index: 1;">
                <h2 style="text-align: center; color: #00FF9D;">What Our Users Say</h2>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                    <div class="testimonial-card">
                        <p>"Gesture Friend has transformed how I learn sign language. The AI feedback is incredibly helpful!"</p>
                        <p style="color: #00FF9D; margin-top: 1rem;">- Sarah M., Student</p>
                    </div>
                    <div class="testimonial-card">
                        <p>"The pro features are worth every penny. My signing skills have improved dramatically!"</p>
                        <p style="color: #00FF9D; margin-top: 1rem;">- John D., Professional Interpreter</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Latest News Section with Background
    st.markdown("""
        <div class="news-section">
            <div class="news-overlay"></div>
            <div style="position: relative; z-index: 1;">
                <h2 style="color: #00FF9D;">Latest Updates</h2>
                <div class="news-card">
                    <h4>New Feature Release <span class="badge">NEW</span></h4>
                    <p>Advanced gesture recognition algorithm now supports 50+ sign languages!</p>
                </div>
                <div class="news-card">
                    <h4>Community Milestone <span class="badge">ACHIEVEMENT</span></h4>
                    <p>Celebrating 10,000+ active users worldwide!</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Tutorial Preview Videos
    # Replace Featured Tutorials section with Notable Contributors
# Contributors Section with aligned containers
    st.markdown("""
        <div class="section-header">
            <h2 style='margin-top: 3rem; color: #00FF9D;'>Inspiring Voices: Breaking Barriers</h2>
            <p style='color: #888; text-align: center; margin-bottom: 2rem;'>Celebrating remarkable individuals who transformed challenges into triumphs</p>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for the cards with equal spacing
    col1, col2, col3 = st.columns(3)

    # Define the content for each card
    people = [
        {
            "name": "Helen Keller",
            "title": "Author, Political Activist, Lecturer",
            "image": "th.jpeg",
            "quote": "The only thing worse than being blind is having sight but no vision.",
            "description": "First deafblind person to earn a Bachelor of Arts degree. Her life story became an inspiration for people around the world.",
            "achievements": ["Presidential Medal of Freedom", "Author of 12 Books"]
        },
        {
            "name": "Thomas Edison",
            "title": "Inventor, Businessman",
            "image": "th2.jpeg",
            "quote": "I have not failed. I've just found 10,000 ways that won't work.",
            "description": "Nearly deaf since childhood, Edison became one of history's most prolific inventors with 1,093 US patents.",
            "achievements": ["1,093 Patents", "Congressional Gold Medal"]
        },
        {
            "name": "Ludwig van Beethoven",
            "title": "Composer, Pianist",
            "image": "th1.jpeg",
            "quote": "Music is the mediator between the spiritual and the sensual life.",
            "description": "Composed some of his most profound works after becoming completely deaf, including the Ninth Symphony.",
            "achievements": ["Symphony No. 9", "Moonlight Sonata"]
        }
    ]

    # Create cards in columns
    for col, person in zip([col1, col2, col3], people):
        with col:
            st.markdown(f"""
                <div class="person-card">
                    <div class="image-container">
                        <img src="{person['image']}" alt="{person['name']}">
                        <div class="overlay"></div>
                    </div>
                    <div class="content">
                        <h3>{person['name']}</h3>
                        <p class="title">{person['title']}</p>
                        <div class="quote">"{person['quote']}"</div>
                        <p class="description">{person['description']}</p>
                        <div class="achievements">
                            {''.join(f'<span class="achievement-badge">{achievement}</span>' for achievement in person['achievements'])}
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # Updated CSS for aligned containers
    st.markdown("""
        <style>
        /* Grid Container */
        .stHorizontalBlock {
            align-items: stretch !important;
            gap: 1rem;
        }

        /* Card Container */
        .person-card {
            background: rgba(45,45,45,0.95);
            border-radius: 15px;
            height: 100%;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(0,255,157,0.2);
        }

        .person-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0,255,157,0.2);
        }

        /* Image Container */
        .image-container {
            position: relative;
            width: 100%;
            padding-top: 75%;
            overflow: hidden;
            flex-shrink: 0;
        }

        .image-container img {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            filter: grayscale(50%);
            transition: all 0.3s ease;
        }

        .person-card:hover .image-container img {
            filter: grayscale(0%);
            transform: scale(1.05);
        }

        .overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 50%;
            background: linear-gradient(transparent, rgba(45,45,45,0.95));
        }

        /* Content Section */
        .content {
            padding: 1.5rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }

        .content h3 {
            color: #00FF9D;
            margin: 0;
            font-size: 1.4rem;
        }

        .title {
            color: #888;
            font-size: 0.9rem;
            margin: 0.3rem 0;
        }

        .quote {
            margin: 1rem 0;
            padding: 1rem;
            border-left: 3px solid #00FF9D;
            font-style: italic;
            color: #fff;
            background: rgba(0,255,157,0.1);
            border-radius: 0 10px 10px 0;
            flex-shrink: 0;
        }

        .description {
            color: #ccc;
            font-size: 0.9rem;
            line-height: 1.4;
            margin: 1rem 0;
            flex-grow: 1;
        }

        /* Achievements Section */
        .achievements {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: auto;
        }

        .achievement-badge {
            background: rgba(0,255,157,0.1);
            color: #00FF9D;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            border: 1px solid rgba(0,255,157,0.3);
            white-space: nowrap;
        }

        /* Header Styling */
        .section-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .section-header p {
            max-width: 600px;
            margin: 0 auto;
        }

        /* Column Alignment */
        [data-testid="column"] {
            width: calc(33.33% - 1rem) !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Start Learning button with enhanced styling
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        if st.button('Start Learning Now', use_container_width=True):
            st.session_state.page = 'Main'
            st.rerun()

    # Quick Links Section with Background
    st.markdown("""
        <div class="quick-links-section">
            <div class="testimonial-overlay"></div>
            <div style="position: relative; z-index: 1;">
                <h3 style="color: #00FF9D; text-align: center;">Quick Links</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; margin-top: 2rem;">
                    <div style="text-align: center;">
                        <h4 style="color: #00FF9D;">📚 Resources</h4>
                        <p>Tutorial Library</p>
                        <p>Practice Materials</p>
                        <p>Sign Dictionary</p>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="color: #00FF9D;">🤝 Community</h4>
                        <p>Discussion Forums</p>
                        <p>Study Groups</p>
                        <p>Events Calendar</p>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="color: #00FF9D;">❓ Support</h4>
                        <p>FAQ</p>
                        <p>Contact Us</p>
                        <p>Help Center</p>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'confirmed_gestures' not in st.session_state:
    st.session_state.confirmed_gestures = pd.DataFrame(
        columns=['Timestamp', 'Gesture', 'Confidence']
    )
if 'game_scores' not in st.session_state:
    st.session_state.game_scores = {'speed_sign': [], 'speed_gesture': []}
if 'advanced_ai_history' not in st.session_state:
    st.session_state.advanced_ai_history = deque(maxlen=10)

# Initialize model
if 'model' not in st.session_state:
    try:
        st.session_state.model = load_model('gesture_recognition_model.h5')
        st.session_state.model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state.model_loaded = False

# Welcome Page


# Main Application
elif st.session_state.page == 'Main':
    selected = option_menu(
        menu_title=None,
        options=["Tutorials", "Gesture Examples", "Practice Games", "AI Assistant", "Real-time Recognition"],
        icons=["book", "hand-index", "controller", "robot", "camera"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"background-color": "#2D2D2D"},
            "icon": {"color": "#00FF9D"},
            "nav-link": {"color": "#FFFFFF"},
            "nav-link-selected": {"background-color": "#00FF9D", "color": "#1E1E1E"},
        }
    )
    
    # Tutorials Section
    # Tutorials Section
    if selected == "Tutorials":
        st.header("Tutorials")
        tutorial_type = st.radio("Select Tutorial Level", ["Getting Started", "Advanced (Pro)"])
        
        if tutorial_type == "Getting Started":
            st.subheader("Basic Sign Language Tutorial")
            col1, col2 = st.columns(2)
            with col1:
                st.video("https://www.youtube.com/watch?v=0FcwzMq4iWg")
                st.write("Learn basic hand positions and movements")
            with col2:
                st.video("https://www.youtube.com/watch?v=3yYjYvdcCw8")
                st.write("Practice common everyday signs")
        
        else:  # Advanced
            st.warning("⭐ This is a PRO feature")
            if st.button("Upgrade to PRO"):
                show_pricing_modal()
    
    # Gesture Examples Section
    elif selected == "Gesture Examples":
        st.markdown("""
    <style>
    .gesture-container {
        background: #1E1E1E;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .category-selector {
        background: #2D2D2D;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #404040;
    }
    .gesture-card {
        background: #2D2D2D;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 20px;
        border: 1px solid #404040;
        transition: transform 0.3s ease;
    }
    .gesture-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 255, 157, 0.1);
    }
    .gesture-header {
        padding: 15px;
        background: #343541;
        border-bottom: 1px solid #404040;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .gesture-content {
        padding: 15px;
    }
    .gesture-description {
        color: #AAAAAA;
        font-size: 14px;
        margin-top: 10px;
    }
    .difficulty-badge {
        padding: 4px 8px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    .beginner { background: #4CAF50; color: white; }
    .intermediate { background: #FFA726; color: white; }
    .advanced { background: #EF5350; color: white; }
    
    .gesture-actions {
        display: flex;
        gap: 10px;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #404040;
    }
    .action-button {
        background: #404040;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .action-button:hover {
        background: #00FF9D;
        color: #1E1E1E;
    }
    # Update the video-container style in your CSS
.video-container {
    position: relative;
    padding-top: 40%;  /* Changed from 56.25% to make video smaller */
    width: 80%;  /* Added to make container smaller */
    margin: 15px auto;  /* Centered the container */
    background: #1E1E1E;
    border-radius: 8px;
    overflow: hidden;
}
    }
    .video-container iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        border: none;
    }
    .category-chips {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-bottom: 20px;
    }
    .category-chip {
        padding: 8px 16px;
        border-radius: 20px;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid #565869;
        background: #2D2D2D;
    }
    .category-chip:hover, .category-chip.active {
        background: #00FF9D;
        color: #1E1E1E;
        border-color: #00FF9D;
    }
    </style>
    """, unsafe_allow_html=True)

        st.header("Gesture Examples")
    
        st.markdown('<div class="gesture-container">', unsafe_allow_html=True)
    
    # Enhanced category selector with chips
        st.markdown('<div class="category-chips">', unsafe_allow_html=True)
        categories = {
        "Basic Examples": "👋",
        "Numbers": "🔢",
        "Common Phrases": "💬",
        "Emergency Signs": "🚨",
        "Daily Communication": "📅"
    }
    
        selected_category = st.session_state.get('selected_category', "Basic Examples")
        cols = st.columns(len(categories))
        for i, (category, emoji) in enumerate(categories.items()):
            with cols[i]:
                if st.button(
                f"{emoji} {category}",
                key=f"cat_{category}",
                use_container_width=True,
                type="primary" if category == selected_category else "secondary"
            ):
                    selected_category = category
                    st.session_state.selected_category = category

        st.markdown('</div>', unsafe_allow_html=True)

    # Gesture examples based on category
        if selected_category == "Basic Examples":
            gestures = [
            {
                "name": "Hello",
                "video_id": "uKKvNqA9N20",
                "difficulty": "beginner",
                "description": "A friendly greeting sign, wave your hand from side to side.",
                "tips": "Keep your palm facing forward and maintain a natural smile."
            },
            {
                "name": "Thank You",
                "video_id": "EPlhDhll9mw",
                "difficulty": "beginner",
                "description": "Express gratitude with this simple yet important sign.",
                "tips": "Touch your chin with your fingertips and move forward naturally."
            },
            {
                "name": "Love You",
                "video_id": "jzJjdvTF10A",
                "difficulty": "beginner",
                "description": "The ILY sign combines I, L, and Y handshapes.",
                "tips": "Keep your thumb, index finger, and pinky extended while other fingers are closed."
            },
            {
                "name": "Sorry",
                "video_id": "AMGGpAKltS0",
                "difficulty": "intermediate",
                "description": "A common sign for apologizing.",
                "tips": "Make a fist and rub it in a circular motion on your chest."
            }
        ]

            for gesture in gestures:
                st.markdown(f"""
            <div class="gesture-card">
                <div class="gesture-header">
                    <h3>{gesture['name']}</h3>
                    <span class="difficulty-badge {gesture['difficulty']}">{gesture['difficulty'].title()}</span>
                </div>
                <div class="gesture-content">
                    <div class="video-container">
                        <iframe src="https://www.youtube.com/embed/{gesture['video_id']}"></iframe>
                    </div>
                    <p class="gesture-description">{gesture['description']}</p>
                    <div class="gesture-actions">
                        <button class="action-button">
                            <span>📝 Practice Tips</span>
                        </button>
                        <button class="action-button">
                            <span>📹 Record Practice</span>
                        </button>
                        <button class="action-button">
                            <span>💾 Save to Favorites</span>
                        </button>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add practice tips expandable section
                with st.expander("Practice Tips & Common Mistakes"):
                    st.write(gesture['tips'])
                    st.write("Common mistakes to avoid:")
                    st.write("• Incorrect hand orientation")
                    st.write("• Moving too quickly or slowly")
                    st.write("• Forgetting facial expressions")

        st.markdown('</div>', unsafe_allow_html=True)
    
    # Practice Games Section
    elif selected == "Practice Games":
        st.header("Practice Games")
        game_tab1, game_tab2, game_tab3 = st.tabs(["Speed Sign", "Speed Gesture", "Leaderboard"])
        
        with game_tab1:
            st.subheader("Speed Sign Challenge 🏃‍♂️")
            st.write("Match the sign description with the correct sign name as quickly as possible!")
    
    # Initialize session state variables
            if 'speed_sign_started' not in st.session_state:
                st.session_state.speed_sign_started = False
            if 'sign_data' not in st.session_state:
                st.session_state.sign_data = None
            if 'answers' not in st.session_state:
                st.session_state.answers = {}
    
    # Start Game Button
            if not st.session_state.speed_sign_started and st.button("Start Speed Sign Game"):
                st.session_state.speed_sign_started = True
                st.session_state.sign_data = [
            ('hello', 'Wave your hand in greeting'),
            ('thank you', 'Touch your chin with your fingertips and move forward'),
            ('love you', 'Show ILY hand sign'),
            ('yes', 'Nod your fist up and down'),
            ('no', 'Shake your index finger side to side'),
            ('help', 'One hand resting on the other, both palms up'),
            ('want', 'Hands in pulling motion towards chest'),
            ('more', 'Fingers together, tap fingertips multiple times')
        ]
                shuffle(st.session_state.sign_data)
                st.session_state.answers = {}
                st.session_state.game_start_time = time.time()
                st.rerun()
    
    # Game Interface
            if st.session_state.speed_sign_started and st.session_state.sign_data:
        # Show all questions at once
                for i, (sign, description) in enumerate(st.session_state.sign_data):
                    st.write(f"\nQuestion {i+1}/{len(st.session_state.sign_data)}")
                    st.write(f"Description: {description}")
            
            # Get all possible answers
                    options = [s[0] for s in st.session_state.sign_data]
                    shuffle(options)
                    st.session_state.answers[i] = st.radio(
                "Select the correct sign:",
                options,
                key=f"sign_{i}"
            )
        
        # Submit All Answers button
                if st.button("Submit All Answers"):
                    correct_answers = 0
                    st.write("\nResults:")
            
                    for i, (sign, _) in enumerate(st.session_state.sign_data):
                        user_answer = st.session_state.answers.get(i)
                        if user_answer == sign:
                            st.success(f"Question {i+1}: Correct! ✅")
                            correct_answers += 1
                        else:
                            st.error(f"Question {i+1}: Wrong ❌ (Your answer: {user_answer}, Correct answer: {sign})")
            
                    end_time = time.time()
                    score = calculate_score(
                st.session_state.game_start_time,
                end_time,
                correct_answers,
                len(st.session_state.sign_data)
            )
            
                    st.session_state.game_scores['speed_sign'].append(score)
                    st.success(f"\nGame Complete! Your score: {score}")
                    st.balloons()
            
            # Reset game button
            if st.button("Play Again"):
                st.session_state.speed_sign_started = False
                st.session_state.sign_data = None
                st.session_state.answers = {}
                st.rerun()
        
        with game_tab2:
            st.subheader("Speed Gesture Recognition 🎯")
            st.write("Perform the requested gestures as quickly and accurately as possible!")
    
            if 'speed_gesture_started' not in st.session_state:
                st.session_state.speed_gesture_started = False
    
            if 'current_gesture_index' not in st.session_state:
                st.session_state.current_gesture_index = 0
    
            if st.button("Start Speed Gesture Game"):
                st.session_state.speed_gesture_started = True
                st.session_state.gestures = initialize_speed_gesture_game()
                st.session_state.current_gesture_index = 0
                st.session_state.correct_gestures = 0
                st.session_state.game_start_time = time.time()
    
            if st.session_state.speed_gesture_started:
                if st.session_state.current_gesture_index < len(st.session_state.gestures):
                    target_gesture = st.session_state.gestures[st.session_state.current_gesture_index]
                    st.write(f"\nPerform gesture: {target_gesture}")
                    st.write(f"Progress: {st.session_state.current_gesture_index + 1}/{len(st.session_state.gestures)}")
            
            # Single frame placeholder for camera feed
                    frame_placeholder = st.empty()
            
                    try:
                        cap = cv2.VideoCapture(0)
                        ret, frame = cap.read()
                
                        if ret:
                            processed_landmarks, hand_landmarks = preprocess_frame(frame)
                            if processed_landmarks is not None:
                                prediction = st.session_state.model.predict(processed_landmarks, verbose=0)
                                current_gesture, confidence = decode_prediction(prediction)
                        
                                frame = draw_landmarks(frame, hand_landmarks)
                                frame = cv2.putText(
                            frame,
                            f"Detected: {current_gesture} ({confidence:.2%})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                                if current_gesture == target_gesture and confidence > 0.8:
                                    st.success("Gesture recognized correctly! ✅")
                                    st.session_state.correct_gestures += 1
                                    st.session_state.current_gesture_index += 1
                            
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame)
                
                        cap.release()
                
                    except Exception as e:
                        st.error(f"Error accessing camera: {str(e)}")
            
                    if st.button("Skip Gesture"):
                        st.session_state.current_gesture_index += 1
        
                else:
                    end_time = time.time()
                    score = calculate_score(
                        st.session_state.game_start_time,
                        end_time,
                        st.session_state.correct_gestures,
                        len(st.session_state.gestures)
            )
            
                    st.session_state.game_scores['speed_gesture'].append(score)
                    st.success(f"Game Complete! Your score: {score}")
                    st.balloons()
            
                    if st.button("Play Again"):
                        st.session_state.speed_gesture_started = False
                        st.session_state.current_gesture_index = 0
                        st.rerun()
                        
        
        with game_tab3:
            st.subheader("Game Leaderboard 🏆")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Speed Sign High Scores")
                if st.session_state.game_scores['speed_sign']:
                    scores_df = pd.DataFrame({
                        'Score': st.session_state.game_scores['speed_sign']
                    }).sort_values('Score', ascending=False)
                    st.dataframe(scores_df)
                else:
                    st.info("No Speed Sign scores yet!")
            
            with col2:
                st.write("Speed Gesture High Scores")
                if st.session_state.game_scores['speed_gesture']:
                    scores_df = pd.DataFrame({
                        'Score': st.session_state.game_scores['speed_gesture']
                    }).sort_values('Score', ascending=False)
                    st.dataframe(scores_df)
                else:
                    st.info("No Speed Gesture scores yet!")
    
    # AI Assistant Section
    # AI Assistant Section
    # Enhanced AI Assistant Section
    elif selected == "AI Assistant":
        st.header("Advanced AI Sign Language Assistant")
    
    # Enhanced styling for AI Assistant
        st.markdown("""
    <style>
    .ai-assistant-container {
        background: #2D2D2D;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .model-selector {
        display: flex;
        gap: 15px;
        margin-bottom: 20px;
    }
    .model-card {
        background: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        flex: 1;
        text-align: center;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .model-card:hover, .model-card.active {
        border-color: #00FF9D;
        transform: translateY(-5px);
    }
    .model-icon {
        font-size: 3rem;
        margin-bottom: 10px;
        color: #00FF9D;
    }
    .advanced-feature-section {
        background: #343541;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
    .feature-button {
        background: #00FF9D;
        color: #1E1E1E;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        margin-right: 10px;
        margin-bottom: 10px;
        transition: all 0.3s ease;
    }
    .feature-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 255, 157, 0.2);
    }
    .context-chip {
        background: #404040;
        color: #FFFFFF;
        padding: 8px 15px;
        border-radius: 20px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .context-chip:hover, .context-chip.active {
        background: #00FF9D;
        color: #1E1E1E;
    }
    </style>
    """, unsafe_allow_html=True)

        ai_tab1, ai_tab2, ai_tab3 = st.tabs(["AI Chat", "Learning Paths", "Advanced Features"])

        with ai_tab1:
            gesture_ai_chat_interface()



        with ai_tab2:
            st.subheader("Personalized Learning Paths")
        
        # Learning Path Selector
            learning_levels = ["Beginner", "Intermediate", "Advanced", "Professional"]
            selected_level = st.select_slider("Select Your Skill Level", options=learning_levels)
        
        # Dynamic Learning Resources
            if selected_level == "Beginner":
                st.markdown("""
            ### 🌱 Beginner Learning Path
            - Learn basic hand shapes
            - Master the sign language alphabet
            - Practice simple greetings
            - Understanding basic grammar
            """)
            
                learning_resources = [
                "ASL Basics YouTube Playlist",
                "Fingerspelling Practice Tools",
                "Beginner Sign Language Courses"
            ]
        
            elif selected_level == "Intermediate":
                st.markdown("""
            ### 🌿 Intermediate Learning Path
            - Expand conversational vocabulary
            - Learn complex grammatical structures
            - Practice storytelling
            - Cultural awareness
            """)
            
                learning_resources = [
                "Intermediate Sign Language Workshops",
                "Community Interaction Guides",
                "Advanced Grammar Tutorials"
            ]
        
            else:
            # Advanced and Professional paths
                st.markdown("""
            ### 🌳 Advanced & Professional Learning Path
            - Specialized vocabulary
            - Professional interpretation techniques
            - Cultural nuanced communication
            - Advanced linguistic studies
            """)
            
                learning_resources = [
                "Professional Interpreter Certification Prep",
                "Advanced Linguistic Analysis",
                "Specialized Domain Vocabulary"
            ]
        
        # Display learning resources
            st.subheader("Recommended Resources")
            for resource in learning_resources:
                st.markdown(f"- 📚 {resource}")

        with ai_tab3:
            st.subheader("Advanced AI Features")
        
        # Advanced Feature Buttons
            feature_columns = st.columns(3)
        
            features = [
            {"name": "Sign Similarity Analysis", "icon": "🔍"},
            {"name": "Pronunciation Feedback", "icon": "🎙️"},
            {"name": "Cultural Context Explainer", "icon": "🌍"},
            {"name": "Grammar Breakdown", "icon": "📝"},
            {"name": "Regional Dialect Translator", "icon": "🗺️"},
            {"name": "Learning Progress Tracker", "icon": "📊"}
        ]
        
            for col, feature in zip(feature_columns, features):
                with col:
                    if st.button(f"{feature['icon']} {feature['name']}", 
                             key=feature['name'], 
                             use_container_width=True):
                        st.info(f"Feature: {feature['name']} is coming soon!")

# Additional JavaScript for dynamic model selection
        st.markdown("""
<script>
function setSelectedModel(modelName) {
    // This is a placeholder. In a real Streamlit app, 
    // you'd use Streamlit's Python-based state management
    console.log('Selected Model:', modelName);
}
</script>
""", unsafe_allow_html=True)  # Replace experimental_rerun() with rerun()

    # Rest of the code remains the same...
        
        with ai_tab2:
            st.subheader("Advanced Sign Analysis")
            st.write("Upload a video or use your camera to get detailed feedback on your signing technique.")
            
            analysis_method = st.radio("Choose analysis method:", ["Upload Video", "Live Camera"])
            
            if analysis_method == "Upload Video":
                uploaded_file = st.file_uploader("Upload a video of your signing:", type=['mp4', 'mov'])
                if uploaded_file is not None:
                    st.video(uploaded_file)
                    if st.button("Analyze Video"):
                        with st.spinner("Analyzing your signing..."):
                            time.sleep(2)  # Simulate processing
                            st.write("Analysis Results:")
                            st.write("- Hand positioning: Good")
                            st.write("- Sign clarity: 85%")
                            st.write("- Speed: Appropriate")
                            st.write("- Areas for improvement: Hand transitions could be smoother")
            
            else:  # Live Camera
                st.write("Position yourself in front of the camera and perform signs for real-time analysis.")
                if st.button("Start Analysis"):
                    if st.session_state.model_loaded:
                        try:
                            cap = cv2.VideoCapture(0)
                            analysis_placeholder = st.empty()
                            
                            while True:
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                    
                                processed_landmarks, hand_landmarks = preprocess_frame(frame)
                                if processed_landmarks is not None:
                                    prediction = st.session_state.model.predict(processed_landmarks, verbose=0)
                                    current_gesture, confidence = decode_prediction(prediction)
                                    
                                    # Advanced analysis metrics
                                    frame = draw_landmarks(frame, hand_landmarks)
                                    analysis_text = f"""
                                    Detected Gesture: {current_gesture}
                                    Confidence: {confidence:.2%}
                                    Hand Stability: {'Good' if confidence > 0.8 else 'Needs Improvement'}
                                    Speed: {'Appropriate' if confidence > 0.7 else 'Too Fast/Slow'}
                                    """
                                    
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    analysis_placeholder.image(frame)
                                    st.write(analysis_text)
                                
                                if not st.session_state.camera_on:
                                    break
                            
                            cap.release()
                            
                        except Exception as e:
                            st.error(f"Error during analysis: {str(e)}")
                    else:
                        st.error("Advanced analysis model not loaded. Please check system configuration.")
    
    # Real-time Recognition Section
    elif selected == "Real-time Recognition":
        st.markdown("""
    <style>
    .recognition-container {
        background: #1E1E1E;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .camera-container {
        position: relative;
        border-radius: 12px;
        overflow: hidden;
        background: #2D2D2D;
        padding: 10px;
        border: 2px solid #404040;
    }
    .prediction-box {
        background: rgba(0, 255, 157, 0.1);
        border: 2px solid #00FF9D;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    .confidence-bar {
        height: 6px;
        background: #404040;
        border-radius: 3px;
        margin: 8px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: #00FF9D;
        transition: width 0.3s ease;
    }
    .feedback-container {
        background: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
    .gesture-list {
        max-height: 400px;
        overflow-y: auto;
        border-radius: 8px;
        background: #343541;
    }
    .gesture-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px;
        border-bottom: 1px solid #404040;
        animation: fadeIn 0.3s ease;
    }
    .gesture-item:hover {
        background: #404040;
    }
    .control-button {
        background: #00FF9D;
        color: #1E1E1E;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .control-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 255, 157, 0.2);
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .status-active {
        background: #00FF9D;
        color: #1E1E1E;
    }
    .status-inactive {
        background: #FF4444;
        color: white;
    }
    .feedback-button {
        padding: 5px 10px;
        border-radius: 4px;
        border: 1px solid #565869;
        background: none;
        color: #888;
        cursor: pointer;
        transition: all 0.2s;
    }
    .feedback-button:hover {
        background: #40414F;
        color: #00FF9D;
        border-color: #00FF9D;
    }
    </style>
    """, unsafe_allow_html=True)

        st.header("Real-time Gesture Recognition")
    
        col1, col2 = st.columns([2, 1])
    
        with col1:
            st.markdown('<div class="recognition-container">', unsafe_allow_html=True)
            st.subheader("📹 Camera Feed")
            # In your session state initialization section
            if 'camera_on' not in st.session_state:
                st.session_state.camera_on = False
        # Camera status indicator
            camera_status = "Active" if st.session_state.camera_on else "Inactive"
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span class="status-badge status-{'active' if st.session_state.camera_on else 'inactive'}">
                    {camera_status}
                </span>
                <span style="color: #888;">
                    {f"Processing feed..." if st.session_state.camera_on else "Camera is off"}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
            if st.button('Toggle Camera', key='camera_toggle', use_container_width=True):
                st.session_state.camera_on = not st.session_state.camera_on
        
            st.markdown('<div class="camera-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
        
        # Prediction display
            prediction_placeholder = st.markdown('<div class="prediction-box"></div>', unsafe_allow_html=True)
            confidence_placeholder = st.empty()
        
            if st.session_state.camera_on and st.session_state.model_loaded:
                try:
                    cap = cv2.VideoCapture(0)
                
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to grab frame")
                            break
                    
                        processed_landmarks, hand_landmarks = preprocess_frame(frame)
                    
                        if processed_landmarks is not None:
                            prediction = st.session_state.model.predict(processed_landmarks, verbose=0)
                            current_pred, current_conf = decode_prediction(prediction)
                        
                            frame = draw_landmarks(frame, hand_landmarks)
                            frame = cv2.putText(
                            frame,
                            f"{current_pred} ({current_conf:.2%})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        
                        # Update prediction display with animation
                            prediction_placeholder.markdown(f"""
                            <div class="prediction-box">
                                <h3>Current Prediction:</h3>
                                <h2 style="color: #00FF9D;">{current_pred}</h2>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: {current_conf*100}%;"></div>
                                </div>
                                <p>Confidence: {current_conf:.2%}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame)
                        
                            st.session_state.current_pred = current_pred
                            st.session_state.current_conf = current_conf
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame)
                            prediction_placeholder.markdown("""
                            <div class="prediction-box" style="border-color: #FF4444;">
                                <h3>No Hand Detected</h3>
                                <p>Please show your hand in the camera view</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                        if not st.session_state.camera_on:
                            break
                
                    cap.release()
            
                except Exception as e:
                    st.error(f"Error accessing camera: {str(e)}")
        
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
        with col2:
            st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
            st.subheader("✅ Confirmed Gestures")
        
        # Add feedback section
            if hasattr(st.session_state, 'current_pred'):
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('Confirm Gesture', use_container_width=True):
                        new_row = pd.DataFrame({
                        'Timestamp': [datetime.now()],
                        'Gesture': [st.session_state.current_pred],
                        'Confidence': [st.session_state.current_conf]
                    })
                        st.session_state.confirmed_gestures = pd.concat(
                        [st.session_state.confirmed_gestures, new_row],
                        ignore_index=True
                    )
                        st.success(f"Gesture '{st.session_state.current_pred}' confirmed!")
            
                with col2:
                    if st.button('Incorrect ❌', use_container_width=True):
                        st.session_state.feedback = st.session_state.feedback.append({
                        'Timestamp': datetime.now(),
                        'Predicted': st.session_state.current_pred,
                        'Status': 'Incorrect'
                    }, ignore_index=True)
        
            if len(st.session_state.confirmed_gestures) > 0:
                st.markdown('<div class="gesture-list">', unsafe_allow_html=True)
                for _, row in st.session_state.confirmed_gestures.iterrows():
                    st.markdown(f"""
                    <div class="gesture-item">
                        <div>
                            <strong>{row['Gesture']}</strong>
                            <br>
                            <small style="color: #888;">{row['Timestamp'].strftime('%H:%M:%S')}</small>
                        </div>
                        <div style="color: #00FF9D;">{row['Confidence']:.2%}</div>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
                col1, col2 = st.columns(2)
                with col1:
                    if st.button('Export CSV', use_container_width=True):
                        csv = st.session_state.confirmed_gestures.to_csv(index=False)
                        b64 = base64.b64encode(csv.encode()).decode()
                        href = f'<a href="data:file/csv;base64,{b64}" download="gesture_data.csv">Download CSV</a>'
                        st.markdown(href, unsafe_allow_html=True)
            
                with col2:
                    if st.button('Clear All', use_container_width=True):
                        st.session_state.confirmed_gestures = pd.DataFrame(
                        columns=['Timestamp', 'Gesture', 'Confidence']
                    )
                        st.success("All gestures cleared!")
            else:
                st.info("No confirmed gestures yet. Use the camera to detect and confirm gestures.")
        
        st.markdown('</div>', unsafe_allow_html=True)