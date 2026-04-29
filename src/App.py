from flask import Flask, render_template, request, jsonify
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import heapq
from collections import Counter
from bitarray import bitarray
import os


app = Flask(__name__, template_folder='../templates')

# Global variables for Email Spam detector
email_model = None
email_cv = None 
email_accuracy = 0.0 

# Global model variables for SMS Spam detector
sms_model = None
sms_cv = None
sms_accuracy = 0.0

# ==================== Data Preparation and Model Training ====================

def train_email_model():
    ''' Train the email spam detection model '''

    global email_model, email_cv, email_accuracy 


# ======================= Load Dataset  ==========================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EMAIL_DATASET_PATH = os.path.join(BASE_DIR, '../data/csv/email_spam.csv')

    #Load data set
    try:
        print(f"Looking for dataset at: {EMAIL_DATASET_PATH}")
        spam_df = pd.read_csv(EMAIL_DATASET_PATH, encoding='latin-1')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: Database not found. Please ensure the dataset is in the correct path.")
        return 
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    

# ======================= Data Cleaning  ==========================
    #Clean the data

    spam_df = spam_df.dropna() #remove row with missing values

    spam_df = spam_df[spam_df['Message'].str.strip() != '']  # remove empty messages
    
    # Encode labels: spam = 1, ham = 0
    spam_df['spam'] = spam_df['Category'].apply( lambda x: 1 if x.lower() == 'spam' else 0)
   
        
# ======================= Train/Test Split  ==========================

    # Train/Test split 80/20
    x_train, x_test, y_train, y_test = train_test_split(
        spam_df.Message,
        spam_df.spam,
        test_size=0.20,
        random_state=42
    )

# ======================= Feature Extraction and Model Training  ========================== 
    # Vectorize the text data
    email_cv = CountVectorizer()
    x_train_count = email_cv.fit_transform(x_train.values)

    # Train the model
    email_model = MultinomialNB()
    email_model.fit(x_train_count, y_train)


    # Calculate and Test Accuracy 
    x_test_count = email_cv.transform(x_test)

    email_accuracy = email_model.score(x_test_count, y_test) * 100 
    print (f"Model trained with {email_accuracy:.2f}% accuracy")


# ======================= SMS Spam Detection Model Training ==========================

def train_sms_model():
    ''' Train the SMS spam detection model '''

    global sms_model, sms_cv, sms_accuracy

# ======================= Load SMS Dataset  ==========================

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SMS_DATASET_PATH = os.path.join(BASE_DIR, '../data/csv/sms_spam.csv')

    try:
        print(f"Looking for SMS dataset at: {SMS_DATASET_PATH}")
        sms_df = pd.read_csv(SMS_DATASET_PATH, encoding='latin-1')
        print("SMS Dataset loaded successfully!")
    except FileNotFoundError:
        print("Error: SMS Database not found. Please ensure the dataset is in the correct path.")
        return
    except Exception as e:
        print(f"Error loading SMS dataset: {e}")
        return  

# ======================= Data Cleaning  ==========================

    sms_df = sms_df.dropna() #remove row with missing values

    sms_df = sms_df[sms_df['Message'].str.strip() != '']  # remove empty messages

    # Encode labels: spam = 1, ham = 0
    sms_df['spam'] = sms_df['Category'].apply( lambda x: 1 if x.lower() == 'spam' else 0)   

# ======================= Train/Test Split  ==========================

    # Train/Test split 80/20
    x_train, x_test, y_train, y_test = train_test_split(
        sms_df.Message,
        sms_df.spam,
        test_size=0.20,
        random_state=42
    )

# ======================= Feature Extraction and Model Training  ==========================
    # Vectorize the text data
    sms_cv = CountVectorizer()
    x_train_count = sms_cv.fit_transform(x_train.values)    

    # Train the model
    sms_model = MultinomialNB()
    sms_model.fit(x_train_count, y_train)

    # calculate and Test Accuracy
    x_test_count = sms_cv.transform(x_test)
    sms_accuracy = sms_model.score(x_test_count, y_test) * 100
    print (f"SMS Model trained with {sms_accuracy:.2f}% accuracy")



# ==================== Prediction Function ======================================
def predict_spam(text, model_type='email'):

    ''' Predict if email is spam or not '''
    if model_type == 'email':
        model = email_model
        cv = email_cv
        overall_accuracy = email_accuracy

    else: # sms model
        model = sms_model
        cv = sms_cv
        overall_accuracy = sms_accuracy


    # Check if model is trained
    if model is None or cv is None:
        return {
            'error': 'Model not trained yet. Please restart the app.',
            'classification': 'Unknown'
        }
        
    # Convert the email text to a count vector
    email_txt_count = cv.transform([text])

    # make prediction 0 = ham, 1 = spam
    predicition = model.predict(email_txt_count)[0]

    # Get probability scores
    probabilities = model.predict_proba(email_txt_count)[0] 
    ham_confidence = probabilities[0] * 100 
    spam_confidence = probabilities[1] * 100

    # Determine classification
    classification = 'Spam' if predicition == 1 else 'Not Spam'

    # Determine risk lvl based on spam probability 
    if spam_confidence >= 80:
        risk_lvl = 'High Risk'
        risk_class = 'high-risk'
    elif spam_confidence >= 60:
        risk_lvl = 'Medium Risk'
        risk_class = 'medium-risk'
    elif spam_confidence >= 40:
        risk_lvl = 'Low Risk'
        risk_class = 'low-risk'
    else:
        risk_lvl = 'Safe'
        risk_class = 'safe'


    return {
        'classification' : classification,
        'confidence' : round(spam_confidence if predicition == 1 else ham_confidence, 2),
        'spam_probability': round(spam_confidence, 2),  # FIX: Fixed typo 'probaility'
        'ham_probability': round(ham_confidence, 2),
        'risk_lvl': risk_lvl,
        'risk_class': risk_class,
        'model_accuracy' : round(overall_accuracy, 2)
    }


# ====================  Endpoints =======================================
@app.route('/')
def index():
    # FIX: Removed 'templates/' prefix since template_folder already points to ../app
    return render_template('indextesting.html', 
                           email_accuracy = round(email_accuracy, 2),
                           sms_accuracy = round(sms_accuracy, 2)
                           )


# ==================== Email Spam Detection Endpoints =======================================
@app.route('/upload', methods=['POST'])
def upload():

    # If user try to submit nothing 
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']


        # if I cannot read the file
        if file.filename == '':
            return jsonify({'error': 'Could not read file'}), 400
        
        # read the file 
        try:
            email_txt = file.read().decode('utf-8').strip()
        except:
            return jsonify({'error': 'File is empty'}), 400
        

        # make prediction
        result = predict_spam(email_txt, model_type='email')
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500
    


# ==================== SMS Spam Detection Endpoints =======================================
@app.route('/analyze_sms', methods=['POST'])
def analyze_sms():
    try:
        data = request.get_json()
        sms_text = data.get('text', '').strip()

        if not sms_text:
            return jsonify({'error': 'Please enter some text'}), 400

        result = predict_spam(sms_text, model_type='sms')
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500


# ====================  Model Training on Startup =======================================
def train_model():      
    ''' Train both email and SMS spam detection models '''

    train_email_model()
    train_sms_model()       
    
    # just to confirm models are trained
    print("\n" + "="*50)
    if email_model is not None:
        print("Email Spam Detection Model is ready.")
    else:
        print("Email model failed to train.")
        

    if sms_model is not None:
        print("SMS Spam Detection Model is ready.")
    else:
        print("SMS model not available (dataset missing).")
    print("="*50 + "\n")
        

    
# =========================== Text Analysis Endpoint =======================================================

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()

        if not text:
            return jsonify({'error': 'Please enter some text'}), 400

        metrics = analyze_text_metrics(text)
        return jsonify(metrics)
    
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500
    

# =========================== Plagiarism Detection Endpoint =======================================================


@app.route('/plagiarism', methods=['POST'])
def plagiarism():
    """Check for plagiarism between two texts"""

    try:
        data = request.get_json()
        text1 = data.get('text1', '').strip()
        text2 = data.get('text2', '').strip()
        
        if not text1 or not text2:
            return jsonify({'error': 'Both texts are required'}), 400
        
        similarity = calculate_similarity(text1, text2)
        
        if similarity > 80:
            plagiarism_level = 'Very High'
            plagiarism_class = 'very-high'
        elif similarity > 60:
            plagiarism_level = 'High'
            plagiarism_class = 'high'
        elif similarity > 40:
            plagiarism_level = 'Medium'
            plagiarism_class = 'medium'
        else:
            plagiarism_level = 'Low'
            plagiarism_class = 'low'
        
        return jsonify({
            'similarity': similarity,
            'plagiarism_level': plagiarism_level,
            'plagiarism_class': plagiarism_class
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================String Search (Plagiarism)===================================================

def calculate_similarity(text1, text2):

    if not text1 or not text2:
        return 0.0 
    
    #Simple: count matching words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if len(words1) == 0 or len(words2) == 0:
        return 0.0 
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    similarity = (intersection / union) * 100 if union > 0 else 0

    return round(similarity, 2)


# =============================Text Analysis ===================================================

def analyze_text_metrics(text):

    if not text:
        return {
            'word_count': 0,
            'char_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'unique_words' : 0,
            'most_common_words': []
        }

    words =text.split()
    sentences = text.split('.')

    word_count = len(words)
    char_count = len(text)
    sentences_count = len([s for s in sentences if s.strip() != ''])
    average_word_length = (char_count / word_count) if word_count > 0 else 0
    unique_words = len(set(words))      


    word_freq = Counter(w.lower() for w in words)
    most_common = [{'word' : word, 'count': count} for word, count in word_freq.most_common(10)]

    return {
        'word_count': word_count,
        'char_count': char_count,
        'sentence_count': sentences_count,
        'avg_word_length': round(average_word_length, 2),
        'unique_words': unique_words,
        'most_common_words': most_common
    }



# ============================= MAIN =============================================

if __name__ == '__main__':
    print('\n' + '=' * 50)  
    print("Starting Email Spam Detector")
    print("="*50)

    # Train the model on startup
    train_model()

    print("Open on localhost")
    print("="*50 + '\n')  



    app.run(debug=True, host='localhost', port=5000)
