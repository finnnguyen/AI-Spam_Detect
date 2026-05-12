Current Github Branch - Group Collaboration 

This project WILL REMAIN PRIVATE until final submission.


## Project Structure

```text
Spam_Detector_with_Additional_tools
├── src
│   └── App.py              # Flask app, train model, predict classification
├── templates
│   └── indextesting.html   # HTML user interface
├── data
│   └── csv
│       ├── email_spam.csv
│       └── sms_spam.csv
├── requirements.txt        # Required libraries
└── README.md               # Project documentation
```


## Project Description
This project is a Spam Detector web application built using Flask and a Naive Bayes classification model. 
The application allows users to upload text messages and classify them as either "spam" or "not spam". 
The model is trained on a dataset of labeled messages (email_spam.csv and sms_spam.csv) and utilizes various text processing techniques to improve classification accuracy.

In addition to spam detection, the application includes additional tools for text analysis, such as word frequency visualization and sentiment analysis.
The user interface is built with HTML and provides a simple User Interface for users to interact with the application.

## How to Run the Application
1. Clone the repository to your local machine or download the zip file.
2. Navigate to the project directory. 
3. Install the required libraries using the command: pip install -r ./requirements.txt (make sure you have Python and pip installed)
4. Run the Flask application using the command: python ./src/App.py   
5. Open your web browser and go to http://127.0.0.1:5000/ to access the application.
6. Upload a text file to classify it as "spam" or "not spam".
7. Video Demo Link: will be provided upon final submission.

## REQUIREMENTS 
Libraries needed to run the application:
- Flask
- pandas
- scikit-learn
- numpy
- Werkzeug
## CONTRIBUTORS 
- Finn Nguyen
- Alan Hoang
- Michael Aladesuru
- Osvaldo Torres Guerrero
## Testing  
Test Common Words : https://gist.github.com/MattIPv4/045239bc27b16b2bcf7a3a9a4648c08a #file-bee-movie-script
