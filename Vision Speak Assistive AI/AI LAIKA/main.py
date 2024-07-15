import speech_recognition as sr
import pyttsx3
import pywhatkit
import datetime
import wikipedia
import pyjokes
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np

# Initialize speech recognition and text-to-speech engines
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)

# Function to speak out text
def talk(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize voice commands
def take_command():
    command = ""  # Initialize command
    try:
        with sr.Microphone() as source:
            print('Listening...')
            listener.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            voice = listener.listen(source, timeout=20)  # 20-second timeout
            command = listener.recognize_google(voice)
            command = command.lower()
            if 'laika' in command:
                command = command.replace('laika', '')
                print(command)
    except sr.WaitTimeoutError:
        talk('Say something. What can I do for you?')
    except sr.UnknownValueError:
        talk('Please say that command again.')
    except Exception as e:
        print(f"Error: {e}")
    return command

# Function to fetch and speak weather information
def Temperature(query):
    try:
        city = query.split("weather in", 1)[1].strip()
        url = f"https://www.google.com/search?q=weather+in+{city}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Using CSS selectors to target specific elements
        region = soup.select('div.BNeawe.iBp4i.AP7Wnd')[0].get_text()
        temperature = soup.select('div.BNeawe.iBp4i.AP7Wnd')[1].get_text()
        weather_condition = soup.select('div.BNeawe.tAd8D.AP7Wnd')[0].get_text()

        response_text = f"It's currently {weather_condition} and {temperature} in {region}."
        talk(response_text)
        print(response_text)
    
    except Exception as e:
        print(f"Error fetching weather: {e}")
        talk("Sorry, I couldn't fetch the weather information right now.")

# Function to perform object detection
def object_detection():
    text = pyttsx3.init()
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    classnames = []
    classfiles = 'coco.names.txt'
    with open(classfiles, 'rt') as f:
        classnames = f.read().rstrip('\n').split('\n')

    configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configpath)

    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        ret, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        print(classIds, bbox)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0))
                cv2.putText(img, classnames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('output', img)
        cv2.waitKey(1)
        ans = classnames[classIds[0] - 1] if len(classIds) != 0 else "No object detected"
        rate = 200
        text.setProperty('rate', rate)
        text.say(ans)
        text.runAndWait()

# Main function to run the AI assistant
def run_laika():
    talk('I am AI Laika. What can I do for you today?')
    while True:
        command = take_command()
        if command:
            print(command)
            if 'play' in command:
                song = command.replace('play', '')
                talk('playing ' + song)
                pywhatkit.playonyt(song)
                print('Playing: ' + song)
                
            elif 'time' in command:
                time = datetime.datetime.now().strftime('%H:%M %p')
                talk('Current time is ' + time)
                print(time)
            
            elif 'who is' in command:
                person = command.replace('who is', '')
                info = wikipedia.summary(person, 2)
                print(info)
                talk(info)
            
            elif 'date' in command:
                talk('Sorry, I have a headache')

            elif 'are you single' in command:
                talk('I am in a relationship with my brain')

            elif 'sing a song' in command:
                talk('I am not a singer, but I can sing a song for you')
                talk('La la lalalala, la la la, la la la laaaaiyykaaa')
                print('Singing a song')

            elif 'my birthday' in command:
                talk('Many many happy returns of the day from the Laika community')    

            elif 'developed you' in command:
                talk('Team Inteli biz, Soorya S, Swaathy S, and Aadithyaa Lakshmi')

            elif 'food' in command:
                talk('Thank you! As an AI assistant, I don\'t need food, but I appreciate your concern. How can I assist you today?')
            
            elif 'who are you' in command:
                talk('I am AI Laika by Team Inteli biz')
            
            elif 'joke' in command:
                talk(pyjokes.get_joke())
            
            elif 'weather' in command:
                Temperature(command)
            
            elif 'detect object' in command:
                talk('Initiating object detection. Please wait.')
                object_detection()
            
            else:
                talk('Please say that command again.')
        else:
            talk('Say something. What can I do for you?')

# Run the AI assistant
run_laika()



