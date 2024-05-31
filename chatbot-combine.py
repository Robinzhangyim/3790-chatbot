import cv2
import numpy as np
import os
import pickle
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from nltk.chat.util import Chat, reflections

# PCA

# Create user data storage path
USER_DATA_PATH = 'user_data.pkl'

# Load or initialize user data
def load_user_data():
    if os.path.exists(USER_DATA_PATH):
        with open(USER_DATA_PATH, 'rb') as f:
            return pickle.load(f)
    return {}

def save_user_data(data):
    with open(USER_DATA_PATH, 'wb') as f:
        pickle.dump(data, f)

user_data = load_user_data()

# Preprocessing images
def preprocess_image(image, size=(50, 50)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, size)  # Adjust the image to a fixed size
    flat_image = resized_image.flatten()
    # print(f"Preprocessed image shape: {flat_image.shape}")
    return flat_image

# Perform PCA principal component analysis
def perform_pca(images, n_components=50):
    scaler = StandardScaler()
    scaled_images = scaler.fit_transform(images)
    # print(f"Scaled images shape: {scaled_images.shape}")
    pca = PCA(n_components=n_components)
    pca_images = pca.fit_transform(scaled_images)
    # print(f"PCA images shape: {pca_images.shape}")
    return pca, scaler, pca_images

# Facial recognizer class
class FaceRecognizer:
    def __init__(self, pca, scaler, threshold=0.6):
        self.pca = pca
        self.scaler = scaler
        self.threshold = threshold

    def recognize(self, pca_face, user_face):
        # print(f"X: {pca_face.shape}")
        # print(f"Y: {user_face.shape}")
        similarity = cosine_similarity([pca_face], [user_face])
        print(f"The similarity is: {similarity}")
        if similarity > self.threshold:
            return True
        else:
            return False

# User registration function
def register_user():
    username = input("Enter username: ")
    password = input("Enter password: ")

    if username in user_data:
        print("The username already exists, please choose a different username.")
        return False

    print("The system is about to input facial information...")
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    face_images = []
    start_time = time.time()

    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Recording Face', frame)
            face_image = preprocess_image(frame)
            face_images.append(face_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not face_images:
        print("Failed to capture facial image.")
        return

    face_images = np.array(face_images)
    # print(f"Face images shape: {face_images.shape}")
    pca, scaler, pca_images = perform_pca(face_images)
    # print(f"PCA images after registration shape: {pca_images.shape}")

    user_data[username] = {
        'password': password,
        'face': np.mean(pca_images,axis=0),  # Using the average face after PCA as the user's facial feature
        'pca': pca,
        'scaler': scaler
    }
    save_user_data(user_data)
    print("Registered successfully!")

# User login function
def login_user():
    username = input("Enter username: ")
    password = input("Enter password: ")

    if username not in user_data:
        print("Account error, please register or enter the correct username.")
        return False, None

    if user_data[username]['password'] != password:
        print("Password error, please re-enter the password.")
        return False, None

    print("Starting to scan faces soon...")
    time.sleep(2)

    cap = cv2.VideoCapture(0)
    face_images = []
    start_time = time.time()

    while time.time() - start_time < 2:  # Scan for 2 seconds
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Recognizing Face', frame)
            face_image = preprocess_image(frame)
            face_images.append(face_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if not face_images:
        print("Failed to capture facial image.")
        return False, None

    face_images = np.array(face_images)
    # print(f"Face images shape: {face_images.shape}")

    user_info = user_data[username]
    pca = user_info['pca']
    scaler = user_info['scaler']

    scaled_face_images = scaler.transform(face_images)
    # print(f"Scaled face images shape: {scaled_face_images.shape}")
    pca_face_images = pca.transform(scaled_face_images)
    # print(f"PCA face images shape: {pca_face_images.shape}")
    pca_face = np.mean(pca_face_images, axis=0)  # Using average face for recognition

    recognizer = FaceRecognizer(pca, scaler, threshold=0.7)
    recognized_label = recognizer.recognize(pca_face, user_data[username]['face'])

    if recognized_label == True:
        print("Face comparison successful!")
        success=True
        return success, username
    else:
        print("Facial comparison unsuccessful.")
        success = False
        return success, None

# NLP

# Chat robot function
def chat():
    import nltk
    from threading import Timer
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    import os
    import sys

    # Ensure necessary NLTK resources are downloaded
    # nltk.download('punkt')

    class ChatBot:
        def __init__(self):
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            self.responses = [
                "This product belongs to the PCIE4.0 series of solid-state drives.",
                "Introduction: This product adopts a crystal stack based approach ® The new generation TLC flash memory chips of the Xtacking ® 3.0 architecture support PCIe Gen4x4 interface, with a cache free design scheme and a single chip interface speed of 2400MT/s, which is 50% faster than the previous generation, allowing the four channel scheme to achieve sequential read speeds of up to 7000 MB/s.",
                "The product capacity includes 512G, 1T, 2T, 4T, and there are a total of 4 capacity types for you to choose from.",
                "This product comes with a five-year or TBW limited warranty for more secure data storage.",
                "The product has different performance for different capacities. The 512G, 1T, and 2T interface types are PCIe Gen4x4, NVMe1.4， The 4T interface type is NVMe 2.0. All capacity sequential read and write speeds are 7000MB/s. The 512G sequential write speed is 3600MB/s, while the remaining capacity speeds are 6000MB/s. The average time between failures for all capacities is 1.5 million hours. The service life is 300TBW, 600TBW, and 1200TB respectively."
            ]
            self.embeddings = self.model.encode(self.responses)
            self.terminate = False
            self.timeout = 60  # seconds

        def bert_response(self, user_input):
            user_embedding = self.model.encode([user_input])
            similarities = util.pytorch_cos_sim(user_embedding, self.embeddings)
            best_match_idx = np.argmax(similarities)
            return self.responses[best_match_idx]

        def get_response(self, user_input):
            return self.bert_response(user_input)

        def run(self):
            print(
                "Welcome to the chat system. Please enter the question you want to ask about ZhiTi Tiplus7100 (to exit the chat, please enter 'No', to continue the chat, please enter 'Yes')")
            while not self.terminate:
                user_input = self.get_user_input()
                if self.terminate:
                    sys.exit(0)
                    # os.exit(0)
                    # Immediately exit the program
                if user_input.lower() == "no":
                    print("Bye！")
                    self.terminate = True
                    sys.exit(0)
                    # os.exit(0)
                elif user_input.lower() == "yes":
                    continue
                else:
                    response = self.get_response(user_input)
                    print("chatbot:", response)

        def get_user_input(self):
            timer = Timer(self.timeout, self.prompt_continue)
            timer.start()
            user_input = input("Please enter information: ")
            timer.cancel()
            return user_input

        def prompt_continue(self):
            print("\nUser timeout did not respond, program ended.")
            self.terminate = True
            os._exit(0)  # Immediately exit the program

    bot = ChatBot()
    bot.run()



# Main program
def main():
    if not user_data:
        print("The user information database is empty, please register.")
        register_user()

    while True:
        print("1. Sign up")
        print("2. Log in")
        choice = input("Select Operation (1/2): ")

        if choice == '1':
            register_user()
        elif choice == '2':
            success, username = login_user()
            if success:
                chat()
            else:
                print("Login failed, please try again.")
        else:
            print("Invalid selection, please re-enter.")

if __name__ == "__main__":
    main()
