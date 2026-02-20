# test_endpoint.py
import requests
import os
import base64

# --- Configuration ---
API_URL = "http://api.chess-atlas.com/api/v1/analyze-board"
IMAGE_PATH = "./test_images/-ZVbDR3sRRo_frame_15.png" # Make sure this image exists

# --- Check if image exists ---
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Test image not found at '{IMAGE_PATH}'")
else:
    # --- Prepare the request ---
    # The 'files' dictionary is how 'requests' handles multipart/form-data
    files = {'image': open(IMAGE_PATH, 'rb')}

    try:
        # --- Send the request ---
        print("Sending request to API...")
        print(f"URL: {API_URL}")
        response = requests.post(API_URL, files=files, data={"orientation": "Black", "include_cropped_image": "true"})
        
        # --- Print the results ---
        print(f"Status Code: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("Response JSON:")
            # print(data)
            print(data['data']['fen'])  # Print just the FEN string for clarity
            
            # Bonus: Decode and save the returned cropped image to verify it
            if data['status'] == 'success':
                image_data_url = data['data']['cropped_image']
                header, encoded = image_data_url.split(",", 1)
                image_bytes = base64.b64decode(encoded)
                with open("cropped_result.jpg", "wb") as f:
                    f.write(image_bytes)
                print("\n✅ Successfully decoded and saved 'cropped_result.jpg'")
        else:
            print("Error Response:")
            print(response.text)

    except requests.exceptions.ConnectionError as e:
        print(f"\nConnection Error: Could not connect to the server at {API_URL}.")
        print("Please make sure your Flask server is running.")