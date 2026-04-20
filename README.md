# Mouse Control using Eye Gaze and Eye Blinks

This Python script utilizes your webcam and MediaPipe's Face Landmarker API to control the computer's mouse cursor using your facial gestures. No physical mouse needed!

## Instructions to Execute

1. **Install Dependencies:**  
   Ensure you have the required Python libraries installed in your environment:
   ```bash
   pip install opencv-python mediapipe pyautogui
   ```

2. **Run the Script:**  
   Execute the python script from your terminal:
   ```bash
   python eye_controlled_mouse.py
   ```
   *(Note: The first time you run this, the script will automatically download the `face_landmarker.task` model file, which is ~10MB).*

3. **To Exit:**  
   Because the script takes over your cursor, it includes built-in failsafes to stop the execution:
   - **Open your mouth wide.**
   - Bring the camera display window into focus and press the **`q`** key or let the program timeout after 5 minutes.

---

## Gesture to Mouse Mapping

The script dynamically translates your eye and facial movements into standard mouse functions:

| Facial Gesture | Mouse Action | Description |
| :--- | :--- | :--- |
| **Gaze Direction** | **Cursor Movement** | The cursor coordinates map dynamically to the horizontal and vertical orientation of your right eye iris. |
| **Left Eye Wink** | **Left Click** | Close your left eye fully to issue a single standard left click. |
| **Right Eye Wink** | **Right Click** | Close your right eye fully to issue a single right click. |
| **Head Tilt Up (Right eye higher)** | **Scroll Up** | Tilt your head so your right eye is positioned higher than your left eye to scroll the window upwards. |
| **Head Tilt Down (Right eye lower)** | **Scroll Down** | Tilt your head so your right eye is positioned lower than your left eye to scroll the window downwards. |
| **Open Mouth Wide** | **Exit Program** | Drops out of the loop and exits the application. |
