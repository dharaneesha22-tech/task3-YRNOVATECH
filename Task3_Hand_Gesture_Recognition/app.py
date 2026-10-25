
import streamlit as st
import cv2
import mediapipe as mp
import time

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="wide"
)

st.title("AI Hand Gesture Recognition")
st.write("Real-time hand gesture detection using Python, OpenCV and MediaPipe.")

st.info("Click Start Webcam and allow camera permission.")

st.sidebar.header("Camera Controls")
start = st.sidebar.toggle("Start Webcam", value=False)

st.subheader("Gestures")
st.write("✋ Open Palm | ✊ Fist | ☝️ One Finger | ✌️ Victory")

def identify_gesture(landmarks):
    points = landmarks.landmark

    # Finger tips and joints
    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    # Count extended fingers
    extended = [
        points[tip].y < points[joint].y
        for tip, joint in zip(tips, joints)
    ]

    count = sum(extended)

    if count == 0:
        return "Fist"
    elif count == 4:
        return "Open Palm"
    elif extended[0] and extended[1] and not extended[2] and not extended[3]:
        return "Victory / Peace"
    elif extended[0] and not any(extended[1:]):
        return "One Finger"
    else:
        return "Other Gesture"

if start:
    frame_area = st.empty()
    status = st.empty()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        st.error("Camera not found. Check camera permission.")
    else:
        mp_hands = mp.solutions.hands
        drawing = mp.solutions.drawing_utils

        try:
            with mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5
            ) as hands:

                while start:
                    success, frame = camera.read()

                    if not success:
                        status.error("Unable to read camera.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                    results = hands.process(rgb_frame)

                    gesture = "No Hand Detected"

                    if results.multi_hand_landmarks:
                        for hand in results.multi_hand_landmarks:
                            gesture = identify_gesture(hand)

                            drawing.draw_landmarks(
                                frame,
                                hand,
                                mp_hands.HAND_CONNECTIONS
                            )

                        cv2.putText(
                            frame,
                            gesture,
                            (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )

                    frame_area.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        width="stretch"
                    )
                    status.write("Detected Gesture: " + gesture)
                    time.sleep(0.03)

        except Exception as error:
            st.error(f"Error: {error}")
        finally:
            camera.release()

else:
    st.markdown("### How to use")
    st.markdown("""
    1. Click **Start Webcam** in the sidebar.
    2. Allow camera access.
    3. Show your hand in front of the camera.
    4. Try an open palm, fist, one finger or peace sign.
    """)

st.divider()
st.caption("Task 3 | AI Hand Gesture Recognition")
