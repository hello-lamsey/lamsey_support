import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

# Speech Recognition
import speech_recognition as sr
from speech_recognition.audio import AudioData

class Transcriber(Node):
    def __init__(self):
        super().__init__("transcriber_node")

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

    def _predict_text(self, audio_clip: AudioData) -> str:
        """
        Predicts text contained in an audio snippet.

        Parameters
        ----------
        audio_clip : AudioData
            Audio data output from the SpeechRecognition recognizer object

        Returns
        -------
        str
            English text contained in the audio data
        """

        self.get_logger().info("Processing Audio...")
        try:
            return self.recognizer.recognize_google(audio_clip)  # you can change this to be something else
        except sr.UnknownValueError:
            self.get_logger().info("Speech recognizer could not understand audio")
            return None
        except sr.RequestError as e:
            self.get_logger().info("Speech recognition error; {0}".format(e))
            return None

    def start_recording(self, recording_length_s: float=3.) -> str:
        """
        Triggers an audio recording and returns text contained in the recording.

        Parameters
        ----------
        recording_length_s : float
            Number of seconds to record for

        Returns
        -------
        str
            English text contained in the audio data
        """

        with sr.Microphone() as source:
            audio_clip = self.recognizer.record(source, duration=recording_length_s)
            text_string = self._predict_text(audio_clip)
            self.get_logger().info("recognized text: {}".format(text_string))
            return text_string

    def run(self):
        """
        Main method for node.
        """

        # you could put a loop here
        self.start_recording()
        self.get_logger().info("Transcriber done.")

def main():
    rclpy.init()
    node = Transcriber()
    executor = MultiThreadedExecutor(num_threads=4)

    # Spin in the background since detecting faces will block the main thread
    spin_thread = threading.Thread(
        target=rclpy.spin,
        args=(node,),
        kwargs={"executor": executor},
        daemon=True,
    )
    spin_thread.start()

    # Run node
    try:
        node.run()
    except KeyboardInterrupt:
        pass

    # Terminate this node
    node.destroy_node()
    rclpy.shutdown()

    # Join the spin thread (so it is spinning in the main thread)
    spin_thread.join()


if __name__ == '__main__':
    main()
