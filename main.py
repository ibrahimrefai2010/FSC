import pygame
from pygame.locals import OPENGL, DOUBLEBUF
from OpenGL.GL import *
import numpy as np
import cv2
from SpoutGL import SpoutReceiver

WIDTH, HEIGHT = 1280, 720

def main():
    # --- OpenGL context (hidden window) ---
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), OPENGL | DOUBLEBUF)
    pygame.display.set_caption("Spout OpenGL Context")

    receiver = SpoutReceiver()
    receiver.setReceiverName("PythonOpenCVReceiver")
    receiver.setActiveSender("Pisa")

# IMPORTANT: force sender selection
    sender_name = "Pisa"
    print("Connected to:", sender_name)

    print("Waiting for Spout sender...")

    clock = pygame.time.Clock()

    while True:
        # Required for OpenGL context
        pygame.event.pump()

        # Receive frame
        receiver.receiveTexture()

        w, h = receiver.getSenderWidth(), receiver.getSenderHeight()
        if w == 0 or h == 0:
            continue

        # Read pixels from GPU
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        pixels = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)

        # Convert to NumPy
        frame = np.frombuffer(pixels, dtype=np.uint8)
        frame = frame.reshape((h, w, 4))

        # OpenGL → OpenCV fixes
        frame = np.flipud(frame)              # Vertical flip
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        # Show in OpenCV
        cv2.imshow("Spout → OpenCV", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

        clock.tick(60)

    receiver.releaseReceiver()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()
