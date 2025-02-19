#!/usr/bin/env python3
import os
import time
from picamzero import Camera
from PIL import Image


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "images")

    camera = Camera()
    image_counter = 1

    print("Starting Preview")
    camera.start_preview()
    time.sleep(2)

    print("Start Saving Images")

    try:
        while True:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            raw_filename = "temp.jpg"  # Temporary filename for full-resolution capture
            zoomed_filename = f"image_{timestamp}_{image_counter}.jpg"
            final_path = os.path.join(output_dir, zoomed_filename)

            # Capture the full frame first
            camera.take_photo(raw_filename)

            # Crop the central third (3x zoom) using Pillow
            with Image.open(raw_filename) as img:
                width, height = img.size

                left = int(width * (1 / 3))
                right = int(width * (2 / 3))
                top = int(height * (1 / 3))
                bottom = int(height * (2 / 3))

                cropped = img.crop((left, top, right, bottom))

                # Optional: resize cropped region back to original dimensions
                # to fill the same resolution with a 3x digital zoom effect.
                cropped = cropped.resize((width, height), Image.LANCZOS)

                cropped.save(final_path)

            print(f"Captured {final_path}")
            image_counter += 1
            time.sleep(1.5)
    except KeyboardInterrupt:
        print("Script interrupted by user. Exiting.")
        camera.stop_preview()


if __name__ == "__main__":
    main()
