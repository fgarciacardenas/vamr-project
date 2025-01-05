#!/usr/bin/env python3
from dataloader import *

SHOW_IMG = True
DATASET = 'kitty'

def main():
    # Initialize FrameManager
    dataset_dir = {'kitty': 0, 'malaga': 1, 'parking': 2}
    frame_manager = FrameManager(base_path = '../data', dataset=dataset_dir[DATASET], bootstrap_frames=[0, 1])
    
    # Continuous operation: Process frames until the end
    start_frame = frame_manager.current_index + 1
    for i in range(start_frame, frame_manager.last_frame + 1):
        print(f"\n\nProcessing frame {i}\n=====================")

        try:
            frame_manager.update()
            current_frame = frame_manager.get_current()
            previous_frame = frame_manager.get_previous()

            # Display the image
            if SHOW_IMG:
                cv2.imshow('Previous Frame', previous_frame)
                cv2.imshow('Current Frame', current_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting processing loop.")
                    break

            # Add your processing logic here

        except IndexError as e:
            print(e)
            break

    print("Processing completed.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
