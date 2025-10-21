# predict_video3.py
import argparse
import os
import cv2
import numpy as np
from skimage.transform import resize
from violencemodel import souhaiel_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to input video")
    ap.add_argument("-o", "--output", required=True,
                    help="path to output video")
    ap.add_argument("-q", "--queue", type=int, default=30,
                    help="number of frames per prediction window (default=30)")
    args = vars(ap.parse_args())

    input_path = args["input"]
    output_path = args["output"]
    queue_len = args["queue"]

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    print("[INFO] Loading model...")
    try:
        model = souhaiel_model("fightw.h5")  # pass weights filename as string
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {input_path}")
        return

    # Prepare output writer
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc,
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frames = np.zeros((queue_len, 160, 160, 3), dtype=np.float32)
    datav = np.zeros((1, queue_len, 160, 160, 3), dtype=np.float32)

    frame_count = 0
    print("[INFO] Processing video...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for model
        resized = resize(frame, (160, 160, 3), anti_aliasing=True)
        frames[frame_count % queue_len] = resized

        frame_count += 1

        if frame_count >= queue_len:
            datav[0] = frames
            preds = model.predict(datav, verbose=0)
            pred_class = np.argmax(preds)
            label = "Violence" if pred_class == 1 else "No Violence"

            # Put label on frame
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255) if pred_class == 1 else (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"[INFO] Finished. Output saved to {output_path}")


if __name__ == "__main__":
    main()
