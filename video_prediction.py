from pipeline import LPR_Pipeline
import cv2

pipeline = LPR_Pipeline(
    ocr_weights_path="weights/OCR.pth"
)

cap = cv2.VideoCapture("demo_footage/video4.mp4")

# Specify the codec and create VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    print(current_frame)
    
    # cars = pipeline.predict_cars(frame)
    cars = [frame]
    if cars:
        car = cars[0]
        plate = pipeline.predict_license_plate(car, for_affine=True)
        if plate is None:
            continue
        else:
            plate = pipeline.predict_affine_plate(plate)
            chars = pipeline.predict_plate_characters(plate)
            
            # Resize the cropped plate to make it larger
            scale_factor = 3
            plate_large = cv2.resize(plate, (0, 0), fx=scale_factor, fy=scale_factor)
            plate_height, plate_width = plate_large.shape[:2]
            frame[0:plate_height, 0:plate_width] = plate_large
            
            # Prepare the text background
            font_scale = 2.5
            font_thickness = 3
            text_size, _ = cv2.getTextSize(chars, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_width, text_height = text_size
            background_x = plate_width + 10
            background_y = text_height + 20
            cv2.rectangle(frame, (background_x, 0), (background_x + text_width + 20, background_y), (255, 255, 255), -1)
            
            # Display the predicted characters next to the plate
            cv2.putText(frame, chars, (background_x + 10, text_height + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('frame', frame)

    # Print the current frame number and the total number of frames
    print(f"Frame: {current_frame}/{total_frames}")
    current_frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
