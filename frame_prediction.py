from pipeline import LPR_Pipeline
import cv2

pipeline = LPR_Pipeline(
    ocr_weights_path="weights/OCR.pth"
)

# Open the video file
video_capture = cv2.VideoCapture("demo_footage/video3.mp4")

# Set the frame number you want to read
nth_frame = 50  # Change this to the frame number you want to read

# Set the video to the nth frame
video_capture.set(cv2.CAP_PROP_POS_FRAMES, nth_frame)

# Read the nth frame from the video
success, frame = video_capture.read()
if not success:
    print(f"Failed to read frame number {nth_frame} from the video file")
    exit(1)

# Use the nth frame as the image
image = frame

# cars = pipeline.predict_cars(image)
# if len(cars) == 0:
#     print("No cars detected in the frame")
#     exit(1)

# car = cars[0]
# cv2.imshow('car', car)

plate = pipeline.predict_license_plate(image, for_affine=True)
plate = pipeline.predict_affine_plate(plate)
cv2.imshow('plate', plate)

chars = pipeline.predict_plate_characters(plate)
print(chars)

cv2.waitKey(0)

# Release the video capture object
video_capture.release()
cv2.destroyAllWindows()
