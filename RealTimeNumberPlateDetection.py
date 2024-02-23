import cv2
import easyocr
import pandas as pd

harcascade = "models/haarcascade_plate_number.xml"

cap = cv2.VideoCapture(0)
cap.set(3, 640) # width
cap.set(4, 480) # height

min_area = 500
count = 0

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Initialize dataframe to store plate numbers
plate_data = pd.DataFrame()

def write_csv(text, score, output_path):
    with open(output_path, 'a') as f:
        license_plate_text = text
        license_plate_score = score
        f.write('{},{}\n'.format(license_plate_text, license_plate_score))

    print("CSV write operation completed.")


while True:
    success, img = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x,y,w,h) in plates:
        area = w*h

        if area>min_area:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255),2)

            img_roi = img[y:y+h, x:x+w]
            cv2.imshow("ROI", img_roi)
            
            # Perform OCR on the plate region
            result = reader.readtext(cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))
            print(result)
            plate_text = result[0][1] if len(result) > 0 else "Unable to detect"
            text = plate_text.upper().replace(' ', '')

            score = -1
            if result:
                score = result[0][2]
                print("Score:", score)
            else:
                print("No text detected.")

            write_csv(text, score, 'data/data.csv')

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF==ord('s'):
        cv2.imwrite("images/image_"+ str(count) + ".jpg", img_roi)
        
        cv2.rectangle(img, (0, 200), (640, 300), (0,255,0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255), 2)
        cv2.imshow("Results", img)
        cv2.waitKey(500)
        count+=1

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()

