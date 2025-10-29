import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import os
import glob
import re

# 🔧 Tesseract OCR path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 📁 Folder path
IMAGE_FOLDER = r"E:\NumberPlate_project\plates"
EXCEL_FILE = 'number_plates.xlsx'


def initialize_excel():
    """Create or load Excel file with headers if it doesn't exist."""
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame(columns=['Date', 'Time', 'Plate Number'])
        df.to_excel(EXCEL_FILE, index=False)
    return EXCEL_FILE


def get_latest_image(folder_path):
    """Return the most recently added image file from the folder."""
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    if not image_files:
        print("❌ No image files found in the folder.")
        return None

    latest_image = max(image_files, key=os.path.getctime)
    print(f"🖼️ Latest image found: {os.path.basename(latest_image)}")
    return latest_image


def clean_text(text):
    """Clean OCR text by removing unwanted characters and normalizing."""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)  # Keep only A-Z and 0-9
    return text


def extract_plate_pattern(text):
    """
    Extract valid Indian number plate pattern from OCR output.
    Handles messy cases like MH15GF5187, MH15.GF5187, MH15 GF 5187, etc.
    """
    cleaned = clean_text(text)
    pattern = r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}'
    match = re.search(pattern, cleaned)
    if match:
        return match.group(0)
    else:
        return None


def extract_plate_text(image_path):
    """Extract text from cropped number plate images with preprocessing + pattern matching."""
    try:
        print("🔍 Extracting text using Tesseract OCR...")

        # Read image
        img = cv2.imread(image_path)

        # Step 1: Upscale to improve clarity
        h, w = img.shape[:2]
        scale = max(2, int(400 / w))
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Step 4: Noise reduction
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Step 5: Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 35, 10
        )

        # Step 6: Invert if necessary
        white_pixels = cv2.countNonZero(thresh)
        if white_pixels < thresh.size / 2:
            thresh = cv2.bitwise_not(thresh)

        # Step 7: Morph closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 8: OCR with whitelist
        config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        raw_text = pytesseract.image_to_string(thresh, config=config)

        # Clean and join multiline output
        raw_text = raw_text.replace("\n", "").replace(" ", "")
        print(f"🧾 Raw OCR Output: {raw_text}")

        # Pattern extraction
        plate_text = extract_plate_pattern(raw_text)

        if plate_text:
            print(f"✅ Extracted Plate Text: {plate_text}")
        else:
            print("⚠️ No valid number plate pattern detected.")
            # Show preprocessing if debugging
            cv2.imshow("Preprocessed (No Text Detected)", thresh)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

        return plate_text

    except Exception as e:
        print(f"❌ Error during text extraction: {e}")
        return None


def save_to_excel(plate_text):
    """Save the extracted plate number to Excel with timestamp."""
    try:
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        time_str = now.strftime('%H:%M:%S')

        df = pd.read_excel(EXCEL_FILE)
        new_row = pd.DataFrame([[date_str, time_str, plate_text]], columns=['Date', 'Time', 'Plate Number'])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        print(f"💾 Saved plate number: {plate_text}")
    except Exception as e:
        print(f"❌ Error saving to Excel: {e}")


def main():
    """Main function to process the latest image in the folder."""
    initialize_excel()
    latest_image = get_latest_image(IMAGE_FOLDER)

    if latest_image:
        plate_text = extract_plate_text(latest_image)
        if plate_text:
            save_to_excel(plate_text)
        else:
            print("⚠️ No text was extracted from the latest image.")
    else:
        print("❌ No valid image found to process.")


if __name__ == "__main__":
    main()
