import cv2
import pytesseract
import pandas as pd
from datetime import datetime
import os
import glob

# 🔧 Set the Tesseract OCR path (only needed for Windows)
# Uncomment and update this path if Tesseract is not in your PATH environment variable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# 📁 Hardcoded folder path containing images
IMAGE_FOLDER = r"E:\NumberPlate_project\plates"   # <-- Change this path to your folder
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


def extract_plate_text(image_path):
    """Extract text from cropped number plate images with aggressive preprocessing."""
    try:
        print("🔍 Extracting text using Tesseract OCR...")

        # Read image
        img = cv2.imread(image_path)

        # Step 1: Upscale to improve clarity (super important for small crops)
        h, w = img.shape[:2]
        scale = max(2, int(400 / w))  # upscale until ~400px wide
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Enhance contrast using CLAHE (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Step 4: Reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # Step 5: Adaptive threshold for robust binarization
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 35, 10
        )

        # Step 6: Invert if needed (text darker than background)
        white_pixels = cv2.countNonZero(thresh)
        if white_pixels < thresh.size / 2:
            thresh = cv2.bitwise_not(thresh)

        # Step 7: Morphological closing to connect broken letters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Step 8: OCR using whitelist and single-line mode
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        text = text.strip().replace(" ", "").replace("\n", "")

        # Debugging: show intermediate result if nothing detected
        if not text:
            cv2.imshow("Preprocessed", thresh)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()

        if text:
            print(f"✅ Extracted Plate Text: {text}")
        else:
            print("⚠️ No text detected in image.")

        return text
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
