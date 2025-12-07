import cv2
import pytesseract
from paddleocr import PaddleOCR, TextRecognition
from rapidocr import RapidOCR, ModelType

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def TesseractOCR(img_path):
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print("Hasil tesseract OCR: ", pytesseract.image_to_string(img_rgb))

def paddle(img_path):
    ocr = PaddleOCR(lang='en')
    result = ocr.predict(img_path)
    for res in result:
        res.print()

def textrecogPaddle(img_path):
    model = TextRecognition()
    output = model.predict(input=img_path)
    for res in output:
        res.print()
    
def line():
    
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    print("-----------------------------------------------------------")
    
def rapid(img_path):
    img_cv = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    engine = RapidOCR(
        params={
            "Det.model_type": ModelType.MOBILE
        }
    )
    result = engine(img_rgb)
    print(result)

if __name__ == "__main__":
    rapid("indonesia2.png")