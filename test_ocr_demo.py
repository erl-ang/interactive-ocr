import streamlit as st
import pandas as pd
import numpy as np
import copy

# file processing packages
import base64
from pdf2image import convert_from_path, convert_from_bytes
from PyPDF2 import PdfFileReader
import tempfile

# ocr packages
from PIL import Image
import pytesseract
import cv2

# site copy
MAIN_TITLE_TEXT = 'Inspecting OCR Quality\n'
TITLE_DESCRIPTION = '*Built for http://history-lab.org/*\n'
THRESHOLD_HELP = "If the pixel value is smaller than this number, it is set to 0"
MAXTHRESHOLD_HELP = "If the pixel value is larger than the threshold, it is set to this number"
PSM_HELP = "Set Tesseract to assume a certain form of image. See resources for more info"
OEM_HELP = "0 = Original Tesseract only.\n\
	    1 = Neural nets LSTM only.\n\
	    2 = Tesseract + LSTM.\n\
	    3 = Default, based on what is available."

THRESHOLDING_METHODS = ["Simple", "Adaptive", "Otsu"]
DET_ARCHS = ['Tesseract', 'Other OCR Method to test', 'EasyOCR']

# testing
TEST_IMAGE_PATH = "test_images/plain_text.png"

def main():
	# Wide mode
	st.set_page_config(layout="centered")
	st.title(MAIN_TITLE_TEXT)
	st.write(TITLE_DESCRIPTION)

	# File selection UI
	st.sidebar.title("File Selection")
	uploaded_file = st.sidebar.file_uploader("Upload a File Containing Text",type=None)
	st.set_option('deprecation.showfileUploaderEncoding', False) # Disabling warning

	st.sidebar.subheader("Enter Page Range")
	col1, col2 = st.sidebar.columns(2)
	start = col1.text_input("First Page")
	end = col2.text_input("Last Page")

	if start != "" and end != "" and uploaded_file is not None:
		uploaded_file_copy = copy.copy(uploaded_file) # PyPDF2 malforms input pdf, need to copy
		start = int(start)
		end = int(end)
		if not is_range_valid(start, end, count_num_pages(uploaded_file_copy)):
			st.sidebar.error("Out of bounds page reference, limit is max")
	
	st.sidebar.title("Pre-Processing Parameters")
	thresh_method = st.sidebar.radio("Thresholding Method", THRESHOLDING_METHODS)
	if thresh_method == "Simple":
		thresh_val = st.sidebar.slider("Threshold Value", value=127, min_value=50, max_value=300, help=THRESHOLD_HELP)
		max_val = st.sidebar.slider("Maximum Value", value=255, min_value=100, max_value=1000, help=MAXTHRESHOLD_HELP)

	st.sidebar.title("Tesseract Parameters")
	st.sidebar.markdown("Have no idea what these options *actually* mean? See [Resources](#resources)", unsafe_allow_html=True)
	psm = st.sidebar.slider("Page Segmentation Mode", 0, 13, value=1, help=PSM_HELP)
	oem = st.sidebar.slider("OCR Engine Mode", 0, 3, value=1, help=OEM_HELP)
	tess_config = get_tess_config(psm, oem)

	if uploaded_file is None or start == "" or end == "":
		st.info('Please upload a file and select a range')
		st.subheader("Text Extracted")
		st.info('Please Upload a file and select a range')

	else: # use uploaded a file
		st.subheader("Uploaded File")
		with st.expander("File", expanded=True):
			images = get_images_from_upload(uploaded_file, start, end)
			st.image(images)
		
		st.markdown("---")

		# attempt to extract text
		st.subheader("Image Pre-Processing")
		with st.expander("Pre-Processing Results", expanded=True):

			imgs = grayscale_images(start, end, images, thresh_val, max_val)
			st.image(imgs)

		st.markdown("---")

		st.subheader("Text Extraction")
		with st.expander("Text from Pre-Processed Image", expanded=True):

			# refactor later
			for page_idx in range(start-1, end):
				extracted_text = pytesseract.image_to_string(imgs[page_idx], config=tess_config, lang="eng")
				if extracted_text == "":
					st.info("No Tesseract output :( Try tweaking more parameters!")
					break
				else:
					st.write(extracted_text)
			st.write('\n')
		
		st.markdown("---")
		st.subheader("OCR Text Quality")
		with st.container():
			
			# tokenize text
			st.info("Coming soon!")
		
		st.markdown("---")
		st.subheader("Resources")
		st.write("[Tesseract Page Segmentation Modes Explained](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)")

	# For newline
	st.sidebar.write('\n')

def is_range_valid(first: int, last: int, max_range: int):
	"""
	Helper For Page Range Input Errors
	"""
	if last - first < max_range:
		return True
	return False

def count_num_pages(file):
	"""
	Return number of pages in a pdf
	"""
	pdfReader = PdfFileReader(file)
	return pdfReader.numPages

def get_images_from_upload(file, start: int, end: int):
	"""
	Return PIL images from uploaded file
	"""
	tfile = tempfile.NamedTemporaryFile(delete=False)
	tfile.write(file.read())

	images = convert_from_path(tfile.name, first_page=start, last_page=end)
	return images

def get_tess_config(psm: int, oem: int):
	"""
	Returns formatted config for Tesseract
	"""
	tess_config = "--psm " + str(psm)+ " --oem " + str(oem) + " -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz, 0123456789.%"
	return tess_config


def grayscale_images(start, end, images, thresh_val, max_val):
	"""
	Returns PIL grayscaled images in start-end range
	"""
	imgs = []
	for page_idx in range(start-1, end):
		img = cv2.cvtColor(np.array(images[page_idx]), cv2.COLOR_BGR2GRAY)
		ret, img = cv2.threshold(img, thresh_val, max_val, cv2.THRESH_BINARY)
		img = Image.fromarray(img.astype(np.uint8))
		imgs.append(img)
	
	return imgs


if __name__ == '__main__':
    main()