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
	fpage = col1.text_input("First Page")
	lpage = col2.text_input("Last Page")

	if fpage != "" and lpage != "" and uploaded_file is not None:
		uploaded_file_copy = copy.copy(uploaded_file) # PyPDF2 malforms input pdf, need to copy
		if not isRangeValid(int(fpage), int(lpage), countNumPages(uploaded_file_copy)):
			st.sidebar.error("Out of bounds page reference, limit is blah")
	
	st.sidebar.title("Pre-Processing Parameters")
	thresh_method = st.sidebar.radio("Thresholding Method", THRESHOLDING_METHODS)
	if thresh_method == "Simple":
				thresh_val = st.sidebar.slider("Threshold Value", value=127, min_value=50, max_value=300, help=THRESHOLD_HELP)
				max_val = st.sidebar.slider("Maximum Value", value=255, min_value=100, max_value=1000, help=MAXTHRESHOLD_HELP)

	st.sidebar.title("Tesseract Parameters")

	if uploaded_file is None or fpage == "" or lpage == "":
		st.info('Please upload a file')
		st.subheader("Text Extracted")
		st.info('Please Upload a file')

	else: # use uploaded a file
		st.subheader("Uploaded File")
		with st.expander("File", expanded=True):

			# uploaded pdf -> temporary file -> image
			tfile = tempfile.NamedTemporaryFile(delete=False)
			tfile.write(uploaded_file.read())
			image = convert_from_path(tfile.name, first_page=int(fpage), last_page=int(lpage)) # in-memory image

			# display
			st.image(image)
		st.markdown("---")

		# attempt to extract text
		st.subheader("Image Pre-Processing")
		with st.expander("Pre-Processing Results", expanded=True):

			# grayscale image to improve processing
			# for image files
			# res
			# testing
			
			button = st.sidebar.button("Extract Text")
			ret,img = cv2.threshold(np.array(image[0]), thresh_val, max_val, cv2.THRESH_BINARY)
			img = Image.fromarray(img.astype(np.uint8))
			st.image(img) # display grayscaled image
		st.markdown("---")


		
		st.subheader("Text Extraction")
		test_extracted_expander = st.expander("Text from Pre-Processed Image", expanded=True)
		with test_extracted_expander:

			extracted_text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz, 0123456789.%', lang="eng")

			st.write(extracted_text)
			st.write('\n')
		
		st.markdown("---")
		st.subheader("OCR Text Quality")
		with st.container():
			
			# tokenize text
			st.write("[filler]")
			st.write("[\% garbled text]")
			

	# For newline
	st.sidebar.write('\n')

def isRangeValid(first: int, last: int, max_range: int):
	"""
	Helper For Page Range Input Errors
	"""
	if last - first < max_range:
		return True
	return False

def countNumPages(file):
	"""
	Return number of pages in a pdf
	"""
	pdfReader = PdfFileReader(file)
	return pdfReader.numPages


if __name__ == '__main__':
    main()