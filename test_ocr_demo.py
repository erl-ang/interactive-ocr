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
MAIN_TITLE_TEXT = 'Inspecting Text Quality of OCR\n'
TITLE_DESCRIPTION = '*Built for http://history-lab.org/*\n'

DET_ARCHS = ['Tesseract', 'Other OCR Method to test', 'EasyOCR']
NER_ARCHS = ["filler1", "filler2", "filler3"]

# testing
TEST_IMAGE_PATH = "test_images/plain_text.png"


def main():
	# Wide mode
	st.set_page_config(layout="centered")
	st.title(MAIN_TITLE_TEXT)
	st.write(TITLE_DESCRIPTION)

	# File selection UI
	st.sidebar.title("File Selection")
	st.subheader("Uploaded Image")
	uploaded_file = st.sidebar.file_uploader("Upload a File",type=None)
	st.set_option('deprecation.showfileUploaderEncoding', False) # Disabling warning

	# TODO: find min_value, max_value of the file,
	st.sidebar.subheader("Enter Page Range")
	col1, col2 = st.sidebar.columns(2)
	fpage = col1.text_input("First Page")
	lpage = col2.text_input("Last Page")
	# fpage, lpage = "blah", "blah"

	if fpage != "" and lpage != "" and uploaded_file is not None:
		uploaded_file_copy = copy.copy(uploaded_file) # PyPDF2 malforms input pdf, need to copy
		if not isRangeValid(int(fpage), int(lpage), countNumPages(uploaded_file_copy)):
			st.sidebar.error("Out of bounds page reference, limit is blah")
	
	# Model selection -- setting up UI for future use
	st.sidebar.title("Model selection")
	ner_arch = st.sidebar.selectbox("NER Model", NER_ARCHS)
	button = st.sidebar.button("Extract Text")

	if uploaded_file is None or fpage == "" or lpage == "":
		st.info('Please upload a file')
		st.subheader("Text Extracted")
		st.info('Please Upload a file')

	else: # use uploaded a file
		with st.container():
			# uploaded pdf -> temporary file -> image
			tfile = tempfile.NamedTemporaryFile(delete=False)
			tfile.write(uploaded_file.read())
			image = convert_from_path(tfile.name, first_page=int(fpage), last_page=int(lpage)) # in-memory image

			# select a range of pages to do text processing on
			# image[0]

			# display
			st.image(image)
			# st.write(pdf_html, unsafe_allow_html=True)
			st.markdown("---")

		# attempt to extract text
		st.subheader("Text Extracted")
		test_extracted_expander = st.expander("Text from Image", expanded=True)
		with test_extracted_expander:

			# grayscale image to improve processing
			# for image files
			# img = Image.open(timage.name).convert("L")
			# ret,img = cv2.threshold(np.array(img), 90, 400, cv2.THRESH_BINARY)

			# testing
			ret,img = cv2.threshold(np.array(image[0]), 90, 400, cv2.THRESH_BINARY)
			img = Image.fromarray(img.astype(np.uint8))
			st.image(img) # display grayscaled image

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