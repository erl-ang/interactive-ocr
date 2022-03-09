import streamlit as st
import base64
import pandas as pd
import numpy as np

# ocr
from PIL import Image
import pytesseract
import cv2

# found a temp file method to work with cv2
import tempfile


# global vars for site copy
MAIN_TITLE_TEXT = 'Inspecting Text Quality of OCR\n'
TITLE_DESCRIPTION = '*Built for http://history-lab.org/*\n'

DET_ARCHS = ['Tesseract', 'Other OCR Method to test', 'EasyOCR']
NER_ARCHS = ["filler1", "filler2", "filler3"]

# testing
TEST_IMAGE_PATH = "test_images/plain_text.png"

def main():
	# Wide mode
	st.set_page_config(layout="centered")
	# Designing the interface
	st.title(MAIN_TITLE_TEXT)
	st.write(TITLE_DESCRIPTION)
	st.subheader("Uploaded Image")

	"""Sidebar"""
	# File selection
	st.sidebar.title("Image selection")

	# Disabling warning
	st.set_option('deprecation.showfileUploaderEncoding', False)

	# Choose your own image
	uploaded_file = st.sidebar.file_uploader("Upload files", type=None)

	# Model selection -- setting up UI for future use
	st.sidebar.title("Model selection")
	det_arch = st.sidebar.selectbox("OCR Method", DET_ARCHS)
	ner_arch = st.sidebar.selectbox("NER Model", NER_ARCHS)
	button = st.sidebar.button("Analyze Page")

	if uploaded_file is None:
		st.info('Please upload an image')
		st.subheader("Text Extracted")
		st.info('Please Upload an image')

	else: # use uploaded a file
		with st.container():
			# Displaying markdown instead
			# pdf_html = get_pdf_html_iframe(uploaded_file)

			# how to get path 
			# image = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_COLOR)
			tfile = tempfile.NamedTemporaryFile(delete=True)
			tfile.write(uploaded_file.read())
			image = cv2.imread(tfile.name, cv2.IMREAD_COLOR)
			st.image(image)
				
			# st.write(pdf_html, unsafe_allow_html=True)
			st.markdown("---")

		# attempt to extract text
		st.subheader("Text Extracted")
		test_extracted_expander = st.expander("Text from Image", expanded=True)
		with test_extracted_expander:

			# grayscale image to improve processing
			# for image paths 
			img = Image.open(tfile.name).convert("L")
			# img = Image.open().convert("L")
			ret,img = cv2.threshold(np.array(img), 90, 400, cv2.THRESH_BINARY)
			img = Image.fromarray(img.astype(np.uint8))
			st.image(img)

			extracted_text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz, 0123456789.%', lang="eng")

			st.write(extracted_text)
			st.write('\n')
		
		st.markdown("---")
		st.subheader("OCR Text Quality")
		with st.container():
			
			# tokenize text
			st.write("filler")
			

	# For newline
	st.sidebar.write('\n')


def get_html(html: str):
    """
    Convert HTML so it can be rendered.
    """
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    formatted_html = WRAPPER.format(html)
    styled_html = "<style>mark.entity { display: inline-block }</style>" + formatted_html
    return styled_html



def get_pdf_html_iframe(file):
	base64_pdf = base64.b64encode(file.read()).decode('utf-8')
	pdf_html_iframe = f'<iframe src="data:application/pdf;base64,{base64_pdf}"\
			 width="700" height="400" type="application/pdf">\
			</iframe>'
	return pdf_html_iframe

if __name__ == '__main__':
    main()