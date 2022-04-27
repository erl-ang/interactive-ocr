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
from pdf import improvingOCR # for evaluation

# site copy
MAIN_TITLE_TEXT = 'Interactive OCR Evaluation\n'
TITLE_DESCRIPTION = '*Built for http://history-lab.org/*\n'
THRESHOLD_HELP = "If the pixel value is smaller than this number, it is set to 0"
MAXTHRESHOLD_HELP = "If the pixel value is larger than the threshold, it is set to this number"
PSM_HELP = "Set Tesseract to assume a certain form of image. See resources for more info"
OEM_HELP = "0 = Original Tesseract only.\n\
	    1 = Neural nets LSTM only.\n\
	    2 = Tesseract + LSTM.\n\
	    3 = Default, based on what is available."
DENOISE_HELP = "" # TODO

THRESHOLDING_METHODS = ["None", "Simple"] # ["None", "Simple", "Adaptive", "Otsu"]
DENOISING_METHODS = ["None", "Erosion", "Dilation", "Opening", "Closing"]
DET_ARCHS = ['Tesseract', 'Other OCR Method to test', 'EasyOCR']

def main():
	# Wide mode
	st.set_page_config(layout="centered")
	st.title(MAIN_TITLE_TEXT)
	st.write(TITLE_DESCRIPTION)

	# File selection UI
	start, end, uploaded_file = render_file_select_sidebar()
	
	st.sidebar.title("Pre-Processing Parameters")
	thresh_method = st.sidebar.radio("Thresholding Method", THRESHOLDING_METHODS)
	if thresh_method == "Simple" or thresh_method == "None": # TODO: diff logic
		thresh_val = st.sidebar.slider("Threshold Value", value=127, min_value=50, max_value=300, help=THRESHOLD_HELP)
		max_val = st.sidebar.slider("Maximum Value", value=255, min_value=100, max_value=1000, help=MAXTHRESHOLD_HELP)
	denoise_selection = st.sidebar.radio("Denoising Methods", DENOISING_METHODS)


	st.sidebar.title("Tesseract Parameters")
	st.sidebar.markdown("Have no idea what these options *actually* mean? See [Resources](#resources)", unsafe_allow_html=True)
	psm = st.sidebar.slider("Page Segmentation Mode", 0, 13, value=1, help=PSM_HELP)
	oem = st.sidebar.slider("OCR Engine Mode", 0, 3, value=1, help=OEM_HELP)
	tess_config = get_tess_config(psm, oem)

	if uploaded_file is None or start == "" or end == "":
		render_landing_layout()

	else: # use uploaded a file
		main_col1, main_col2 = st.columns(2)
		main_col1.subheader("Uploaded File")
		
		with main_col1.expander("File", expanded=True):
			images = get_images_from_upload(uploaded_file, start, end)
			st.image(images)
		

		# attempt to extract text
		main_col2.subheader("Image Pre-Processing")
		with main_col2.expander("Pre-Processing Results", expanded=True):
			imgs = grayscale_images(start, end, images, thresh_val, max_val)
			imgs = PIL_to_np(start, end, imgs)
			imgs = denoise_images(start, end, imgs, denoise_selection)
			st.image(imgs)

		st.markdown("---")

		st.subheader("Text Extraction")
		

		with st.expander("Text from Pre-Processed Image", expanded=True):

			# refactor later
			extracted_text = ""
			for page_idx in range(start-1, end):
				extracted_text += pytesseract.image_to_string(imgs[page_idx], config=tess_config, lang="eng")
				if extracted_text == "":
					st.info("No Tesseract output :( Try tweaking more parameters!")
					break
				else:
					st.write(extracted_text)
			st.write('\n')
		
		st.markdown("---")
		st.subheader("OCR Text Quality")
		with st.container():
			text_file = open("ocr_text.txt", "w")
			text_file.write(extracted_text)
			text_file.close()
			result_df = improvingOCR.garbageDetector("ocr_text.txt")
			st.dataframe(result_df)
		
		st.markdown("---")

		# TODO: render resources refactor
		st.subheader("Resources")
		st.write("[Tesseract Page Segmentation Modes Explained](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)")
		st.write("[Denoising Methods: Erosion, Dilation, Opening, Closing](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html)")


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
	# tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz, 0123456789.%
	tess_config = "--psm " + str(psm)+ \
		" --oem " + str(oem) + \
		" -c " + \
		"preserve_interword_spaces=1"
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

def denoise_image(img, denoise_method):
	"""
	Returns image with specified denoising method applied
	"""
	kernel = np.ones((1, 1), np.uint8)
	if denoise_method == "None":
		pass
	elif denoise_method == "Erosion":
		img = cv2.erode(img, kernel, iterations=1)
	elif denoise_method == "Dilation":
		img = cv2.dilate(img, kernel, iterations=1)
	elif denoise_method == "Opening":
		img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
	elif denoise_method == "Closing":
		img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
	
	return img

def denoise_images(start, end, images, denoise_method):
	"""
	Denoises a series of images
	"""
	imgs = []
	for page_idx in range(start-1, end):
		img = denoise_image(images[page_idx], denoise_method)
		imgs.append(img)
	
	return imgs

def PIL_to_np(start, end, pils):
	"""
	converts PIL objects to np arrays for cv2 to use
	"""
	imgs = []
	for page_idx in range(start-1, end):
		img = np.array(pils[page_idx])
		imgs.append(img)

	return imgs

def render_landing_layout():
	st.info('Please upload a file and select a range')
	st.subheader("Text Extracted")
	st.info('Please Upload a file and select a range')
	st.subheader("Resources")
	st.write("[Tesseract Page Segmentation Modes Explained](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)")
	st.write("[Denoising Methods: Erosion, Dilation, Opening, Closing](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html)")

def render_file_select_sidebar():
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
	
	return (start, end, uploaded_file)

def render_preprocess_sidebar():
	st.sidebar.title("Pre-Processing Parameters")
	thresh_method = st.sidebar.radio("Thresholding Method", THRESHOLDING_METHODS)
	if thresh_method == "Simple":
		thresh_val = st.sidebar.slider("Threshold Value", value=127, min_value=50, max_value=300, help=THRESHOLD_HELP)
		max_val = st.sidebar.slider("Maximum Value", value=255, min_value=100, max_value=1000, help=MAXTHRESHOLD_HELP)
	denoise_selection = st.sidebar.radio("Denoising Methods", DENOISING_METHODS)
	return (thresh_val, max_val, denoise_selection)

if __name__ == '__main__':
    main()