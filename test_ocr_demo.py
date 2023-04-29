import streamlit as st
import pandas as pd
import numpy as np
import copy
import requests
import os

# file processing packages
import base64
from pdf2image import convert_from_path, convert_from_bytes
from PyPDF2 import PdfFileReader
import tempfile

# ocr packages
from PIL import Image
import pytesseract
import cv2

# ocr evaluation packages
from pdf import improvingOCR  # for evaluation
from dinglehop.word_error_rate import *
from dinglehop.character_error_rate import *
from lang_confidence.lang_id import *
import nltk

nltk.download("punkt")
from spellchecker import SpellChecker
import string


# site copy
MAIN_TITLE_TEXT = "Interactive OCR Evaluation\n"
TITLE_DESCRIPTION = "*Built for http://history-lab.org/*\n"
PSM_HELP = (
    "Set Tesseract to assume a certain form of image. See resources for more info"
)
OEM_HELP = "0 = Original Tesseract only.\n\
        1 = Neural nets LSTM only.\n\
        2 = Tesseract + LSTM.\n\
        3 = Default, based on what is available."


def main():
    # Wide mode
    st.set_page_config(layout="centered")
    st.title(MAIN_TITLE_TEXT)
    st.write(TITLE_DESCRIPTION)

    # File selection UI
    start, end, uploaded_file = render_file_select_sidebar()

    st.sidebar.title("Tesseract Parameters")
    st.sidebar.markdown(
        "Have no idea what these options *actually* mean? See [Resources](#resources)",
        unsafe_allow_html=True,
    )
    psm = st.sidebar.slider("Page Segmentation Mode", 0, 13, value=1, help=PSM_HELP)
    oem = st.sidebar.slider("OCR Engine Mode", 0, 3, value=1, help=OEM_HELP)
    tess_config = get_tess_config(psm, oem)

    if uploaded_file is None or start == "" or end == "":
        render_landing_layout()

    else:  # use uploaded a file
        st.subheader("Uploaded File")

        with st.expander("File", expanded=True):
            images = get_images_from_upload(uploaded_file, start, end)
            st.image(images)

        imgs = no_threshold_images(start, end, images)

        st.markdown("---")

        st.subheader("Text Extraction")
        st.write("*Runs Tesseract with the specified parameters*")

        with st.expander("Text from Pre-Processed Image", expanded=False):
            # TODO: refactor later
            tesseract_df = run_tesseract(imgs, start, end, tess_config)
            extracted_text = extract_text(tesseract_df)
            st.write(extracted_text)
            st.write("\n")
        st.download_button(
            "Download OCR'd Text", extracted_text, file_name="ocr_results.txt"
        )

        st.markdown("---")
        st.subheader("OCR Text Quality")
        st.write(
            "*There is no standard consensus of how OCR quality should be measured. Below are a few metrics that approximate the quality of OCR. Expand each section to learn more about the metric.*"
        )

        # TODO: refactor this
        # ocr_eval_summary_file = open("ocr_eval_summary1.txt", "w")
        # ocr_eval_summary_file.write("OCR Quality Summary:\n")

        with st.expander("Garbageness Score", expanded=True):
            st.write(
                'Approximates OCR quality via a "garbageness" score. **What percent of the total words are garbage?**'
            )
            st.write(
                "Derived from Wudtke et. al, [*Recognizing Garbage in OCR Output on Historical Documents*](https://dl.acm.org/doi/pdf/10.1145/2034617.2034626)"
            )

            summary_df, garbage_df = improvingOCR.garbageDetector(extracted_text)
            st.write("**OCR Quality Summary:**")
            st.dataframe(summary_df)

            st.write("**Garbage Words:**")
            st.dataframe(garbage_df)

        st.write("\n\n")

        with st.expander("Language Confidence", expanded=True):
            st.write(
                "Approximates OCR quality via a Language Confidence score as measured by [langid](https://github.com/saffsd/langid.py/tree/master/langid). **How confident are we that the result of the OCR output is in English?**"
            )
            st.write(
                "Derived from Baumann's [*Automatic evaluation of OCR quality*](https://ryanfb.github.io/etc/2015/03/16/automatic_evaluation_of_ocr_quality.html)"
            )

            if os.path.exists("./ocr_text.txt"):
                confidence = float(getConf().decode()) * 100
                st.write("Language Confidence (en): " + str(confidence))

            else:
                st.write("Waiting for Tesseract...")

        st.write("\n\n")

        with st.expander("Mean Word Confidence", expanded=True):
            st.write(
                "Approximates OCR quality via a mean word confidence score as measured by [pytesseract](https://pypi.org/project/pytesseract/#:~:text=%23%20Get%20verbose%20data%20including%20boxes%2C%20confidences%2C%20line%20and%20page%20numbers%0Aprint(pytesseract.image_to_data(Image.open(%27test.png%27)))). **How confident is the OCR Engine that the word is this word?**"
            )
            st.write(
                "Modified from Springmann et. al's [*Automatic quality evaluation and (semi-) automatic improvement of OCR models for historical printings*](https://arxiv.org/abs/1606.05157)"
            )

            st.write("\n\n")
            st.write("Mean word confidence: " + str(tesseract_df["conf"].mean()))
            st.dataframe(tesseract_df)

        st.write("\n\n")

        with st.expander("Simple Dictionary Checking", expanded=True):
            st.write(
                "Approximates OCR quality via dictionary checking. The [enchant dictionary](https://pyenchant.github.io/pyenchant/tutorial.html) is used. **What percent of the total words are in the dictionary?**"
            )
            st.write(
                "Derived from Alex et. al's Simple Quality Score in [*Estimating and Rating the Quality of Optically Character Recognised Text*](https://dl.acm.org/doi/pdf/10.1145/2595188.2595214?casa_token=j0lV_LEjZHMAAAAA:_Bntc_y9aMmc7pbYUSVlEIPtrqC_ZyP5x0w9WsOpqTUdtjv9bTaDYNM1PT3oe0Oj--g8l7aKXG8dMw)"
            )

            st.write("\n\n")
            spell = SpellChecker()
            W_good = 0
            W_all = 1  # initialize to 1 to avoid dividing by 0

            # TODO: refactor
            misspelled_list = []
            count = 0
            # Skip numbers, punctucation, and whitespace.
            for word in nltk.word_tokenize(extracted_text):  # extracted_text.split()
                count += 1
                W_all += 1
                if (
                    word in string.punctuation
                    or word in string.whitespace
                    or word.isdigit()
                ):
                    continue
                if spell[word.lower()]:
                    W_good += 1
                else:
                    misspelled_list.append(word)
            misspelled_df = pd.Series(misspelled_list).to_frame()
            score = (W_good / W_all) * 100
            st.write("Simple Quality Score: " + str(score))
            st.write("Word Count: " + str(count))
            st.write("Num Misspelled Words: " + str(W_good))
            st.write("Misspelled Words: ")
            st.dataframe(misspelled_df)

        st.write("\n\n")

        with st.expander("Word and Character Error Rates", expanded=True):
            st.write(
                "Approximates OCR quality via word error rate (WER) and character error rate (CER) calculated by comparing against Ground Truth data."
            )
            st.write(
                "Derived from The Quator Project's open-source tool, [*dinglehopper*](https://github.com/qurator-spk/dinglehopper)."
            )

            gt_text = st.file_uploader("Upload Ground Truth Text File", type=["txt"])
            st.write("\n\n")
            if gt_text is not None:
                gt_str = gt_text.read().decode()
                wer, n_words = word_error_rate_n(gt_str, extracted_text)
                invertedwer = (1 - wer) * 100
                # cer, nc_words = character_error_rate_n(gt_str, extracted_text)
                # gt_words = words_normalized(extracted_text)
                # ocr_words = words_normalized(extracted_text)

                st.write("**Word and Character Error Rates Summary:**")
                st.write("invertedWER: " + str(invertedwer))
                # st.write("CER: " + str(cer))
                st.write("word count: " + str(n_words))
                # st.write("character count: " + str(nc_words))
                # #st.write("gt_words: " + str(gt_words))
                # st.write("ocr_words: " + str(ocr_words))

        st.write("\n\n")

        st.markdown("---")

        # TODO: render resources refactor
        st.subheader("Resources")
        st.write(
            "[Tesseract Page Segmentation Modes Explained](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)"
        )

    # For newline
    st.sidebar.write("\n")


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


@st.cache
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
    tess_config = (
        "--psm "
        + str(psm)
        + " --oem "
        + str(oem)
        + " -c "
        + "preserve_interword_spaces=1"
    )
    return tess_config


def no_threshold_images(start, end, images):
    """
    Returns PIL unmodified images in start-end range
    """
    imgs = []
    for page_idx in range(start - 1, end):
        img = cv2.cvtColor(np.array(images[page_idx]), cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(img.astype(np.uint8))
        imgs.append(img)

    return imgs


@st.cache
def run_tesseract(imgs, start, end, tess_config):
    """
    Extract text using tesseract
    """
    result_df = pd.DataFrame()
    for page_idx in range(start - 1, end):
        output = pytesseract.image_to_data(
            imgs[page_idx], config=tess_config, lang="eng", output_type="data.frame"
        )
        result_df = result_df.append(output[["conf", "text"]])

    # remove all rows with no confidence values
    result_df[result_df.conf > 0]
    return result_df


def extract_text(result_df):
    """
    Extract text from tesseract output
    """
    extracted_text = ""
    for index, row in result_df.iterrows():
        extracted_text += str(row["text"]) + " "
    return extracted_text


def render_landing_layout():
    st.info("Please upload a file and select a range")
    st.subheader("Text Extracted")
    st.info("Please Upload a file and select a range")
    st.subheader("Resources")
    st.write(
        "[Tesseract Page Segmentation Modes Explained](https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/)"
    )
    st.write(
        "[Denoising Methods: Erosion, Dilation, Opening, Closing](https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html)"
    )


def render_file_select_sidebar():
    st.sidebar.title("File Selection")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF Containing Text", type=["pdf"]
    )
    st.set_option("deprecation.showfileUploaderEncoding", False)  # Disabling warning

    st.sidebar.subheader("Enter Page Range")
    col1, col2 = st.sidebar.columns(2)
    start = col1.text_input("First Page")
    end = col2.text_input("Last Page")

    if start != "" and end != "" and uploaded_file is not None:
        uploaded_file_copy = copy.copy(
            uploaded_file
        )  # PyPDF2 malforms input pdf, need to copy
        last_page = count_num_pages(uploaded_file_copy)
        start = int(start)
        end = int(end)
        if not is_range_valid(start, end, last_page):
            st.sidebar.error("Invalid page range, last page is " + str(last_page))
    return (start, end, uploaded_file)


if __name__ == "__main__":
    main()
