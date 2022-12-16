from subprocess import run
import os

def getConf(): 
    data = run("./lang_confidence/scorelines.sh ./ocr_text.txt | ./lang_confidence/ocrquality.rb",capture_output=True,shell=True)
    return data.stdout