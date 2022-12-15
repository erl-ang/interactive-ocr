from subprocess import run
import os

def getConf(): 
    data = run("./lang_confidence/scorelines.sh ./ocr_text.txt | ./lang_confidence/ocrquality.rb",capture_output=True,shell=True)
    print(data.stderr)
    return data.stdout