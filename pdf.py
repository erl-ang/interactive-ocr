import argparse
import pandas as pd
from collections import Counter
import PyPDF2
import rmgarbage as rmgarbage
from rmgarbage import Rmgarbage


class improvingOCR:
    def __init__(self):
        pass

    def garbageDetector(filepath, is_filepath=False):
        """Stores garbage text in text file provided by filepath in excel sheet.
        Paramaters: filepath
        File path to text file
        is_file_content: 1 if first param is content, 0 if filepath
        """
        content = ""
        if is_filepath:
            fileName = (filepath).rsplit("/", 1)[-1]
            with open(filepath) as file:
                content = file.read()

        else:
            content = filepath

        words = content.split()
        garbagecount = 0
        wordcount = 0
        garbageWords = []
        garbage = Rmgarbage()
        garbage.__init__()

        for word in words:
            isGarbage = garbage.is_garbage(word)
            wordcount += 1
            if isGarbage != False:
                garbageWords.append(word)
                garbagecount += 1

        frequency = Counter(garbageWords)
        # stringWriter = fileName + '.xlsx'

        # writer = pd.ExcelWriter(stringWriter, engine='xlsxwriter', engine_kwargs={
        # 			'options': {'strings_to_formulas': False, 'strings_to_urls': False}})
        df = pd.DataFrame.from_records(
            frequency.most_common(), columns=["page", "count"]
        )
        # df.to_excel(writer, sheet_name='Sheet1', index=False)
        # writer.save()

        ratio = 0

        # account for divide by 0 error
        if wordcount == 0:
            ratio = 100

        else:
            ratio = 100 - ((garbagecount / wordcount) * 100)
            ratio = int(ratio)

        summary = [[wordcount, garbagecount, ratio]]
        summaryTable = pd.DataFrame(
            summary, columns=["Number of Words", "Number of Garbage Words", "Score"]
        )
        return summaryTable, df

    def cli():
        """Process command line arguments."""
        parser = argparse.ArgumentParser(description="Process OCR Output")
        parser.add_argument("ocrFile")
        args = parser.parse_args()
        fileName = (filepath).rsplit("/", 1)[-1]

        with open(args.ocrFile) as file:
            content = file.read()
            words = content.split()
            garbagecount = 0
            wordcount = 0
            garbageWords = []
            garbage = Rmgarbage()
            garbage.__init__()

            for word in words:
                isGarbage = garbage.is_garbage(word)
                wordcount += 1
                if isGarbage != False:
                    garbageWords.append(word)
                    garbagecount += 1

            frequency = Counter(garbageWords)

            stringWriter = fileName + ".xlsx"

            # writer = pd.ExcelWriter(stringWriter, engine='xlsxwriter', engine_kwargs={
            # 			'options': {'strings_to_formulas': False, 'strings_to_urls': False}})

            df = pd.DataFrame.from_records(
                frequency.most_common(), columns=["page", "count"]
            )

            # df.to_excel(writer, sheet_name='Sheet1', index=False)

            # writer.save()

            ratio = 0

            if wordcount == 0:
                ratio = 100

            else:
                ratio = 100 - ((garbagecount / wordcount) * 100)
                ratio = int(ratio)

            summary = [[fileName, wordcount, garbagecount, ratio]]

            summaryTable = pd.DataFrame(
                summary,
                columns=[
                    "File Name",
                    "Number of Words",
                    "Number of Garbage Words",
                    "Score",
                ],
            )

            print(summaryTable)
            print()
