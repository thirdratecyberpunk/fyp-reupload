import csv
import os

txtOutput = open("chaucer_texts/output/englishToChaucerModel.txt", "w+")
with open('chaucer_texts/output/englishToChaucerModel.csv', 'w') as csvOutput:
    for file in [f for f in os.listdir('chaucer_texts/reversed-line-by-line') if f.endswith('.txt')]:
        fileToOpen = open('chaucer_texts/reversed-line-by-line/' + file, "r")
        csvWriter = csv.writer(csvOutput, delimiter=",", quotechar="|", quoting=csv.QUOTE_ALL)
        print ("Processing " + file + "...")
        i = 1
        lineMerge = ""
        for line in fileToOpen:
            line = line[:-1]
            if (i % 2 == 1):
                lineMerge = "" + line + "\t"
            else:
                lineMerge += line + "\n"
                txtOutput.write(lineMerge)
            csvWriter.writerow(line)
            i = i + 1
