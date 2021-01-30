import argparse
import glob
import os
import cv2
import editdistance
import numpy as np
from PIL import Image
from DataLoaderIAM import DataLoaderIAM, Batch
from Model import Model, DecoderType
from SamplePreprocessor import preprocess
from path import Path
import sys
import re
##############################################################################################################
#############################   SEGMENTATION   ###################################
image = cv2.imread(r'./static/input/test.png')
image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#cv2.imshow('orig',image)
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,100), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
#cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

image_to_detect = []
for i, ctr in enumerate(sorted_ctrs):

    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi=image[y:y+h, x:x+w]
    image_to_detect.append(roi)
    # show ROI
    #cv2.imshow('segment no:'+str(i),roi)
    cv2.rectangle(image,(x-1,y-1),( x + w, y + h ),(90,0,255),2)

        #os.system("main.py")
    #os.system("main.py")
    cv2.waitKey(0)
#for items in image_to_detect:
 #   os.system("main.py")
#print(image_to_detect)
# This line is having error
# cv2.imshow('marked areas',image)
cv2.waitKey(0)
word = []
for i in range(0, len(image_to_detect)):
    img = Image.fromarray(image_to_detect[i], 'RGB')
    img.save('./temporary/my{}.png'.format(i))
############################################################################################
#################################    WORD SEGMENTATION   ###################################
    image = cv2.imread(r'./temporary/my{}.png'.format(i))
    image = cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)
    # cv2.imshow('orig',image)
    # cv2.waitKey(0)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)
    cv2.waitKey(0)

    # binary

    ret, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('second',thresh)
    cv2.waitKey(0)

    # dilation
    kernel = np.ones((5, 15), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    # cv2.imshow('dilated',img_dilation)
    cv2.waitKey(0)

    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])


    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]
        word.append(roi)
        # show ROI
        # cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(image, (x - 3, y - 3), (x + w, y + h), (90, 0, 255), 2)

        # os.system("main.py")
        # os.system("main.py")
        cv2.waitKey(0)
    # for items in image_to_detect:
    #   os.system("main.py")
    # print(image_to_detect)
    # This line is having error
    # cv2.imshow('marked areas', image)
    cv2.waitKey(0)
    for i in range(0, len(word)):
        img = Image.fromarray(word[i], 'RGB')
        img.save('./temporary/words/my{}.png'.format(i))

############################################################################################
############################################################################################

class FilePaths:
    "filenames and paths to data"
    fnCharList = r'./model/charList.txt'
    fnAccuracy = r'./model/accuracy.txt'
    #fnInfer = (r'C:\Users\ABHISHEK MOHARIR\PycharmProjects\htr\SimpleHTR\data\test.png')

    fnCorpus = r'./data/corpus.txt'


def train(model, loader):
    "train NN"
    epoch = 0  # number of training epochs since start
    bestCharErrorRate = float('inf')  # best valdiation character error rate
    noImprovementSince = 0  # number of epochs no improvement of character error rate occured
    earlyStopping = 25  # stop training after this number of epochs without improvement
    while True:
        epoch += 1
        print('Epoch:', epoch)

            # train
        print('Train NN')
        loader.trainSet()
        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print(f'Epoch: {epoch} Batch: {iterInfo[0]}/{iterInfo[1]} Loss: {loss}')

            # validate
        charErrorRate = validate(model, loader)

            # if best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                f'Validation character error rate of saved model: {charErrorRate * 100.0}%')
        else:
            print(f'Character error rate not improved, best so far: {charErrorRate * 100.0}%')
            noImprovementSince += 1

            # stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print(f'No more improvement since {earlyStopping} epochs. Training stopped.')
            break


def validate(model, loader):
    "validate NN"
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print(f'Batch: {iterInfo[0]} / {iterInfo[1]}')
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                    '"' + recognized[i] + '"')

        # print validation result
    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print(f'Character error rate: {charErrorRate * 100.0}%. Word accuracy: {wordAccuracy * 100.0}%.')
    return charErrorRate


#def infer(model, fnImg):
#    global final_string

#    "recognize text in image provided by file path"
#    img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
#    batch = Batch(None, [img])
#    (recognized, probability) = model.inferBatch(batch, True)
#    final_string.append(recognized)
##    print(f'Recognized: "{recognized[0]}"')
#    print(f'Probability: {probability[0]}')
#    print(final_string)

#data = image_to_detect
#for i in range(0, len(data)):
    #img = Image.fromarray(data[i], 'RGB')
    #img.save('G:\location htr img\my{}.png'.format(i))
 #('G:\location htr img\my{}.png'.format(i))
final_string =[]

def main():
    s = ' '
    trial_string = ' '
    "main function"
            # optional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--wordbeamsearch', help='use word beam search instead of best path decoding',
                                action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    parser.add_argument('--fast', help='use lmdb to load images', action='store_true')
    parser.add_argument('--data_dir', help='directory containing IAM dataset', type=Path, required=False)
    parser.add_argument('--batch_size', help='batch size', type=int, default=100)

    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch
    elif args.wordbeamsearch:
        decoderType = DecoderType.WordBeamSearch

            # train or validate on IAM dataset
    if args.train or args.validate:
                # load training data, create TF model
        loader = DataLoaderIAM(args.data_dir, args.batch_size, Model.imgSize, Model.maxTextLen, args.fast)

            # save characters of model for inference mode
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))

                # save words contained in dataset into file
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))

                # execute training or validation
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)

            # infer text on test image
    else:
        print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)
        for i in range(0, len(word)):
            fnInfer = ('./temporary/words/my{}.png'.format(i))
            #infer(model, fnInfer)
            fnImg = fnInfer
            img = preprocess(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), Model.imgSize)
            batch = Batch(None, [img])
            (recognized, probability) = model.inferBatch(batch, True)
            final_string.append(recognized[0])
            print(f'Recognized: {recognized[0]}')
            print(f'Probability: {probability[0]}')
            os.remove('./temporary/words/my{}.png'.format(i))
        print(final_string)
        trial_string = trial_string.join(str(item) for item in final_string)

        print(trial_string)
        file1 = open("testfile.txt", "w")
        file1.write(trial_string)
        file1.close()
        # return 0
        #os.remove('G:\location htr img\words\my{}.png'.format(i))

#for i in range(0,len(data)):


if __name__ == '__main__':
    main()
