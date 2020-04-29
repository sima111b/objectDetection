import cv2, os
import pandas as pd
import calc_iou
#initializing selectivesearch
ssearch = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
def extractImages(annotations,dirpath):
    numberofsamples=50
    positiveThreshold=0.7
    negativeThreshold=0.3
    train_images = []
    train_labels = []
    for entries in os.listdir(annotations):
        try:
            if entries.startswith("airplane"):
                filename = entries.split(".")[0]+".jpg"
                image = cv2.imread(os.path.join(dirpath,filename))
                gtFile = pd.read_csv(os.path.join(annotations,entries)) #csv file including groundTruthes
                groundTruth=[]
                for row in gtFile.iterrows():
                    x1 = int(row[1][0].split(" ")[0])
                    y1 = int(row[1][0].split(" ")[1])
                    x2 = int(row[1][0].split(" ")[2])
                    y2 = int(row[1][0].split(" ")[3])
                    groundTruth.append({"x1":x1,"x2":x2,"y1":y1,"y2":y2})
                ssearch.setBaseImage(image)
                ssearch.switchToSelectiveSearchFast()
                boxProposals = ssearch.process()
                image_copy = image.copy()
                counter = 0
                falsecounter = 0
                flag = 0
                foregroundFlag = 0
                backgroundFlag = 0
                for ctr,bxp in enumerate(boxProposals):
                    if ctr < 2000 and flag == 0:        #choose the first 2000 images
                        for gt in groundTruth:
                            x,y,w,h = bxp
                            iou = calc_iou(gt,{"x1":x,"x2":x+w,"y1":y,"y2":y+h})
                            if counter < numberofsamples:  #collect 50 positive examples
                                if iou > positiveThreshold:
                                    image_cropped = image_copy[y:y+h,x:x+w]
                                    resized = cv2.resize(image_cropped, (224,224), interpolation = cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(1)
                                    counter += 1
                            else :
                                foregroundFlag =1
                            if falsecounter <numberofsamples:    #collect 50 negative examples
                                if iou < negativeThreshold:
                                    image_cropped = image_copy[y:y+h,x:x+w]
                                    resized = cv2.resize(image_cropped, (224,224), interpolation = cv2.INTER_AREA)
                                    train_images.append(resized)
                                    train_labels.append(0)
                                    falsecounter += 1
                            else :
                                backgroundFlag = 1
                        if foregroundFlag == 1 and backgroundFlag == 1:
                            flag = 1
        except Exception as e:
            print(e)
            print("error in "+filename)
            continue

    return train_images,train_labels