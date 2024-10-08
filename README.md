# Dataset
## Exploratory data analysis
- ~55k images, polygon and box coordinates of lesion are annotated
- We did the EDA report and cleaned the dataset inconsistencies
- 6 different lesion types, with no negative examples
To see the full EDA open [EDA html file](EDA_Pet_skin_disease.html)
![EDA example](assets/eda-example.png "EDA")

## Object detection results

![Training logs](assets/training_logs.png)

![Detection examples](assets/detection-examples.png)

## Dataset preperation
1. After downloading the dataset unzip the `TL01` and `VL01`  in `라벨링데이터` folders in both training and validation. We are currently not using the `원천데이터` folder and its contents.
2. Move the `152.반려동물 피부질환 데이터` folder into `data` folder. 

After the steps above overall structure of the `data` folder looks like following:
```bash
───data
    ├───training_dataset.csv
    ├───training_dataset.json
    ├───validation_dataset.csv
    ├───validation_dataset.json
    └───152.반려동물 피부질환 데이터
        └───01.데이터
            ├───1.Training
            │   ├───1_원천데이터
            │   └───2_라벨링데이터
            │       └───TL01
            │           ├───반려견
            │           │   └───피부
            │           │       └───일반카메라
            │           │           └───유증상
            │           │               ├───A1_구진_플라크
            │           │               ├───A2_비듬_각질_상피성잔고리
            │           │               ├───A3_태선화_과다색소침착
            │           │               ├───A4_농포_여드름
            │           │               ├───A5_미란_궤양
            │           │               └───A6_결절_종괴
            │           └───반려묘
            │               └───피부
            │                   └───일반카메라
            │                       └───유증상
            │                           └───A2_비듬_각질_상피성잔고리
            └───2.Validation
                ├───1_원천데이터
                └───2_라벨링데이터
                    └───VL01
                        ├───반려견
                        │   └───피부
                        │       └───일반카메라
                        │           └───유증상
                        │               ├───A1_구진_플라크
                        │               ├───A2_비듬_각질_상피성잔고리
                        │               ├───A3_태선화_과다색소침착
                        │               ├───A4_농포_여드름
                        │               ├───A5_미란_궤양
                        │               └───A6_결절_종괴
                        └───반려묘
                            └───피부
                                └───일반카메라
                                    └───유증상
                                        └───A2_비듬_각질_상피성잔고리
```

## Parsing the dataset 
To reparse the json files run the `Json Parser.ipynb` file. It will generate training and validation json and csv files inside the `data` folder. 

Json file is organized as list of dictionary objects which corresponds to the metadata of images

## Generating YOLO format labels
In order to train a `YOLO` model, the labels stored in the `.json` files need to be extracted and saved separately as `.txt` files following the `YOLO` format. For that purpose, a python script is available under the folder `yolo_labels`.

After running the code, the user will be prompted to select a folder. Please select the `01.데이터` present in the directory of the dataset (refer to the dataset directory tree above). After selecting the folder, the script will automatically extract and create the `.txt` files of the labels with their respective images.

## Missing image data
While generating the YOLO labels, we noticed that the following images were missing from the directory `A3_태선화_과다색소침착`:

`IMG_D_A3_112794` 
`IMG_D_A3_112826` 
`IMG_D_A3_112921` 
`IMG_D_A3_113012` 
`IMG_D_A3_113089` 
`IMG_D_A3_113145` 

## Classification using transfer learning
We tried image classifcation using transfer learning on pretrained models like resnet50, efficientnet etc but the performance (accuracy) is still low due to data inconsistancies. Most images have varying backgrounds which serve as a lot of noise while training. The next step is to extract the region of interest from the image resize and create a new dataset for better classfication results.

After cropping and resizing images the results are still very poor with low accuracy and high loss. This seems to be the result of the inconsistant image resolution when cropping and resizing images.

## Semantic segmentation
Semantic segmentation models overfitted heavily using the standard training. After exploring  the dataset we found that the objects are relatively small with 93% of them occupying less than 5% of the image. 

| Object to image ratio    | 0.5   | 0.1   | 0.05  | 0.01 | 0.005 | 0.001 |
|--------------------------|-------|-------|-------|------|-------|-------|
| Number of images < ratio | 99.9% | 97.4% | 93.2% | 65%  | 46%   | 11%   |

