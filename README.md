# Dataset
## Dataset preperation
1. After downloading the dataset unzip the `TL01` and `VL01`  in `라벨링데이터` folders in both training and validation. We are currently not using the `원천데이터` folder and its contents.
2. Move the `152.반려동물 피부질환 데이터` folder into `data` folder. 

After the steps above overall structure of the `data` folder looks like following:
```bash
───data
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
To reparse the json files run the `Json Parser.ipynb` file. It will generate training and validation json and csv files. Json file is organized as list of dictionary objects which corresponds to the metadata of images
