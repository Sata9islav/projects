# A neural network for predicting the calorie content of dishes

## Structure

```
nn_for_predicting_the_calorie/
├── data/              
│   ├── images/
│   ├── ingredients.csv
│   └── dish.csv  
│
├── scripts/                
│   ├── dataset.py   
|   └── utils.py   
│
├── models/
|
├── solution.ipynb   
└── requirements.txt
```

## Task description

Create and train a neural network that will predict the calorie content of meals. The resulting model can be integrated into many fitness and health applications, and it will be useful for those who monitor their health.

## Job description

A photo and a list of ingredients are sent to the entrance. In the example in the photo, salmon with grilled vegetables and pesto sauce. Ingredients: salmon, zucchini, eggplant, pepper, champignons, cherry tomatoes, pesto sauce. At the output, we get the calorie content of the dish — in the example, it turned out to be 631 kCal.

## Goal

Get a DL model that can estimate the calorie content of a dish

## Target metric

 The accuracy of the MAE estimate is < 50 (MAE — Mean Absolute Error) in the test sample.

## Dataset

A dataset (1.3 GB) that contains photos of dishes, a description of the ingredients, as well as the total serving weight and calorie content.

### data/ingredients.csv

- id - is the ingredient ID.
- ingr -  is the ingredient name.

### data/dish.csv

- dish_id - is the dish ID.
- total_calories — the total number of calories. This is the target variable.
- total_mass — the mass of the dish.
- ingredients — a list of all ingredient IDs in the format ingr_0000000122;ingr_0000000026;.., where the non-zero part corresponds to the ingredient ID from data/ingredients.csv.
- split — a label indicating where to take the dish: train/test. This tag will help you split the dataset during training.

### data/images

Dataset with photos of dishes, each directory corresponds to the dish_id from data/dish.csv and contains an rgb.png photo of the dish.
