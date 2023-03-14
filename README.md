# Data-Science-Portfolio-Projects
The repo contains details of all my data science projects. 

<hr>

![](https://d2r55xnwy6nx47.cloudfront.net/uploads/2022/02/SCALING_NETS_2880x1620_Lede.svg)

<hr>

<hr>

# ML & DL Web Apps

<hr>

# Market Segmentation in Insurance | Streamlit Web App
- `Streamlit application URL`: https://customer-segmmentation-app.onrender.com
- `Streamlit application UI Interface screenshot`
![image](https://user-images.githubusercontent.com/41443395/224965383-66d6ed94-f848-4f1f-b654-cf489c94c3b1.png)

-`Model Output`
![image](https://user-images.githubusercontent.com/41443395/224971300-f0c415f9-f667-4e5e-ba4f-c9b042a3d3ba.png)

-`Objective` :
This case requires to develop a customer segmentation to give recommendations like saving plans, loans, wealth management, etc. on target customer groups.
<img align="center" src="https://user-images.githubusercontent.com/34673684/137431219-a5d99ac4-ce63-4435-8a49-4e19b09d0a07.png" alt="image">
-`Data Description`: 
The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.
- `Data`:  
Use the below link to download the Data Set:[here](https://github.com/pik1989/MarketSegmentation/blob/main/Clustered_Customer_Data.csv) 
- `Algorithms used`:  
In this dataset i've used clustering algorithm to perform segmentation.
- [K-Means Clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- `Final Model`:
I have created a Streamlit Application based on this clustering technique, taking the cuand stomer details & identifying which cluster the custoemr belongs to.

<hr>

# Deep Learning Projects

<hr>

 - ## Training Generative Adversarial Networks (GANs) | PyTorch | Generative Modeling ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.org/github/Raghu-murugankutty/Deep-Learning-/blob/main/Training%20Generative%20Adversarial%20Networks%20%28GANs%29%20in%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains over 63,000 cropped anime faces, we are generating fake images from the existing real images using generative adversarial networks (GANs).
    - `Dataset`: Anime Face Dataset, which consists of over 63,000 cropped anime faces.
    - `Discriminator Network:`
    The discriminator takes an image as input, and tries to classify it as "real" or "generated". 
In this sense, it's like any other neural network. We'll use a convolutional neural networks (CNN) which outputs a single number output for every image. 
used stride of 2 to progressively reduce the size of the output feature map. 
    - `Activation function:` Used Leaky ReLU activation for the discriminator.
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290483-2e2bb625-afc7-4ec0-af85-e41882c7997a.png">

    - `Generator Network:`
The input to the generator is typically a vector or a matrix of random numbers (referred to as a latent tensor) which is used as a seed for generating an image. 
The generator will convert a latent tensor of shape (128, 1, 1) into an image tensor of shape 3 x 28 x 28. 
To achive this,I used the ConvTranspose2d layer from PyTorch, which is performs to as a transposed convolution (deconvolution)
    - `Activation Function:` The ReLU activation is used in the generator with the exception of the output layer which uses the Tanh function.
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290517-8b8d4bc0-5e18-4be8-b692-b36371cfe90d.png">
    - I have to build a GAN with CNN architecture using PyTorch to achieve `accuracy of ~93.5%`.
    - `Real Images:`
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290297-45c481cf-1170-4152-8e55-c9e8f496abe7.png">
    - `Fake generated images using (GAN):`
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290414-89ad07c8-2937-482c-a7b6-de8b92d4bd1a.png">

<hr>

 - ## Image classification using ResNets | Regularization | Data Augmentation in PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.org/github/Raghu-murugankutty/Deep-Learning-/blob/main/Image%20classification%20using%20ResNets%2C%20Regularization%20and%20Data%20Augmentation%20in%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains over 60,000 images belonging to 10 classes,Image classification using ResNets.
    - `Dataset`: The dataset contains over 60,000 images belonging to 10 classes. 
    - `Residual Block:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294435-d610a010-18ee-46ba-ab05-71d674b7f625.png">
    - `Convolution Block with ResNet9:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294492-bc60c300-2986-4262-bba5-b4796fa13712.png">
    - `One Cycle Learning Rate Policy:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294535-b3faebb9-45d1-4ebe-8528-2433ba5019a3.png">
    - `Built Feed Forward neural network(ANN)` and achievied an accurcy of 48%.
    - `Built Convolutional Neural Network(CNN)` and improved the accuracy till 75%.
    - <img width="230" alt="image" src="https://user-images.githubusercontent.com/41443395/208294842-4c51b6c7-449d-4b1c-b207-0f9e16bc370a.png">
    - <img width="268" alt="image" src="https://user-images.githubusercontent.com/41443395/208294918-d307e51a-d6e9-48c2-b3dd-6251c9e0a97f.png">
    - Applied technique like `Data normalization`, `Data augmentation`, `Batch normalization`, `Learning rate scheduling`, `Weight Decay`, `Gradient clipping`...etc
    - Using ResNet architecture, I achieved the `accuracy of 90.45%`.
    
<hr>

 - ## Transfer Learning for Image Classification | ResNets |  PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Transfer%20Learning%20for%20Image%20Classification%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains 37 category (breeds) pet dataset with roughly 200 images for each class, 
Performing image classification using tranfser learning models.
    - `Dataset`: We'll use the Oxford-IIIT Pets dataset from "https://course.fast.ai/datasets". It is 37 category (breeds) pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting.
    - `Sample Images from Oxford-IIIT Pets dataset:`
    - <img width="438" alt="image" src="https://user-images.githubusercontent.com/41443395/208370719-f082c2e4-f319-4884-a42f-2312b8798e3f.png">
    - `Using Pre-trained weights:`
    - <img width="530" alt="image" src="https://user-images.githubusercontent.com/41443395/208370341-e1731788-74d3-4f8d-a8af-084fc8f3a0f5.png">
    - `Parameter:`
    - <img width="405" alt="image" src="https://user-images.githubusercontent.com/41443395/208370924-f70659a6-2990-440c-a3f7-42c42f13e84b.png">
    - Using pre-trained transfer learning model weights, I achieved the `accuracy of 80.01%`.(with minimal epochs)
    
 <hr>

 - ## Image Classification using Convolutional Neural Networks (CNN) in PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Image_Classification_using_Convolutional_Neural_Networks_in_PyTorch%20v2.ipynb)):
    - `Problem Statement`: The dataset contains over 60,000 images belonging to 10 classes,Image Classification using Convolutional Neural Networks using PyTorch.
    - `Dataset`:  Dataset contains 2 folders train and test,The training set contains (50000 images) and test set (10000 images) respectively. The images belonging to 10 classes. 
    - `Sample Image Grid:`
    - <img width="491" alt="image" src="https://user-images.githubusercontent.com/41443395/208302244-cedfac65-f7c7-4d52-8f95-4f2d59ba10b5.png">
    - `Convolution Example:`
    - <img src="https://miro.medium.com/max/1070/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif" style="max-width:400px;">
    - `CNN Block:`
    - <img width="523" alt="image" src="https://user-images.githubusercontent.com/41443395/208302290-0172c6c0-0de9-433b-a856-541e04e45118.png">
    - `Result:`Using Convolutional Neural Networks(CNN), I achieved the accuracy of 78.19% 
    - `Sample prediction Results:`
    - <img width="215" alt="image" src="https://user-images.githubusercontent.com/41443395/208302380-3263f178-59d1-4168-b5cb-5ef4d5874fbe.png">
    - <img width="232" alt="image" src="https://user-images.githubusercontent.com/41443395/208302395-c5ccfdb3-72c1-43b7-8e1f-9a85f03eb879.png">

<hr>

 - ## Regularization of CNN models | Image classification model | PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Image_classification_using_CNN%2C_Technique_to_Regularize_Convolutional_Neural_Network.ipynb)):
    - `Problem Statement`: The dataset contains over 90483 images of fruits or vegitables, we are using CNN to build a classifcation model to predict the class of fruit from a set of 131 classes using PyTorch.
    - `Dataset`:  The daaset contains 90483 images of fruits or vegitables. 
    `Training set size`: 67692 images (one fruit or vegetable per image).
    `Test set size`: 22688 images (one fruit or vegetable per image).
    `Number of classes`: 131 (fruits and vegetables). Image size: 100x100 pixels.
    - `Images`:
    - <img width="503" alt="image" src="https://user-images.githubusercontent.com/41443395/209466991-420b6c77-d7dd-49c1-afbe-c6dc810237a8.png">

    - Model is built on PyTorch along with the implementation of techniques like `Data augmentation`, `Batch normalization`, `learning rate schedule`, `Weight Decay`, `Gradient clipping`, `adam optimizer`, `layer dropouts`, `Minmax pooling` to achieve the best results.
    - The model is trained and evaluated on GPU using PyTorch built-in `CUDA library`.
    - <img width="580" alt="image" src="https://user-images.githubusercontent.com/41443395/209466663-b5d2c208-edb4-41a8-862b-5b9dcf1a8828.png">
    - <img width="580" alt="image" src="https://user-images.githubusercontent.com/41443395/209466675-855b715e-1e41-43de-8419-175f30353f06.png">
    - `Learning rate with batch no`
    - <img width="260" alt="image" src="https://user-images.githubusercontent.com/41443395/209466754-5820384d-691b-4808-b918-e09acd0bf51d.png">
    - `Model achieved the validation accuracy of 99%`.

<hr>
    

# Machine Learning Projects

<hr>

 - ## Walmart store sales forecasting | Random Forest | XGBoost | Hyper parameter tuning ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/walmart-store-sales-forecasting-v4)):
    - Problem Statement: The dataset contains the historical sales data for 45 Walmart stores located in different regions. Each store contains a number of departments,We are forecasting department-wide sales for each store. In addition, Walmart runs several promotional markdown events throughout the year.These markdowns precede prominent holidays, the four largest of which are the Super Bowl, Labor Day, Thanksgiving, and Christmas. The weeks including these holidays are weighted five times higher in the evaluation than non-holiday weeks. In this problem, we are modelling the effects of markdowns on these holiday weeks in the absence of complete or ideal historical data.
    - `Dataset`: The dataset contains `420K+ observations and 16 features` (store, department, date, weekly_sales, isholiday, etc.) of the Walmart sales data.
    - pre-processed data from over 420k+ records with over 16 features to forecast Walmart sales data.
    - performed data cleansing, data imputation, encoding, scaling, and feature engineering to improve the model's performance and reduce the loss.
    - Libraries used: Pandas, Numpy, Sklearn, Matplotlib, and Seaborn. To fully comprehend the data, use Plotly to visualise it and its features.
    - Machine learning models: `Random Forest regressor`, `XGBoost regressor`.
    - The model achieved an accuracy of `Adjusted R^2 = 97.29%` after hyperparameter tuning.
    - This model was entered into the `Kaggle competition and placed in the top 15% of the leaderboard`.
    - Eg: Hyper-parameter tuning sample 
    - <img width="543" alt="image" src="https://user-images.githubusercontent.com/41443395/206430341-c3c15420-daae-4ea1-bb40-81096c2b64d4.png">
    - Eg: Results Sample
    - <img width="545" alt="image" src="https://user-images.githubusercontent.com/41443395/206429971-f5c51992-7f45-4c9c-8767-c3b23c7f4f0c.png">

<hr>

 - ## New York City Taxi Fare Prediction ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.org/github/Raghu-murugankutty/Machine-Learning-Projects-With-Python/blob/main/New%20York%20City%20Taxi%20Fare%20Prediction%20_%20Random%20forest%20_%20XGB.ipynb)):
    - `Problem Statement`: The dataset contains the record of taxi trips in New York City. Our objective is to create a model that will accurately estimate the fare amounts of trips in the test dataset. Accuracy is measured by the root-mean-square error.
    - `Dataset`: The dataset contains `5.5 million observations and 8-features` (fare_amount , pickup_longitude , pickup_latitude, dropoff_longitude    dropoff_latitude, passenger_count, etc.) of taxi trips in New York City.
    - pre-processed data from over 10% of the 5.5 million records with over 8-features to accurately estimate the fare amounts of taxi trips.
    - performed data cleansing, data imputation, encoding, scaling, and feature engineering to improve the model's performance and reduce the loss.
    - Libraries used: Pandas, Numpy, Sklearn, Matplotlib, and Seaborn. To fully comprehend the data, use Plotly to visualise it and its features.
    - Machine learning models: `Random Forest regressor`, `XGBoost regressor`.
    - A gradient boosting decision tree model with `RMSE obtained 3.23`. 
    - This model was entered into the `Kaggle competition and placed in the top 30% of the leaderboard`.
    - Eg: Please find below the snapshot of learning rate parameter tuning.
    - <img width="490" alt="image" src="https://user-images.githubusercontent.com/41443395/206457525-df7a07e1-27b3-4c03-a883-c20c3e5b9ca6.png">
    
<hr>

# Explainatory Data Analysis Using Python

<hr>

 - ## Zomato Resturant Analysis ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/zomato-dataset-eda-visualizations-v1)):
    - `Analyzed over 9000 restaurants with over 20 features`.
    - Performed data analysis using Python(pandas, numpy) and building visualizations using `Matplotlib, Seaborn and Plotly`
    - Identified best ‘Breakfast’, ‘Fast Food’ and ‘Ice Cream’ parlours in various localities.
    - `I used Plotly to build interactive visualizations`.
    - Derived multiple insights from the dataset such as top rated restuarents, most rated and preferable cuisines, best place to eat ice-creams, fast foods, Breakfast etc
    - Eg: Avg rating of ice cream shops:
    - <img width="683" alt="image" src="https://user-images.githubusercontent.com/41443395/206170315-7db8b500-e328-4d76-9fae-4a5a0263cab6.png">
    - Eg: List of counteries the survey is spread accross:
    - <img width="641" alt="image" src="https://user-images.githubusercontent.com/41443395/206171628-29b86452-ebcb-45d2-87e9-a4831e61d149.png">

<hr>
 
 - ## Amazon's Top 50 bestselling books from 2009 to 2019 ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/amazon-bestselling-books-eda-plotly-visualizations-v1)):
    - `Amazon book sales data from 2009 to 2019 was examined`. The dataset contains seven features. 
    - The features such as "User Rating," "User Reviews," "Price,"  Year, Genre, etc.
    - I use Python frameworks such as `Pandas and Numpy` to perform data exploration and understand the dataset deeply.
    - used python visualization libraries to understand the dataset visually, such as distribution of the dataset and its features (User Rating, Reviews, Price, Year, Genre, etc.).
    - Build interactive visuals using Plotly, used Seaborn and Matplotlib to create visualizations (`BarPlot, PieChart, Countplot, Scatter Plot, RegPlot`, and so on).
    - I performed analyses such as "best sellers," "fiction sellers vs. non-fiction sellers," "best authors,", "yearwise best sellers," etc..
    - I derived unique insights from the dataset and visualized them clearly.
    - Genre based best sellers over the yers:
    - <img width="509" alt="image" src="https://user-images.githubusercontent.com/41443395/206168921-1126cc92-e000-45bf-9ac3-07a4116bc658.png">
    - Avg Price of books over the years:
    - <img width="505" alt="image" src="https://user-images.githubusercontent.com/41443395/206169381-41add568-f05e-4d31-8955-56a20bece8cf.png">

<hr>

 - ## Employee attrition rate dataset ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/employee-attrition-rate-eda-visualization-model-creation-evaluation-v1)):
    - `This dataset contains employee details with 35 different features`. This data set was created by IBM data scientists.
    - Some of the features from the datasets are attrition, business travel, environmental satisfaction, job involvement, percentage salary increase, and other features.
    - I used Python frameworks such as Pandas and Numpy to perform data exploration and understand the dataset in depth.
    - used Python visualization libraries to understand the dataset visually, such as the distribution of the dataset and its features.
    - Build interactive visuals using Plotly. I used Seaborn and Matplotlib to create visualizations (BarPlot, PieChart, Countplot, Scatter Plot, RegPlot, and so on).
    - I performed detailed analysis and derived valuable insights from the dataset, such as What types of employees resign on a regular basis? What type of employee stays on the job the longest versus attrition?etc..
    - Eg: Correlation of features:
    - <img width="636" alt="image" src="https://user-images.githubusercontent.com/41443395/206177912-affdeeb5-51b8-4127-89f2-7c0ad3591a03.png">
    - Eg: Attrition rate v/s Salary hike
    - <img width="523" alt="image" src="https://user-images.githubusercontent.com/41443395/206178153-af30bdbb-b776-4960-b467-0fb95ece875b.png">
<hr>

 - ## Indian metro cities flat rental dataset ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/indian-metro-cities-flat-rental-dataset)):
    - `This dataset contains rental flat details across metropolises in India with 12 different features`.
    - The features such as area type, locality, furnishing status, tenant preference, floor, rent, and so on.
    - I use Python frameworks such as Pandas and Numpy to perform data exploration and understand the dataset in depth.
    - used Python visualization libraries to understand the dataset visually, such as the distribution of the dataset and its features.
    - Build interactive visuals using Plotly; use Seaborn and Matplotlib to create visualizations (BarPlot, PieChart, Countplot, Scatter Plot, and so on).
    - I performed detailed analysis and derived valuable insights from the dataset, such as Which city listed the most rental homes? What types of tenants are preferred? What types of homes are listed? and so on.
    - Please find below some sample visualizations, Eg: House rent v/s house size
    - <img width="677" alt="image" src="https://user-images.githubusercontent.com/41443395/206181737-7ba049ef-ea84-4e87-a9b3-954952aaa7ed.png">
    - Eg: Types of furnishing status
    - <img width="664" alt="image" src="https://user-images.githubusercontent.com/41443395/206181893-73e35dcf-bb52-4227-b53f-5abf80b8e1dd.png">

<hr>

 - ## Stack overflow developer survey dataset ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/stack-overflow-developer-survey-dataset-eda-plotly-matplotlib-visualizations)):
    - `This dataset contains nearly 65,000 responses were fielded from over 180 countries and dependent territories`.
    - The annual developer survey examines all aspects of the developer experience, from career satisfaction and job search to education and opinions on open-source software.
    - The features such as Work_year, experience_level, employment_type, job_title, salary_in_usd, remote_ratio, and so on.
    - I use Python frameworks such as Pandas and Numpy to perform data exploration and understand the dataset in depth.
    - used Python visualization libraries to understand the dataset visually, such as the distribution of the dataset and its features.
    - Build interactive visuals using Plotly; use Seaborn and Matplotlib to create visualizations (BarPlot, PieChart, Countplot, Scatter Plot, and so on).
    - I performed detailed analysis and derived the valuable insights from the dataset such as most demanding job titles, Highest avg pay, which job titles earns more avg anual salary, etc..
    - Please find below some visualisations
    - Experience v/s salary in USD:
    - <img width="666" alt="image" src="https://user-images.githubusercontent.com/41443395/206188132-3e7f0466-8421-4178-80ff-97a2f6fbbe64.png">
    - Salary Distribution v/s Experience level:
    - <img width="666" alt="image" src="https://user-images.githubusercontent.com/41443395/206188864-dd50106e-aa99-4c36-87be-91dc7aafbd9c.png">

<hr>

# Power BI Projects 

<hr>

#### 1) Sales Dashboard Link:- https://app.powerbi.com/groups/me/reports/2fc9a6cd-8698-436d-b70d-1b9411c13e03/ReportSection
#### 2) Sales Dashboard Link:-  https://app.powerbi.com/groups/me/reports/f2fc7094-0034-4ff3-9219-92532424b11f/ReportSection 
#### 3) Time series forecasting Link:- https://app.powerbi.com/groups/me/reports/4dcba48a-0228-4eef-8d52-72bc099df4ab/ReportSection87f7fed03b93553b0216


## Sales Dashboard screenshot
<img width="650" alt="Sales Dashboard screenshot" src="https://user-images.githubusercontent.com/41443395/170298963-c76d28fd-9904-43fe-85f9-6d1a0f551c43.png">

<img width="650" alt="image" src="https://user-images.githubusercontent.com/41443395/170796716-301ed1b6-0ee6-4578-b534-6d4299ab4d24.png">

## Time series data forecasting
<img width="650" alt="Time series screenshot" src="https://user-images.githubusercontent.com/41443395/170303785-f23d1e01-52b9-44eb-aff6-830cdafb5e30.png">
<img width="650" alt="image" src="https://user-images.githubusercontent.com/41443395/208118248-7a077bd9-1254-4d6d-b1e3-72e97d5a5c54.png">

## schema
<img width="650" alt="Schema" src="https://user-images.githubusercontent.com/41443395/170076026-ecb666ff-930d-4a6c-81a3-6dfc91fe0850.png">




















