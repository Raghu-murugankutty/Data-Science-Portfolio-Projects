# Data-Science-Portfolio-Projects
The repo contains details of all my data science projects. 



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



















