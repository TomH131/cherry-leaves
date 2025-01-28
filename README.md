## Dataset Content

 - The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
 - The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

 - 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
 - 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

1. We hypothesise that the presence of visible white powdery marks on fungal-infected cherry leaves will enable a machine learning model to reliably distinguish between healthy and infected leaves.

2. We hypothesise that we can reach the minimum accuracy of 97% set by the client. Although this is a high threshold, we believe that the model can meet or exceed this target using the dataset of four thousand images provided by the client.

3. Currently, farm workers spend approximately 30 minutes per tree collecting samples to check for infection. We believe that a machine learning model can drastically reduce this time by instantly identifying whether a leaf is healthy or infected. Additionally, the option to upload and analyse multiple images simultaneously will further streamline the process.

 - Validation of these hypotheses will involve evaluating the model using graphical analysis and subsequent testing. This process will include assessing the model's accuracy and loss over epochs.

 - After validation, the client should be able to use images of future crops in conjunction with the model to accurately identify healthy leaves and those with powdery mildew. This will significantly reduce the time required for manual inspection and allow for targeted treatment of infected leaves.

## The rationale to map the business requirements to the Data Visualisations and ML tasks 

1. Business Requirement 1: Data Visualization
 - As a client I want to display the "mean" and "standard deviation" images for cherry leaves that are healthy and cherry leaves that contain powdery mildew, so that I can visually differentiate cherry leaves.
 - As a client I want to display the difference between an average cherry leaf that is healthy and a cherry leaf that contains powdery mildew, so that I can visually differentiate cherry leaves.
 - As a client I want to display an image montage for cherry leaves that are healthy and cherry leaves that contain powdery mildew, so that I can  visually differentiate cherry leaves.

2. Business Requirement 2: Classification
 - As a client I want to predict if a given cherry leaf is healthy or contains powdery mildew.
 - As a client I want to upload multiple images and download the prediction results to action targeted treatments.

## ML Business Case

We aim to develop a machine learning model capable of predicting whether a given cherry leaf is healthy or affected by powdery mildew, using a dataset provided by the client. The primary objective is to reduce the time employees spend identifying fungal infections in cherry trees. The success criterion for the model is achieving an accuracy of 97% or higher. The model will function as a binary classifier, providing predictions based on uploaded images.

### Current Process and Challenges
The current process involves manually inspecting each tree, which takes approximately 30 minutes per tree. Employees collect samples to visually verify whether the leaves are healthy or infected with powdery mildew. If a tree is found to be infected, a specific compound is applied to eliminate the fungus, a task that typically takes about a minute per tree. Given that the company manages thousands of trees across multiple farms nationwide, this process consumes thousands of hours and demands a more efficient solution.

### Dataset and Privacy Measures
The dataset provided contains over 4,000 images of leaves from the client’s crops, with an even distribution between healthy and infected leaves. This dataset has been preprocessed and divided into training, validation, and test sets for model development. The data was shared under a non-disclosure agreement (NDA) and is subject to strict privacy and protection measures to ensure compliance with confidentiality requirements.

## Dashboard Design

### Page 1: Project Summary
 - This page will provide a summary of the project, the dataset and also the business requirements.

### Page 2: Study Findings
 - This page will present our findings from the study, including visual representations of the differences between healthy cherry leaves and those affected by powdery mildew. The different between the average and variability images will be shown as well as the difference between the average healthy and average infected leaf. There will also be an image montage for both sets of leaves.

### Page 3: Mildew Detector
 - This page will include a link to download a set of cherry leaf images for live predictions.
 - It will feature a widget that allows users to upload images, with support for multiple uploads at once. The uploaded images will be displayed alongside a prediction statement indicating whether each leaf is healthy or infected, along with the associated probability score.
 - A table will also be included to list each image and its prediction results, along with a button to download the table and its data.

### Page 4: Project Hypothesis
 - This page will outline the project hypotheses and detail how they were validated during the course of the project.

### Page 5: Model Performance
 - This page will display the performance metrics of our model, including accuracy, loss over epochs, and other relevant evaluation results.

## Unfixed Bugs

- There are no unfixed bugs.

## Deployment

### Heroku

- The App live link is: `https://mildew-detector-th-0b3d38f46204.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content