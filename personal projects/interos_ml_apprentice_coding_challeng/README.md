
Interos ML Natural Language Processing Coding Challenge 
 
This challenge was originally part of the requirement for the apprenticeship program of Interos. 
- Learn more about the challenge [here.](https://docs.google.com/document/d/1wv_B3VCGKpOS_-SCXOl0MjLWynwnGJGg3sFThl-nLCw/edit)
- Using the following [dataset](https://drive.google.com/drive/folders/1k7MNpw_huZTL9opXU9e7u6M8phupPih7), implement a model or algorithm that can classify news articles of (at least) the type "earn" covered in the dataset. 
 
As against the original Reuters21578 dataset, the dataset provided was purposely amended for this challenge to enable necessary analytical assessment.  

![image](https://user-images.githubusercontent.com/29716987/144979635-8357fab9-00fd-4f15-a7c4-8137962d415e.png)


This data, describing patient waits in a hospital, was built for the following challenges (see our recent publication here: https://rdcu.be/b4ffC):

- Predict patient wait times from the other operational features, as accurately as possible
- Identify the smallest subset of features, sufficient for accurate wait time prediction across all four facilities in the dataset. The smallest model MAE should be at most 1-2 minutes worse compared to the best full model MAE
- Invent (engineer) new features significantly improving model prediction quality
Overall, reducing MAE by more than 70% in comparison with the simple intercept model (predicting wait from its overall average) would be a significant step forward.

Data description
Our dataset represents operational features, captured in four different hospital facilities, processing walk-in (F4) or scheduled (F1, F2, F3) patients. Approximately 600 to 1000 days of full  patient flow data were extracted from each facility. The target variable to be predicted is Wait. The other variables represent different features of the patient flow, captured at the time of patient arrivals and departure (one line per each patient visit event). Data Excel file contains the exact definitions of all features used (see Contents sheet).

This data was extracted from the real hospital information system. Therefore it was anonymized and aggregated to remove any confidential information. In particular, x_ArrivalDTTM, x_ScheduledDTTM, and x_BeginDTTM timestamps were anonymized, and set to the fictitious future dates (to make you aware of this modification). Only their relative timing was preserved. As a result, these timestamps can be used only for sorting, to reflect the correct order of feature events, but nothing else.

Code
We include our Matlab code used to process this data here – feel free to download and experiment.
 
 
The Position 
Learn more about the the Interos ML Apprenticeship here
 
The Challenge 
Using the following dataset, please implement a model or algorithm that can classify news articles of (at least) the type "earn" covered in the dataset. Coding Challenge Data 
 
The Rules 
No time limit is enforced for this project. As such, you are welcome to spend as little or much time on it as you wish, though we want to be respectful of your time spent. Please let us know however much time you take on the task! 
 
Use whatever libraries/frameworks/tools you wish, though Python is the required language. Develop your work however you wish (including on a local machine), but the end result should include all relevant files hosted on Google Drive, and a Google CoLab notebook that can run your code, load your model*, and allow for testing with additional data. After November 22, you are welcome to publicly make available your work here on hosting sites like GitHub for your own portfolio, but posting anything publicly before that time will result in automatic disqualification from the hiring process. 
External libraries, tutorials, codebases, etc. are fine to use, but provide your sources, including tutorials, boilerplate code from frameworks, links to StackOverflow, and so on. Unattributed external code will be considered automatic disqualification from the hiring process. 
 
As part of the evaluation process, data formatted similarly to the data you will use for this project will be run against your model. Please provide a method of running a file like the others provided against your model, which will score it. 
 
How To Submit Your Challenge (before 11/22/21) 
In order to apply for the apprenticeship, you must submit links to a completed Google CoLab notebook, a Google Drive folder with any relevant data, and a Google Slides presentation. Please submit those as part of your application process. Completed projects’ data/slides/Colab notebooks/etc. should be made available to view by anyone with the link. You must include links to the completed project as part of your application. If you encounter any issues with permissions, please email a link (and allow access) to dbishop@interos.ai.
 
Required as part of your Google CoLab: 
A notebook that can be run from top to bottom* (see below for a note on CoLab’s limitations) which will load a pre-saved model (or other approach to the problem) and allow predictions with it for new data, including a similarly-formatted file to those used in training, as well as raw text 
Comments (or Markdown, or other) explaining any functions, your process, etc. 
Comments attributing code adapted from external sources 
 
Required as part of your Slides presentation 
Your Exploratory Data Analysis (EDA) process – How did you check the data for this project? How are your data split for training versus testing, or other ways to segment it? 
Your modeling process – how did you make something that can automatically categorize news articles? Do you predict a single label from a given input, or multiple? Binary decisions, or scored from 0-1, or even something else entirely? Why? 
How you review your results – given your model, how do you then make new predictions with it, and how do you check it for accuracy beyond simple scoring metrics?  
Additional data – given a working model, how would you incorporate more data, or make changes to the training data you already have? 
A bit about you, 
Any other necessary technical details of your project, 
Anything else of note. 
 
* Due to RAM/GPU limitations, your code, if developed locally, may not be able to run on Google CoLab. If this is the case, please provide detailed documentation as to its performance and how to locally reproduce the results (such as a requirements.txt, readme, pipenv/poetry lock file, etc.) 
