# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Capstone: ML, Brain Tumors & MRI
Constance Lin | DSI-41
<br>

## <b>1. Introduction</b>
Brain tumors are growth of cells in the brain or near the brain. Many different types of brain tumors exist [1, 2] and the more common ones are Gliomas, Meningiomas, Pituitary tumors and Nerve tumors. In such circumstanes, using Magnetic Resonance Imaging (MRI) is usually the preferred way to diagnose brain tumors [3].

While MRI machines are only available in hospitals in Singapore, they are considered relatively accessible. The ease of access to such imaging facilities have resulted in a huge increase of orders for such diagnostic testing, which can drive healthcare costs up [8,9]. Consequently, it further exacerbates the issue of manpower shortage of specialised healthcare workers to perform these scans[5], diagnose the images [4,8] and even care for the patients [5,7]. Furthermore MRI scans are generally expensive; a typical brain scan can start from a few hundred dollars. In Singapore, citizens are given only $300 per year in our MediSave to use on these scans [6], which can be costly and this excludes other aspects of healthcare related costs that they may be subjected to [10]. 

With regards to these issues, there is a need (at a high-level) to consider ways to maintain overall healthcare costs in the country, circumvent the industry's manpower issues as well as reducing patient's healthcare costs for diagnosis purposes.

This then brings about the possibility of making use of trained machine-learning models in particular, to act as a complimentary tools in diagnosing specific conditions i.e. brain tumors, given a set of images i.e. MRI images. 

<br>

**Sources:**
* [1] [Brain Tumors](https://www.mayoclinic.org/diseases-conditions/brain-tumor/symptoms-causes/syc-20350084)
* [2] [The Most Common Brain Tumor: 5 Things You Should Know](https://www.hopkinsmedicine.org/health/wellness-and-prevention/the-most-common-brain-tumor-5-things-you-should-know)
* [3] [Brain Tumor: Diagnosis](https://www.cancer.net/cancer-types/brain-tumor/diagnosis#:~:text=MRIs%20create%20more%20detailed%20pictures,are%20different%20types%20of%20MRI)
* [4] [The Growing Problem of Radiologist Shortages: Perspectives From Singapore](https://pubmed.ncbi.nlm.nih.gov/38016677/)
* [5] [Singapore hospitals struggle to meet demand for radiographers, pharmacists amid shortage in healthcare workers](https://www.channelnewsasia.com/singapore/allied-health-professionals-healthcare-nurses-hospitals-shortage-radiographers-pharmacists-3861226)
* [6] [Subsidies for MRI scans](https://www.moh.gov.sg/news-highlights/details/subsidies-for-mri-scans)
* [7] [AI in Southeast Asia: Are jobs being replaced? Not quite yet, but an uncertain future beckons](https://www.channelnewsasia.com/asia/artificial-intelligence-ai-replace-steal-jobs-work-healthcare-doctor-nurse-call-centre-4082086)
* [8] [Transforming radiology to support population health](https://annals.edu.sg/transforming-radiology-to-support-population-health/)
* [9] [Singapore's healthcare spending is likely to continue rising; targeted support needed: Gan](https://www.businesstimes.com.sg/international/singapore-budget-2021/singapores-healthcare-spending-likely-continue-rising-targeted)
* [10] [7 Healthcare Cost Statistics in Singapore (2023)](https://smartwealth.sg/healthcare-cost-statistics-singapore/)
 
<br>

## <b>2. Problem Statement</b>
Are pre-trained ML models accurate enough to identify different brain tumors from MRI images to aid in diagnosis?

### <b>Goal of capstone:</b>
For this capstone, our goal is two-fold:
1. Using pretrained CNN models to accurately identify the tumor shown on the Magnetic Resonance (MR) images. This is a multi-class classification problem.  
2. Create a Question-Answer(QA) model that is able to answer specific questions pertaining to MRI safety and implants based on the documents it has been given only.

### <b>Potential Stakeholders:</b>
1. Healthcare workers
    * radiologists can leverage on technologies to help them with reporting and cope with the workload
    * clinicians can send their patients for less scans and still get good diagnosis to plan treatment

2. Patients
    * can spend less money on scans to get diagnosis
    * can get their treatments quicker

<br>

## <b>3. Datasets</b>
The following data sets are in data folder:
1. Raw data
    * archive.zip (obtained from [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset))
    * contains `Training` and `Testing` folders
        * each folder contains 4 classes: `glioma`, `meningioma`, `notumor`, `pituitary`
    * 5712 images in `Training` folder
    * 1311 images in `Testing` folder
    * all images in .jpg format

2. Training dataset
    * balanced out the proportion of images per class
        * 1321 images per class after balancing
    * 5284 images in total used for training

3. Validation dataset
    * originally from `Testing` folder - images split into half 
    * 655 images

4. Training dataset 
    * originally from `Testing` folder - images split into half
    * 656 images

### Data dictionary:
* filepaths: filepath of the tumor in the respective folders along with an image number
    * i.e. /content/drive/My Drive/Capstone/Images/Testing/glioma/Te-glTr_0000.jpg
* labels: type of tumor
    * i.e. glioma

<br>

## <b>4. Notebooks</b>
- Notebook 1: Using CNN pre-trained models on MRI Brain Tumor dataset
- Notebook 2: Retrieval-Augmented Question Answer (QA) model on PDF documents

### Notebook 1: Using CNN pre-trained models on MRI Brain Tumor dataset

The structure of Notebook 1 is as follows:

1. Image extraction from Google drive
    * Mounting Google drive to Colab for image extraction
    * Import in data
    * Splitting testing images to Validation dataset and Testing dataset

2. EDA on data
    * Class balancing for Training data
    * Visualization for Validation data

3. Image preprocessing
    * Image augmentation & visualization for Training data
    * Image augmentation & visualization for Validation data

4. Modeling & Metric tracking
    * VGG-16 & ResNet50
        * Customising the model
        * Training the model: Version 1, Version 2, Version 3
        * Plotting the accuracy and loss graphs
        * Model evaluation on Testing data
    * EfficientNetB2
        * Reprocessing the image input shape
        * Customising the model
        * Training the model: Version 1, Version 2, Version 3
        * Plotting the accuracy and loss graphs
        * Model evaluation on Testing data

5. Summary

### Notebook 2: Retrieval-Augmented Question Answer (QA) model on PDF documents

The structure of Notebook 2 is as follows:

1. Installations
    * Pinecone
    * Langchain
    * pyPDF
2. Document Loading & Splitting
3. Vector Store & Embeddings
4. Retrieval with QA model
5. Convert code to a function for Streamlit

## <b>5. Streamlit</b>
A .py file is created for demonstration of the QA model on streamlit. 

The following files are available in the `streamlit_demo` folder:

- `streamlit_demo_mvp.py`: 

This script makes use of the chatgpt 3.5 model with a prompt. It takes in a user's question, refer to the pinecone database to look for answers based on cosine similarities and return a concise reply that best answers the question.

### Running the demo
1. Use a terminal such as Anaconda prompt depending on the installation path of Python, change the directory to point the the `streamlit_demo` folder
2. Run the line on the terminal `streamlit run streamlit_demo_mvp.py`
3. When the streamlit demo is successfully running, the terminal will display the Local URL, which shows the localhost port used on the machine, and the Network URL

To allow other machines to connect to the Network URL, go to your machine's Firewall settings and allow `python.exe` to access networks. The other machines should be connected to the same network as the machine you are using to host the streamlit demo.

<br>

## <b>6. Summary of findings </b>
For each pre-trained model, 3 versions were attempted:
* Version 1 with Adam optimizer + following original architecture's top layer with output modified to 4 classes
* Version 2: Version 1 with early stopping incorporated 
* Version 3: Version 2 but with RMSProp optimiser instead

When considering overall accuracy scores, VGG16 performed the best at 96%. However, ResNet50 model types have the lowest loss out of the 3.

To evaluate if the model performs well on identifying a particular tumor, we take into consideration high F-1 score in addition to considering "balanced" precision and recall scores so that the F-1 score will be as "true" as possible.

Furthere exploration of the metrics of each tumor type revealed that:
* ResNet50 with RMSprop performed the best in 2 tumors out of 4 - namely in meningiomas and gliomas. Even though they do not achieve the highest F-1 scores when compared to VGG16 model, it has a "truer" F1-score.

* On images with no tumors, VGG16 with Adam performed the best at 99% (F-1 score).

* For pituitary tumors, ResNet50 with Adam performed the best at 97% (F-1 score) with the highest score.

As such, with all these considerations, ResNet50 with RMSprop (92.2% accuracy),  is considered to be the best overall model for predicting different brain tumors with this particular dataset. 

<br>

## <b>7. Limitations</b>
Wnile doing the capstone project, these are the few limitations and/or potential points of contemplation that I have encountered.

1. Assumption that the images in the dataset are representative of the Singapore population

2. Literature's lack of definition for “accepted accuracy” and "accepted loss" in medical imaging 
    * the quantification of "accepted accuracy" and "accepted loss"
    * ML model training requires a quantifiable value to fine-tune

3. Precision-Recall debate
    * no clear definition which metric is prioritized in the trade-off in the healthcare industry
    * do the metric importance change depending on situations such as:
        * medical specialties
        * treatment targeted at specific conditions to treat
        * in the presence of comorbidities of the patient

4. RAG QA model only trained on a small corpus of PDF data
    * generally works but not 100% accurate all the time
    * basis of QA model is using chatgpt
        * there is a possibility it can give answers that are not exactly from the trained PDFs despite prompt incorporated
 
<br>

## <b>8. Key Insights and Recommendations</b>
### Insights & further discussions:
1. Our customised "basic" models are able to predict testing data with an accuracy of at least 90% despite requiring 4 different outputs
    * potentially able to be implemented to be used in hospitals
    * more crucial part lies in having a framework to be able to quantify a accuracy-loss level that is applicable for different use cases
        * potential discussion point if ML is to be used as a complimentary tool
        * possibility of institution-dependent levels of tolerances as "accepted accuracy" i.e. public hospital vs specialised institutes

2. The potential of using a RAG QA model to answer proprietary related questions is demonstrated 
    * comparison of our QA model vs ChatGPT's answers to the same question showed that ours is more specific and could answer questions that ChatGPT could not.
    * more fine tuning is required - more data required
        * yet to be determined if having more data will mean that the phrasing of the questions can afford to be less precise but still get yield acceptable responses
        
 
### Recommendations:

1. Tailored model training
    * collect diagnostic images of our local population:
        * more accurate reflection of the types of tumors we deal with in local context
    * attempt voting classifier method or ensemble method on these current models to see if better accuracy-loss and f1 scores are produced
    * using CT images to train the model instead of MRI images or a combination of both
    * adaptation to different institutions’ usage requirements
        * For example, for junior radiologists to use in the A&E department during night shifts
            * they would benefit from the help as they have to report many different studies i.e. X-rays and CTs and there may or may not be a senior around at all hours

2. Tailored QA model 
    * obtain the paid PDF textbook version on MRI safety, bioeffects and patient management - from a website which most radiographers commonly use to check for MR compatible devices
       * determine if feeding this PDF into the model will achieve more precise answers
        * determine suitability of general patient usage or more ideal for healthcare professionals usage (if precision in question phrasing is required)
    * If QA model is better for the latter, there is potential to incorporate different companies’ implants information sheets to the model 
        * a common platform across public healthcare institutions (PHIs) where healthcare professionals can make use of it to check implant compatibilities for their patients before ordering the scans
        * most of time they are publicly available but all over the internet/ radiographers have to email to the companies for it/comes with the product in the packaging
    
As with all recommendations, for them to be implemented, the practical side of things have to be considered. One of them will definately be cost, and secondly, infrastructural requirements such as integration into our current hospital systems will have to be considered as well. 