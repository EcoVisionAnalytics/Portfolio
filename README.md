# Data Science & Machine Learning Portfolio

This repository showcases a diverse range of projects where I apply data science, machine learning, and cloud/web deployment techniques. Each project highlights a different aspect of my skill set—from data analysis and visualization in Python and R to full-stack web applications using Django/React.

---

## Table of Contents

- [Bird Ecological Health Analysis](#bird-ecological-health-analysis)
  - [Overview](#overview)
  - [Projects & Code](#projects--code)
  - [Visualizations](#visualizations)
- [Species Vocalization Identification](#species-vocalization-identification)
  - [Overview](#overview-1)
  - [Projects & Code](#projects--code-1)
  - [Field Deployable .bat Executions](#field-deployable-bat-executions)
- [Web & Cloud Deployment](#web--cloud-deployment)
- [How to Run](#how-to-run)
- [Contact](#contact)

---

## Bird Ecological Health Analysis

### Overview

This project focuses on analyzing bird population data in the Greater Yellowstone Ecosystem (GYE). It combines:
- **Python** scripts for data ingestion, exploratory data analysis, and high-quality visualization.
- **R** scripts for advanced statistical modeling (GAMYE models) and interactive dashboards with Shiny.
- **Web** projects using Django and React to create interactive dashboards and APIs.

### Projects & Code

- **Python Modules:**  
  - **Species_plots.py:** Generates species-specific abundance plots with dynamic configuration using JSON.
- **R Modules:**  
  - **GAMYE_Cluster_Modified.r:** Implements advanced species trend modeling with GAMYE and parallel processing.
  - **app.R:** An interactive RShiny dashboard for exploring bird trends over time.
- **Web Projects:**  
  - **Django/React:** (See folder `web/django_portfolio` and `web/react_portfolio`) for cloud-based interactive visualizations and data access.

### Visualizations

![Species Abundance Plot](images/species_abundance_placeholder.png)
![Species Abundance Plot](images/species_abundance_placeholder.png) 
![Species Abundance Plot](images/species_abundance_placeholder.png) 
![Species Abundance Plot](images/species_abundance_placeholder.png) 
![Species Abundance Plot](images/species_abundance_placeholder.png) 
![Species Abundance Plot](images/species_abundance_placeholder.png)   
*Example: Species abundance plot generated using Python modules.*

---

## Species Vocalization Identification

### Overview

This project is designed to identify vocalizations of five species from long-form audio recordings. It is built for ease-of-use, computational efficiency, and field deployment. The pipeline involves:
- **Data Augmentation:** for increasing training sample diversity.
- **Deep Learning Models:** to classify vocalizations.
- **Audio Processing & Detection:** that processes audio chunks to detect and extract species calls.
- **Batch Execution:** Managed via a `.bat` file for simple field deployment.

### Projects & Code

- **model_final.py:** Contains the neural network architecture, training routines, and model evaluation with cross-validation.
- **data_augmentation.py:** Implements several audio augmentation techniques (time stretch, pitch shift, noise addition) to bolster training data.
- **detect_species.py:** Processes long audio recordings in chunks, predicts species calls using a TensorFlow Lite model, and saves the detected calls.
- **run_detection.bat:** A Windows batch script to execute the entire detection pipeline with a single command.

### Field Deployable .bat Executions

This project is optimized for field deployment. The batch script (`run_detection.bat`) allows for easy execution on laptops or portable devices in remote settings. Simply double-click the `.bat` file to start the detection process on new audio recordings.

---

## Web & Cloud Deployment

I also showcase full-stack development skills with:
- **Django:** A backend API to serve processed data and model predictions.
- **React:** A modern front-end application that communicates with the Django API to provide interactive visualizations.
- **Cloud Deployment:** Docker and Kubernetes configurations are included in the `deployment/` folder for scalable cloud-based deployments.

---
