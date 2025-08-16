Diaspora Donor Recommender System

This repository contains an academic prototype of a donor recommender system implemented in Streamlit. The system matches diaspora donors with community projects using three algorithmic approaches: collaborative filtering, content-based similarity, and rule-based heuristics. It is designed to demonstrate recommender methodologies in a donor–project matching context.

Features

Hybrid recommendation framework

Collaborative filtering with Singular Value Decomposition (SVD).

Content-based filtering using cosine similarity.

Rule-based recommendation logic.

Synthetic data integration

Includes donor, project, and interaction data generated for experimentation.

No real-world or sensitive information is used.

Evaluation and metrics

Supports error-based metrics (MAE, RMSE, MSE).

Provides coverage analysis.

Displays recommendation diagnostics.

Interactive user interface

Donor profile registration and progress visualization.

Multi-tab layout for recommendations, metrics, project exploration, and diagnostics.

Interactive charts and tables to support exploration of recommendations.

Repository Structure
├── app.py               # Main application code (Streamlit)  
├── requirements.txt     # Python dependencies  
├── README.md            # Documentation  
└── artifacts/           # Synthetic data and pre-trained models  
    ├── donors_5000.csv  
    ├── projects_2000.csv  
    ├── synthetic_interactions_5000x2000.csv  
    ├── svd_model.joblib  
    └── sim_matrix.npy  

Installation

Clone the repository:

git clone https://github.com/<username>/diaspora_donor_recommender.git
cd diaspora_donor_recommender


Install dependencies:

pip install -r requirements.txt


Run the application:

streamlit run app.py

Data

The artifacts/ directory contains the synthetic datasets required for execution:

donors_5000.csv – synthetic donor profiles.

projects_2000.csv – synthetic project metadata.

synthetic_interactions_5000x2000.csv – donor–project interaction matrix.

svd_model.joblib – pre-trained collaborative filtering model.

sim_matrix.npy – similarity matrix for content-based filtering.

These datasets are synthetic and intended solely for experimentation and reproducibility.

Usage

Launch the application.

Register a donor profile or select an existing synthetic donor.

Request recommendations by selecting algorithm weights.

Navigate across tabs:

Recommendations – donor-specific project suggestions.

Metrics – system-level evaluation metrics.

Explore – filtering and browsing of all projects.

Diagnostics – insights into overlaps, coverage, and reasoning for recommendations.

Deployment

Deployment can be achieved through Streamlit Community Cloud:

Push this repository to GitHub.

Log in at Streamlit Cloud.

Deploy with:

Repository: <username>/diaspora_donor_recommender

Branch: main

File: app.py

A permanent public URL will be generated after deployment.

Limitations

Evaluation metrics may produce limited results due to the synthetic nature of the data.

Precision and recall values depend on donor interaction history and may be zero where no history exists.

Synthetic projects lack full descriptive metadata compared to real-world data.

Future Work

Integration with richer donor and project datasets.

Enhanced evaluation through alternative metrics and larger interaction histories.

Improved visualization of recommendation rationales.

Acknowledgments

The system is implemented using the following libraries and frameworks:

Streamlit for web application development.

Scikit-learn for preprocessing and evaluation methods.

Surprise library for collaborative filtering.

Pandas and NumPy for data handling.

Matplotlib for visualizations.