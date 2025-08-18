# Diaspora Donor Recommender System 🤝

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diasporadonorapp-6y5uqwvfczq5gsrwhahxzy.streamlit.app/)

👉 [Live App: Diaspora Donor Recommender](https://diasporadonorapp-6y5uqwvfczq5gsrwhahxzy.streamlit.app/)

---

## 📖 Overview  

The **Diaspora Donor Recommender System** is an AI-driven platform designed to improve donor–project matching within the humanitarian and development sector. By leveraging hybrid recommender algorithms, the system personalises project suggestions for diaspora donors, considering their **preferences, budget caps, behavior types, and historical interactions**.  

The project contributes academically and practically by combining **rule-based**, **content-based (cosine similarity)**, and **collaborative filtering (CF)** approaches, further enhanced by **ethical AI mechanisms** to reduce overexposure of highly popular projects.  

---

## 🎯 Objectives  

- To design and deploy a **recommender system** that bridges diaspora donors with projects aligned to their interests.  
- To integrate **donor behavior, preferences, and historical interactions** into recommendation pipelines.  
- To evaluate recommender performance using **Precision@K, (Recall@K and MAP@K- included only in the App), Diversity, Novelty, Coverage, and Error metrics (MAE, MSE, RMSE)**.  
- To ensure **fairness and ethical AI** by down-weighting over-exposed projects.  
- To deploy a **live, interactive application** via Streamlit Cloud.  

---

## 📂 Project Structure  

```bash
├── app.py                        # Main Streamlit application
├── artifacts/                    # Datasets and precomputed artifacts
│   ├── donors.csv / donors_5000.csv
│   ├── projects.csv / projects_2000.csv
│   ├── interactions.csv
│   ├── ratings_5000x2000.csv
│   ├── cf_estimates.csv.gz
│   └── proj_vectors.parquet      # Precomputed project embeddings 
├── colab_notebooks/              # Research and training notebooks
│   ├── 01-hybrid-recommender-full-pipeline.ipynb
├── README.md                     # Documentation
├── requirements.txt              # Python dependencies
└── .streamlit/                   # Streamlit configuration
📊 Datasets

The recommender relies on both synthetic and precomputed datasets stored under artifacts/:

Donors dataset (donors.csv / donors_5000.csv)
Includes donor IDs, names, emails, preferences (regions, sectors), behavior types, and budget caps.

Projects dataset (projects.csv / projects_2000.csv)
Contains project IDs, titles, regions, sector focus, organisation types, funding targets, and popularity scores.

Interactions dataset (interactions.csv and ratings_5000x2000.csv)
Represents donor–project interactions and implicit ratings used for collaborative filtering.

Precomputed CF estimates (cf_estimates.csv.gz)
Matrix of predicted donor–project ratings, generated using Surprise SVD (trained also on the MovieLens 100K dataset for robustness and cross-validation).

⚙️ Methods & Algorithms

The system integrates multiple recommender paradigms:

Rule-Based Filtering

Matches donors to projects based on explicit preferences (regions, sectors, budget caps).

Content-Based Filtering (Cosine Similarity)

Projects represented as embeddings (region + sector one-hot vectors).

Donor preferences encoded into a vector for similarity comparison.

Collaborative Filtering (SVD)

Learns latent donor–project interactions.

Precomputed predictions stored in cf_estimates.csv.gz.

Hybrid Recommender

Weighted blend of Rule, Content, and CF scores.

Incorporates an Ethical AI adjustment to down-weight the most popular projects.

Evaluation Metrics

Precision@K, Recall@K, MAP@K – effectiveness of recommendations.

Diversity & Novelty – coverage across projects and exposure to less-known projects.

Coverage@K – percentage of recommended projects already interacted with.

Error metrics (MAE, MSE, RMSE) – accuracy of predictions vs. ground truth.

📈 Results

Hybrid recommender outperforms single algorithms, balancing accuracy and fairness.

Ethical AI switch successfully reduces overexposure of popular projects, improving novelty.

Interactive donor view enables custom weights tuning, showing trade-offs between Rule, Content, and CF.

Metrics evaluation framework highlights strengths and weaknesses across donor profiles.

🌍 Deployment

The application is deployed live via Streamlit Cloud:

👉 Diaspora Donor Recommender App

Features include:

Two-pane Home: donor profile & preferences (left), recommendations (right).

✅ Tick marks for donors with historical interactions.

Donor progress tracking with visual charts (sectors, regions, funding distribution).

Metrics tab for evaluation results.

Explore tab to browse and filter projects manually.

Compare tab for algorithm-level comparisons.

Donor registration form (session-only).

🚀 Installation & Usage
1. Clone the Repository
git clone https://github.com/ANkwocha16/diaspora_donor_app.git
cd diaspora_donor_app

2. Install Dependencies
pip install -r requirements.txt

3. Run Streamlit App Locally
streamlit run app.py


The app will open in your browser at http://localhost:8501/.

📒 Notebooks

The full research and model development pipeline is documented in colab_notebooks/:

00-data-prep.ipynb — data cleaning, preprocessing, normalization.

01-rule-based.ipynb — preference-driven filtering.

02-content-based.ipynb — cosine similarity on donor–project vectors.

03-collaborative-filtering.ipynb — SVD training with MovieLens 100K and donor–project datasets.

04-hybrid-recommender-full-pipeline.ipynb — integration of all approaches, evaluation metrics, and export of CF estimates.

📜 Academic Contribution

This project demonstrates:

Application of hybrid recommender systems in the humanitarian sector.

Novel integration of ethical AI adjustments in recommendations.

Evaluation using both standard recommender metrics and donor-centric progress indicators.

Practical deployment of a live, reproducible recommender system.

🙌 Acknowledgements

University, lecturers, especially my supervisor; Amin Noorozi and fellow students for academic guidance and making learning fun.
And to myself, for the perseverance, research, and sleepless nights that made this work possible.

📌 License

This project is released under the MIT License.
