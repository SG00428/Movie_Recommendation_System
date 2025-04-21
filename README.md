# Movie Recommendation System

A Netflix-style personalized movie recommendation system that blends multiple machine learning models with metadata enrichment and an interactive web interface.

---

## Overview

This project recommends movies to users using a **hybrid model** that combines:

- **Matrix Factorization (SVD)**
- **Neural Collaborative Filtering (NCF)**
- Attention-based explainability
- Context Aware Recommendations

It enriches recommendations with metadata such as genres, languages, IMDb scores, and more using **TMDB API**.

The app offers:
- Personalized recommendations for existing users
- Cold-start support for new users
- Similar movie search
- Metadata display for transparency
- User feedback and liked movies tab
- Dynamic recommendations as per user feedback

---
## How to Run

### 1. Clone this repository

`git clone https://github.com/SG00428/Movie_Recommendation_System.git`

`cd Movie_Recommendation_System`

### 2. To run the Streamlit app

`streamlit run app/app.py`

Then open the local server link in your browser.

---

