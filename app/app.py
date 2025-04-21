import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import streamlit as st
import pandas as pd
import json
import os.path
import random
from utils.helper import recommend_top_n, find_similar_movies, get_movie_metadata, get_attention_score, get_new_user_recommendations

# At the top of the file, after imports, add:
def initialize_session_state():
    if 'all_feedback' not in st.session_state:
        st.session_state.all_feedback = load_feedback()
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    # Add new state for temp user ID persistence
    if 'persistent_temp_user_id' not in st.session_state:
        st.session_state.persistent_temp_user_id = None

# Load enriched movie metadata
movie_df = pd.read_csv("data/Netflix_Movie_Enriched.csv")
# Define project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Feedback storage file
FEEDBACK_FILE = os.path.join(project_root, "data/user_feedback.json")

# Load existing feedback if available
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

# Save feedback to file
def save_feedback(feedback_data):
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_data, f)
    # Update session state to maintain consistency
    st.session_state.all_feedback = feedback_data

# Get similar movies based on a user's liked movies
def get_recommendations_from_liked_movies(user_id, count=10, genre_filter="All"):
    user_id_str = str(user_id)
    user_feedback = st.session_state.all_feedback.get(user_id_str, {})
    
    # Get all movies the user has liked
    liked_movies = []
    for movie_id_str, feedback_type in user_feedback.items():
        if feedback_type == "like":
            liked_movies.append(int(movie_id_str))
    
    if not liked_movies:
        return []  # Return empty list if no liked movies
    
    # Get similar movies for each liked movie
    similar_movies = []
    for movie_id in liked_movies:
        movie_name = ""
        meta = get_movie_metadata(movie_id)
        if meta:
            movie_name = meta.get('Name', '')
        
        if movie_name:
            similar = find_similar_movies(movie_name, top_n=count)
            similar_movies.extend(similar)
    
    # Deduplicate and remove already liked movies
    seen = set(liked_movies)
    unique_similar_movies = []
    
    for movie_id, score in similar_movies:
        if movie_id not in seen:
            seen.add(movie_id)
            # Add a dummy attention score for consistency
            unique_similar_movies.append((movie_id, score, 0.5))
    
    # Apply genre filter if needed
    if genre_filter != "All":
        filtered_ids = []
        for movie_id in movie_df['Movie_ID'].tolist():
            meta = get_movie_metadata(movie_id)
            if meta and meta.get('Genre'):
                try:
                    genre_list = ast.literal_eval(meta.get('Genre')) if isinstance(meta.get('Genre'), str) else meta.get('Genre')
                    genres = [g['name'] for g in genre_list if 'name' in g]
                    if genre_filter in genres:
                        filtered_ids.append(movie_id)
                except (ValueError, SyntaxError, TypeError):
                    continue
        
        unique_similar_movies = [m for m in unique_similar_movies if m[0] in filtered_ids]
    
    # Sort by score and return requested number
    unique_similar_movies.sort(key=lambda x: x[1], reverse=True)
    return unique_similar_movies[:count]


def display_movie_metadata(movie_id, user_id=None, context=None):
    meta = get_movie_metadata(movie_id)
    if meta:
        genres = meta.get('Genre', 'N/A')
        try:
            genre_list = ast.literal_eval(genres) if isinstance(genres, str) else genres
            genres = ", ".join([g['name'] for g in genre_list if 'name' in g])
        except (ValueError, SyntaxError, TypeError):
            pass  

        # Display movie title
        st.markdown(f"<h3 style='font-size: 24px; font-weight: bold;'>🎬 {meta['Name']}</h3>", unsafe_allow_html=True)

        # Display poster image if available
        if meta.get("Poster_URL"):
            st.image(meta["Poster_URL"], width=200)

        # Use columns to display some information side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Genre:** {genres}")
            st.write(f"**Language:** {meta.get('Language', 'N/A')}")
        with col2:
            st.write(f"**IMDb Rating:** {meta.get('IMDb_Rating', 'N/A')}")
            st.write(f"**Released:** {meta.get('Year', 'N/A')}")
        with col3:
            st.write(f"**Votes:** {meta.get('IMDb_Votes', 'N/A')}")
            st.write(f"**Runtime:** {meta.get('Runtime', 'N/A')} min")

        # Display additional information below in a card-style container
        st.markdown(f"""
        <div style="background-color: #1f1f1f; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-top: 15px;">
            <p style="color: #ffffff;"><strong>Tagline:</strong> {meta.get('Tagline', 'N/A')}</p>
            <p style="color: #ffffff;"><strong>Description:</strong> {meta.get('Description', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add like/dislike buttons right after description
        if user_id:  # Check if user_id exists
            # Load existing feedback
            all_feedback = st.session_state.all_feedback
            user_id_str = str(user_id)
            movie_id_str = str(movie_id)
            
            # Check if user has already provided feedback
            user_feedback = all_feedback.get(user_id_str, {})
            existing_feedback = user_feedback.get(movie_id_str, None)
            
            # Create columns for like/dislike buttons
            like_col, dislike_col = st.columns([1, 1])
            
            # Generate unique key based on context
            button_context = context if context else random.randint(0, 100000)
            
            if existing_feedback:
                if existing_feedback == "like":
                    st.success("✅ You liked this movie")
                else:
                    st.error("❌ You disliked this movie")
            else:
                with like_col:
                    if st.button("👍 Like", key=f"like_{movie_id}_{user_id}_{button_context}"):
                        if user_id_str not in all_feedback:
                            all_feedback[user_id_str] = {}
                        all_feedback[user_id_str][movie_id_str] = "like"
                        save_feedback(all_feedback)
                        st.session_state.all_feedback = all_feedback
                        st.success("✅ Movie liked!")
                        st.rerun()

                with dislike_col:
                    if st.button("👎 Dislike", key=f"dislike_{movie_id}_{user_id}_{button_context}"):
                        if user_id_str not in all_feedback:
                            all_feedback[user_id_str] = {}
                        all_feedback[user_id_str][movie_id_str] = "dislike"
                        save_feedback(all_feedback)
                        st.session_state.all_feedback = all_feedback
                        st.error("❌ Movie disliked!")
                        st.rerun()

        # Add separator line after feedback
        st.markdown("---")


def main():
    st.set_page_config(page_title="🎞 Netflix Lite", layout="wide")
    st.title("🍿 Netflix Lite")
    
    # Initialize session state
    initialize_session_state()
    
    # Load feedback data
    if 'all_feedback' not in st.session_state:
        st.session_state.all_feedback = load_feedback()
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Existing User","👤 New User", "🔍 Explore Similar Movies", "❤️ Liked Movies", "📋 My Feedback"])

    # --- TAB 1: Personalized Recommendations ---
    with tab1:
        user_id_input = st.text_input("Enter your User ID")
        user_id = int(user_id_input) if user_id_input.isdigit() else None
        # If user ID changed, clear recommendations
        if 'current_user_id' in st.session_state and st.session_state.current_user_id != user_id:
            if 'recommendations' in st.session_state:
                del st.session_state.recommendations
        
        st.session_state.current_user_id = user_id
        top_n = st.slider("How many recommendations would you like?", 1, 20, 5)

        genre_filter = "All"
        if 'Genre' in movie_df.columns:
            # Extract unique genres from the Genre column
            all_genres = set()
            for genres in movie_df['Genre'].dropna():
                try:
                    genre_list = ast.literal_eval(genres)
                    for genre in genre_list:
                        if isinstance(genre, dict) and 'name' in genre:
                            all_genres.add(genre['name'])
                except (ValueError, SyntaxError):
                    continue
            
            genre_filter = st.selectbox(
                "Filter by Genre (optional):",
                options=["All"] + sorted(list(all_genres))
            )

        # Button to trigger recommendations
        if st.button("Show Recommendations") or ('recommendations' not in st.session_state and user_id is not None and user_id > 0):

            # Personalized recommendations
            st.subheader(f"🎯 Top {top_n} picks for User {user_id}")
            recommendations = recommend_top_n(user_id, top_n * 3, genre_filter)  # Get even more recommendations to account for filtering
            
            # Filter out liked and disliked movies
            user_id_str = str(user_id)
            user_feedback = st.session_state.all_feedback.get(user_id_str, {})
            filtered_recommendations = []
            
            for movie_id, score, attn in recommendations:
                movie_id_str = str(movie_id)
                # Only include movies that haven't received any feedback yet
                if movie_id_str not in user_feedback:
                    filtered_recommendations.append((movie_id, score, attn))
                    
            # Take only top_n after filtering
            filtered_recommendations = filtered_recommendations[:top_n]
                
            # If we filtered out movies, show a message
            if len(filtered_recommendations) < top_n and len(filtered_recommendations) < len(recommendations):
                st.info(f"Some recommended movies were filtered out based on your previous feedback.")
                
            st.session_state.recommendations = filtered_recommendations
            st.session_state.recommend_user_id = user_id

        # Display recommendations if available
        if "recommendations" in st.session_state:
            if not st.session_state.recommendations:
                st.info("No recommendations available after filtering out movies you've already rated. Try exploring different genres or increasing the number of recommendations.")
            else:
                for movie_id, score, attn in st.session_state.recommendations:
                    display_movie_metadata(movie_id, st.session_state.recommend_user_id)

        # --- Display Trending Movies when no User ID is given
        if user_id is None:
            st.subheader("🔥 Trending Movies Based on IMDb Rating and Popularity")
            trending_movies = movie_df.sort_values(
                by=["IMDb_Rating", "popularity"],
                ascending=[False, False]
            ).head(top_n)

            for _, row in trending_movies.iterrows():
                display_movie_metadata(row['Movie_ID'])


    with tab2:
        st.subheader("✨ Discover Movies")
        
        movie_options = movie_df[['Movie_ID', 'Name']].drop_duplicates()
        movie_dict = {f"{row['Name']} ({row['Movie_ID']})": row['Movie_ID'] for _, row in movie_options.iterrows()}

        selected = st.multiselect("Pick 5 movies you like:", options=list(movie_dict.keys()))

        if selected:
            # Generate user ID if not exists
            if st.session_state.persistent_temp_user_id is None:
                # Get all existing user IDs
                existing_user_ids = set()
                for user_id in st.session_state.all_feedback.keys():
                    if user_id.startswith('new_user_'):
                        existing_user_ids.add(int(user_id.split('_')[2]))
                
                # Generate a new unique ID with variable length
                while True:
                    # Generate random length between 4 and 10
                    id_length = random.randint(4, 10)
                    # Generate ID with specified length
                    min_val = 10 ** (id_length - 1)  # e.g., 1000 for 4 digits
                    max_val = (10 ** id_length) - 1  # e.g., 9999 for 4 digits
                    new_id = random.randint(min_val, max_val)
                    
                    if new_id not in existing_user_ids:
                        new_user_id = f"new_user_{new_id}"
                        st.session_state.persistent_temp_user_id = new_user_id
                        st.session_state.temp_user_id = new_user_id
                        # Initialize empty feedback dictionary for new user
                        st.session_state.all_feedback[new_user_id] = {}
                        save_feedback(st.session_state.all_feedback)
                        st.success(f"✅ Your New User ID is: {new_user_id}")
                        st.info("Please save this ID for future use!")
                        break

            st.session_state.temp_user_id = st.session_state.persistent_temp_user_id

            # Display selected movies with feedback buttons
            st.markdown("### Your Selected Movies:")
            for movie_name in selected:
                movie_id = movie_dict[movie_name]
                display_movie_metadata(movie_id, st.session_state.temp_user_id, context=f"selected_{movie_name}")

        if len(selected) == 5:
            # Generate recommendations
            selected_ids = [movie_dict[m] for m in selected]
            st.success("Generating personalized recommendations...")
            
            recs_by_genre = get_new_user_recommendations(selected_ids, movie_df)
            
            st.markdown("### 🎉 Recommended Movies")
            for genre, movie_ids in recs_by_genre.items():
                st.markdown(f"#### {genre}")
                for movie_id in movie_ids:
                    display_movie_metadata(movie_id, st.session_state.temp_user_id, context=f"rec_{genre}_{movie_id}")


    # --- TAB 2: Similar Movie Finder ---
    with tab3:
        movie_name = st.text_input("Type a movie name to discover similar titles")

        if st.button("🔎 Find Similar"):
            if not movie_name:
                st.warning("Please enter a movie name.")
            else:
                st.subheader(f"Movies like: {movie_name}")
                similar = find_similar_movies(movie_name)

                if not similar:
                    st.info("No similar movies found.")
                else:
                    for movie_id, score in similar:
                        display_movie_metadata(movie_id)

    # --- TAB 3: Feedback Viewer ---

    with tab4:
        st.subheader("❤️ Movies You've Liked")
        # Get either current user ID or temp user ID
        user_id = st.session_state.current_user_id if st.session_state.current_user_id else st.session_state.get('temp_user_id')
        
        if user_id:
            user_id_str = str(user_id)
            user_feedback = st.session_state.all_feedback.get(user_id_str, {})
            
            # Get all liked movies
            liked_movies = []
            for movie_id_str, feedback_type in user_feedback.items():
                if feedback_type == "like":
                    liked_movies.append(int(movie_id_str))
            
            if liked_movies:
                st.write(f"You have liked {len(liked_movies)} movies.")
                for movie_id in liked_movies:
                    display_movie_metadata(movie_id, user_id)
            else:
                st.info("You haven't liked any movies yet. Start liking movies to see them here!")
        else:
            st.info("Please enter a User ID or select 5 movies as a new user to see your liked movies.")

    with tab5:
        st.subheader("Your Feedback History")
        
        # Get either current user ID or temp user ID
        user_id = st.session_state.current_user_id if st.session_state.current_user_id else st.session_state.get('temp_user_id')
        
        if user_id:
            user_id_str = str(user_id)
            user_feedback = st.session_state.all_feedback.get(user_id_str, {})
            
            if user_feedback:
                # Create a list of feedback items
                feedback_items = []
                for movie_id, feedback_type in user_feedback.items():
                    meta = get_movie_metadata(int(movie_id))
                    if meta:
                        movie_name = meta.get('Name', f"Movie {movie_id}")
                        feedback_items.append({
                            "User_ID": user_id_str,
                            "Movie_ID": movie_id,
                            "Movie_Name": movie_name,
                            "Feedback": feedback_type
                        })
                
                if feedback_items:
                    df = pd.DataFrame(feedback_items)
                    # Reset index starting from 1 and name it "No."
                    df.index = range(1, len(df) + 1)
                    df.index.name = "No."
                    
                    # Create a styled dataframe with custom CSS
                    styled_df = df.reset_index().set_index("No.")[["Movie_Name", "Feedback"]]
                    
                    # Calculate exact height based on number of rows
                    row_height = 50  # height per row in pixels
                    header_height = 35  # height for header in pixels
                    total_height = (len(styled_df) * row_height) + header_height
                    
                    # Update the CSS styling
                    st.markdown(
                        """
                        <style>
                            .stDataFrame {
                                width: 100%;
                                margin: auto;
                            }
                            .stDataFrame table {
                                width: 100% !important;
                                margin: auto;
                                border-collapse: collapse;
                                height: auto !important;
                            }
                            .stDataFrame td:first-child, .stDataFrame th:first-child {
                                text-align: left !important;
                                padding-left: 15px !important;
                                width: 8% !important;
                                white-space: nowrap !important;
                            }
                            .stDataFrame td:nth-child(2) {
                                text-align: left !important;
                                padding-left: 15px !important;
                                width: 62% !important;
                            }
                            .stDataFrame td:last-child {
                                text-align: center !important;
                                width: 30% !important;
                            }
                            .stDataFrame th {
                                text-align: left !important;
                                padding: 10px !important;
                                font-size: 16px !important;
                                background-color: #1f1f1f;
                            }
                            .stDataFrame td {
                                padding: 10px !important;
                                font-size: 16px !important;
                            }
                            .stDataFrame tbody tr {
                                height: 50px !important;
                            }
                            /* Hide extra rows */
                            .stDataFrame tbody tr:nth-child(n+${len(styled_df)+1}) {
                                display: none !important;
                            }
                        </style>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Display the dataframe with exact dimensions
                    st.dataframe(
                        styled_df,
                        width=800,
                        height=total_height,  # Dynamic height based on content
                        hide_index=False
                    )
                else:
                    st.info(f"No feedback history available yet.")
            else:
                st.info(f"No feedback history available yet.")
            
            if st.button("🧹 Clear My Feedback"):
                if user_id_str in st.session_state.all_feedback:
                    del st.session_state.all_feedback[user_id_str]
                    save_feedback(st.session_state.all_feedback)
                    st.success(f"Feedback cleared successfully!")
                    st.rerun()
        else:
            st.info("Please enter a User ID or select 5 movies as a new user to see your feedback history.")

if __name__ == "__main__":
    main()


