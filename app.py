import plotly.io as pio
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Replace with a strong secret key

# Load data
users_file = "users.csv"
reviews_file = "movie_reviews.csv"

users_df = pd.read_csv(users_file)
reviews_df = pd.read_csv(reviews_file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_df[(users_df["username"] == username) & (users_df["password"] == password)]
        if not user.empty:
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid username or password.")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    global users_df  # Declare `users_df` as global before using it
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if username already exists
        if username in users_df["username"].values:
            flash("Username already exists. Please choose a different one.")
            return redirect(url_for("signup"))

        # Create a new DataFrame for the new user
        new_user = pd.DataFrame({"username": [username], "password": [password]})

        # Concatenate the new user to the existing DataFrame
        users_df = pd.concat([users_df, new_user], ignore_index=True)

        # Save the updated DataFrame to the CSV file
        users_df.to_csv(users_file, index=False)

        flash("Sign-up successful! You can now log in.")
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

        # Sentiment counts for the pie chart
    sentiment_counts = reviews_df['sentiment'].value_counts()
    fig_pie = px.pie(names=sentiment_counts.index, values=sentiment_counts.values,
                     title="Overall Sentiment Distribution")
    pie_chart_html = pio.to_html(fig_pie, full_html=False)

    # Create a bar graph: Sentiments per Movie
    bar_data = reviews_df.groupby("movie_name")["sentiment"].value_counts().unstack(fill_value=0)
    fig_bar = px.bar(
        bar_data,
        x=bar_data.index,
        y=["Positive", "Neutral", "Negative"],
        title="Sentiments Per Movie",
        labels={"value": "Count", "movie_name": "Movie Name"},
        barmode="group"
    )
    bar_chart_html = pio.to_html(fig_bar, full_html=False)

    # Train the SVM model (example using review data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews_df["review"])  # Convert text to feature vectors
    y = reviews_df["sentiment"].map({"Positive": 1, "Neutral": 0, "Negative": -1})  # Encode sentiments

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Compute accuracy
    svm_accuracy = accuracy_score(y_test, y_pred)
    # Perform K-Means clustering
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(reviews_df["review"])  # Convert reviews to TF-IDF features

    num_clusters = 3  # Define the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    reviews_df["Cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Create a bar chart for clusters
    cluster_counts = reviews_df["Cluster"].value_counts()
    fig_cluster_bar = px.bar(
        x=cluster_counts.index,
        y=cluster_counts.values,
        title="Cluster Distribution",
        labels={"x": "Cluster", "y": "Number of Reviews"}
    )
    cluster_bar_html = pio.to_html(fig_cluster_bar, full_html=False)

    return render_template(
        "dashboard.html",
        reviews=reviews_df.to_dict(orient="records"),
        pie_chart_html=pie_chart_html,
        bar_chart_html=bar_chart_html,
        cluster_bar_html=cluster_bar_html,
        svm_accuracy=svm_accuracy
    )

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("index"))


@app.route("/update_review", methods=["POST"])
def update_review():
    if "username" not in session:
        return redirect(url_for("login"))

    movie_name = request.form["movie_name"]
    new_review = request.form["new_review"]
    new_sentiment = request.form["new_sentiment"]

    reviews_df.loc[reviews_df["movie_name"] == movie_name, "review"] = new_review
    reviews_df.loc[reviews_df["movie_name"] == movie_name, "sentiment"] = new_sentiment
    reviews_df.to_csv(reviews_file, index=False)

    return redirect(url_for("dashboard"))

@app.route("/add_review", methods=["POST"])
def add_review():
    global reviews_df  # Declare reviews_df as global
    if "username" not in session:
        return redirect(url_for("login"))

    # Get form data
    movie_name = request.form["movie_name"]
    new_review = request.form["new_review"]
    new_sentiment = request.form["new_sentiment"]

    # Check if the movie already has a review
    if movie_name in reviews_df["movie_name"].values:
        flash("Review for this movie already exists!")
        return redirect(url_for("dashboard"))

    # Create new review data as a DataFrame
    new_review_data = pd.DataFrame({
        "movie_name": [movie_name],
        "review": [new_review],
        "sentiment": [new_sentiment]
    })

    # Concatenate the new review data with the existing DataFrame
    reviews_df = pd.concat([reviews_df, new_review_data], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    reviews_df.to_csv(reviews_file, index=False)

    flash("Review added successfully!")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True)
