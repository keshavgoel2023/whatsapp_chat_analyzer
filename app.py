import streamlit as st
import pandas as pd
import re
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import plotly.express as px
import io

# ------------------ Helper functions ------------------

def parse_chat(file_content):
    """Parse exported WhatsApp chat file (robust to multi-line messages)."""
    pattern = r"(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[APMapm]{2})?) - (.*?): (.*)"
    messages = []
    current_msg = None

    for line in file_content.split("\n"):
        match = re.match(pattern, line)
        if match:
            if current_msg:
                messages.append(current_msg)
            date, time, sender, message = match.groups()
            current_msg = {"datetime": f"{date} {time}", "sender": sender, "message": message}
        else:
            if current_msg:
                current_msg["message"] += " " + line

    if current_msg:
        messages.append(current_msg)

    df = pd.DataFrame(messages)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", dayfirst=True)
    df["message_length"] = df["message"].apply(len)
    df["emoji_count"] = df["message"].apply(lambda x: sum(1 for c in x if c in emoji.EMOJI_DATA))
    df["url_count"] = df["message"].apply(lambda x: len(re.findall(r"http\S+", x)))
    return df.dropna(subset=["datetime"])


def basic_stats(df):
    total_msgs = len(df)
    unique_users = df["sender"].nunique()
    avg_len = df["message_length"].mean()
    return total_msgs, unique_users, avg_len


def plot_message_trend(df):
    df["date"] = df["datetime"].dt.date
    trend = df.groupby("date").size().reset_index(name="message_count")
    fig = px.line(trend, x="date", y="message_count", title="Messages per Day")
    return fig


def plot_wordcloud(df):
    text = " ".join(df["message"])
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


def cluster_messages(df):
    """Cluster messages by features."""
    X = df[["message_length", "emoji_count", "url_count"]]
    X_scaled = StandardScaler().fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    fig = px.scatter_3d(df, x="message_length", y="emoji_count", z="url_count",
                        color=df["cluster"].astype(str),
                        title="KMeans Clustering on Message Features")
    return fig


def regression_model(df):
    """Simple regression: predict message length from emoji + URL count."""
    X = df[["emoji_count", "url_count"]]
    y = df["message_length"]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    df["predicted_length"] = y_pred

    fig = px.scatter(df, x=y, y=y_pred, title="Regression: Actual vs Predicted Message Length",
                     labels={"x": "Actual", "y": "Predicted"})
    st.plotly_chart(fig)
    st.write(f"**R² Score:** {model.score(X, y):.3f}")


# ------------------ Streamlit App ------------------

st.set_page_config(page_title="WhatsApp Chat Analyzer", layout="wide")
st.title("📊 WhatsApp Chat Analyzer")
st.write("Upload an exported WhatsApp chat `.txt` file to explore insights interactively.")

uploaded_file = st.file_uploader("Choose a WhatsApp chat file (.txt)", type=["txt"])

if uploaded_file:
    stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = parse_chat(stringio.read())

    st.subheader("📋 Data Overview")
    st.dataframe(df.head())

    total_msgs, unique_users, avg_len = basic_stats(df)
    st.metric("Total Messages", total_msgs)
    st.metric("Unique Users", unique_users)
    st.metric("Avg. Message Length", f"{avg_len:.1f}")

    st.subheader("📆 Message Trends")
    st.plotly_chart(plot_message_trend(df))

    st.subheader("🧩 Word Cloud")
    plot_wordcloud(df)

    st.subheader("🎯 Clustering Patterns")
    st.plotly_chart(cluster_messages(df))

    st.subheader("📈 Regression Analysis")
    regression_model(df)

    st.subheader("💬 Top Senders")
    top_senders = df["sender"].value_counts().head(10).reset_index()
    top_senders.columns = ["Sender", "Messages"]
    fig = px.bar(top_senders, x="Sender", y="Messages", title="Top 10 Active Senders", color="Messages")
    st.plotly_chart(fig)

else:
    st.info("Upload a `.txt` chat file exported from WhatsApp to begin.")

