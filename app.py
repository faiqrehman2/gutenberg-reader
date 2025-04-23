from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from database import db, init_db
from models import Book
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv
import os
import re
from langdetect import detect, LangDetectException
from collections import Counter
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

app = Flask(__name__)
CORS(app)
load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///books.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

init_db(app)

def extract_metadata(soup):
    metadata = {}

    title_tag = soup.find("meta", attrs={"name": "title"})
    if title_tag:
        metadata["Title"] = title_tag["content"]

    author_tag = soup.find("a", rel="marcrel:aut")
    if author_tag:
        metadata["Author"] = author_tag.get_text(strip=True)

    metadata_table = soup.find("table", class_="bibrec")
    if metadata_table:
        for row in metadata_table.find_all("tr"):
            header = row.find("th")
            value = row.find("td")

            if header and value:
                key = header.get_text(strip=True)
                value_text = value.get_text(" ", strip=True)

                link = value.find("a", href=True)
                if link:
                    value_text += f" ({link['href']})"

                metadata[key] = value_text

    return metadata


@app.route("/summary/<int:book_id>", methods=["GET"])
def get_plot_summary(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    initial_prompt = (
        f"Provide a concise plot summary (3-5 sentences) for '{book.title}' by {book.author}. "
        "If the summary isn't available or you're unsure, reply only with 'not found'."
    )

    try:
        initial_response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": initial_prompt}],
            temperature=0.1,
            max_tokens=500,
        )

        initial_summary = initial_response.choices[0].message.content.strip()

        if initial_summary.lower() == "not found":
            detailed_prompt = (
                "Provide a concise plot summary (3-5 sentences) based on the following book content:\n\n"
                f"{book.content}"
            )

            detailed_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-specdec",
                messages=[{"role": "user", "content": detailed_prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            detailed_summary = detailed_response.choices[0].message.content.strip()

            return jsonify({"summary": detailed_summary, "source": "content"})

        return jsonify({"summary": initial_summary, "source": "title_author"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/characters/<int:book_id>", methods=["GET"])
def identify_key_characters(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    try:
        character_prompt = (
            f"Identify the main characters in '{book.title}' by {book.author}. "
            f"For each character, provide:\n"
            f"1. Character name\n"
            f"2. Brief description (1-2 sentences)\n"
            f"3. Role in the story (protagonist, antagonist, supporting character, etc.)\n\n"
            f"Format your response as a JSON array of objects with 'name', 'description', and 'role' properties. "
            f"Provide all the characters. If you're unsure, only then analyze the first part of the book content:\n\n"
            f"{book.content[:7000]}"
        )

        characters_response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": character_prompt}],
            temperature=0.1,
            max_tokens=800,
            response_format={"type": "json_object"}
        )

        response_content = characters_response.choices[0].message.content.strip()

        return jsonify({"characters": response_content})

    except Exception as e:
        try:
            text_sample = book.content[:10000]
            sentences = nltk.sent_tokenize(text_sample)

            persons = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tagged = nltk.pos_tag(words)
                named_entities = nltk.ne_chunk(tagged)

                for chunk in named_entities:
                    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                        name = ' '.join([c[0] for c in chunk])
                        persons.append(name)

            # Count occurrences and get the most common names
            person_counts = Counter(persons)
            top_characters = [{"name": name, "count": count}
                              for name, count in person_counts.most_common(10)]

            return jsonify({
                "characters": top_characters,
                "note": "Basic character extraction due to API failure. Character details unavailable."
            })
        except Exception as nested_e:
            return jsonify({"error": f"Failed to extract characters: {str(nested_e)}"}), 500


@app.route("/language/<int:book_id>", methods=["GET"])
def detect_language(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    try:
        sample_text = book.content[:1000].strip()
        detected_lang = detect(sample_text)

        language_map = {
            'en': 'English',
            'fr': 'French',
            'es': 'Spanish',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'nl': 'Dutch',
            'ru': 'Russian',
            'ja': 'Japanese',
            'zh-cn': 'Chinese (Simplified)',
            'zh-tw': 'Chinese (Traditional)',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'la': 'Latin'
        }

        language_name = language_map.get(detected_lang, detected_lang)

        return jsonify({
            "language_code": detected_lang,
            "language": language_name
        })
    except LangDetectException as e:
        return jsonify({"error": f"Language detection failed: {str(e)}"}), 500

@app.route("/sentiment/<int:book_id>", methods=["GET"])
def analyze_sentiment(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    section = request.args.get('section', 'beginning')

    try:
        text_sample = ""
        if section == 'beginning':
            text_sample = book.content[:5000]
        elif section == 'middle':
            middle_index = len(book.content) // 2
            text_sample = book.content[middle_index:middle_index + 5000]
        elif section == 'end':
            text_sample = book.content[-5000:]
        elif section == 'overall':
            beginning = book.content[:2000]
            middle_index = len(book.content) // 2
            middle = book.content[middle_index:middle_index + 2000]
            end = book.content[-2000:]
            text_sample = beginning + middle + end
        else:
            text_sample = book.content[:5000]

        # Clean the text
        text_sample = re.sub(r'\n+', ' ', text_sample)
        text_sample = re.sub(r'\s+', ' ', text_sample)

        blob = TextBlob(text_sample)
        sentiment_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity

        sentiment_label = "neutral"
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"

        try:
            emotion_prompt = (
                f"Analyze the emotional tone and overall mood of this text from '{book.title}'. "
                f"Identify the dominant emotions, mood shifts, and overall atmosphere. "
                f"Provide your analysis in a concise paragraph:\n\n{text_sample[:3000]}"
            )

            emotion_response = groq_client.chat.completions.create(
                model="gemma2-9b-it",
                messages=[{"role": "user", "content": emotion_prompt}],
                temperature=0.1,
                max_tokens=300
            )

            emotion_analysis = emotion_response.choices[0].message.content.strip()
        except:
            emotion_analysis = "Emotional analysis unavailable."

        return jsonify({
            "section": section,
            "sentiment_score": sentiment_score,
            "sentiment": sentiment_label,
            "subjectivity": subjectivity_score,
            "emotional_analysis": emotion_analysis
        })

    except Exception as e:
        return jsonify({"error": f"Sentiment analysis failed: {str(e)}"}), 500


@app.route("/readability/<int:book_id>", methods=["GET"])
def analyze_readability(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    try:
        sample_text = book.content[:5000]

        sample_text = re.sub(r'\n+', ' ', sample_text)
        sample_text = re.sub(r'\s+', ' ', sample_text)

        sentences = sent_tokenize(sample_text)
        words = word_tokenize(sample_text)

        words = [word for word in words if word.isalnum()]

        num_sentences = len(sentences)
        num_words = len(words)
        num_characters = sum(len(word) for word in words)

        avg_sentence_length = num_words / num_sentences if num_sentences > 0 else 0

        avg_word_length = num_characters / num_words if num_words > 0 else 0

        # Calculate Flesch Reading Ease
        flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 5))

        difficulty = "Very difficult"
        if flesch_reading_ease > 90:
            difficulty = "Very easy"
        elif flesch_reading_ease > 80:
            difficulty = "Easy"
        elif flesch_reading_ease > 70:
            difficulty = "Fairly easy"
        elif flesch_reading_ease > 60:
            difficulty = "Standard"
        elif flesch_reading_ease > 50:
            difficulty = "Fairly difficult"
        elif flesch_reading_ease > 30:
            difficulty = "Difficult"

        # Calculate vocabulary complexity (percentage of uncommon words)
        stop_words = set(stopwords.words('english'))
        common_words = stop_words.union(set([
            'said', 'would', 'could', 'one', 'two', 'time', 'like', 'man', 'woman',
            'day', 'night', 'went', 'came', 'made', 'make', 'see', 'saw', 'look',
            'looked', 'way', 'well', 'back', 'think', 'thought', 'know', 'knew'
        ]))

        uncommon_words = [word.lower() for word in words if word.lower() not in common_words and len(word) > 3]
        uncommon_word_percentage = (len(uncommon_words) / num_words) * 100 if num_words > 0 else 0

        word_counts = Counter(uncommon_words)
        top_words = word_counts.most_common(10)

        return jsonify({
            "avg_sentence_length": round(avg_sentence_length, 2),
            "avg_word_length": round(avg_word_length, 2),
            "flesch_reading_ease": round(flesch_reading_ease, 2),
            "reading_difficulty": difficulty,
            "vocabulary_complexity": round(uncommon_word_percentage, 2),
            "top_distinctive_words": [{"word": word, "count": count} for word, count in top_words]
        })

    except Exception as e:
        return jsonify({"error": f"Readability analysis failed: {str(e)}"}), 500


@app.route("/themes/<int:book_id>", methods=["GET"])
def identify_themes(book_id):
    book = db.session.get(Book, book_id)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    try:
        theme_prompt = (
            f"Analyze '{book.title}' by {book.author} and identify its main themes, motifs, and symbols. "
            f"For each identified element, provide a brief explanation of its significance in the work. "
            f"If you're not familiar with the complete work, analyze these excerpts:\n\n"
            f"Beginning: {book.content[:2000]}\n\n"
            f"Middle: {book.content[len(book.content) // 2:len(book.content) // 2 + 2000]}\n\n"
            f"End: {book.content[-2000:]}"
        )

        themes_response = groq_client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[{"role": "user", "content": theme_prompt}],
            temperature=0.1,
            max_tokens=800
        )

        themes_analysis = themes_response.choices[0].message.content.strip()

        return jsonify({
            "themes_analysis": themes_analysis
        })
    except Exception as e:
        return jsonify({"error": f"Theme analysis failed: {str(e)}"}), 500


@app.route("/book/<int:book_id>", methods=["GET"])
def get_book(book_id):
    existing_book = db.session.get(Book, book_id)

    if existing_book:
        return jsonify({
            "id": existing_book.id,
            "title": existing_book.title,
            "author": existing_book.author,
            "content": existing_book.content,
            "metadata": existing_book.get_metadata()
        })

    content_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    metadata_url = f"https://www.gutenberg.org/ebooks/{book_id}"

    try:
        content_response = requests.get(content_url)
        if content_response.status_code != 200:
            return jsonify({"error": "Book content not found"}), 404

        content = content_response.text

        metadata_response = requests.get(metadata_url)
        soup = BeautifulSoup(metadata_response.text, "html.parser")

        title = soup.find("title").get_text(strip=True) if soup.find("title") else "Unknown Title"
        author_tag = soup.find("a", rel="marcrel:aut")
        author = author_tag.get_text(strip=True) if author_tag else "Unknown Author"

        metadata = extract_metadata(soup)

        book = Book(id=book_id, title=title, author=author, content=content)
        book.set_metadata(metadata)
        db.session.merge(book)
        db.session.commit()

        return jsonify({
            "id": book.id,
            "title": title,
            "author": author,
            "content": content,
            "metadata": metadata
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/books", methods=["GET"])
def get_saved_books():
    try:
        books = Book.query.order_by(Book.id.desc()).all()
        return jsonify([
            {"id": book.id, "title": book.title, "author": book.author, "content": book.content[:200],
             "metadata": book.get_metadata()}
            for book in books
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
