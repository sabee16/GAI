"""
Text Summarization using Hugging Face Transformers
----------------------------------------------------
Steps performed:
1. Uses the pretrained "sshleifer/distilbart-cnn-12-6" summarization model
   (a distilled version of BART fine-tuned on CNN/DailyMail — fast and
   good quality for short paragraphs).
2. Takes a paragraph (~200-300 words) as input.
3. Generates a summary.
4. Compares original vs. summary word counts.
"""

from transformers import pipeline

# ---------------------------------------------------------
# Step 1 & 2: Load a pretrained summarization pipeline
# ---------------------------------------------------------
print("Loading pretrained summarization model (this may take a moment)...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# ---------------------------------------------------------
# Step 3: Input paragraph (~200-300 words)
# You can replace this with your own text.
# ---------------------------------------------------------
paragraph = """
Artificial intelligence has rapidly transformed the way businesses and individuals
approach problem-solving across nearly every industry. From healthcare to finance,
transportation to entertainment, machine learning models are now capable of
analyzing massive datasets, identifying patterns, and making predictions with
accuracy that often surpasses human capability. In healthcare, AI systems assist
doctors in diagnosing diseases earlier by scanning medical images for subtle signs
that might otherwise go unnoticed. In finance, algorithms detect fraudulent
transactions in real time, protecting both banks and customers from significant
losses. Self-driving cars rely on a combination of computer vision, sensor fusion,
and deep learning to navigate complex environments safely, reducing the likelihood
of accidents caused by human error. Meanwhile, streaming platforms use
recommendation engines powered by AI to personalize content, keeping users engaged
by suggesting movies, shows, or songs tailored to their preferences. Despite these
advances, the rise of AI has also raised important questions about privacy, job
displacement, and the ethical use of automated decision-making systems. Policymakers
and technologists alike are grappling with how to regulate AI responsibly while
still encouraging innovation. Many experts argue that transparency, accountability,
and human oversight must remain central to AI development to ensure these powerful
tools benefit society as a whole rather than concentrating power or causing harm.
As AI continues to evolve, striking the right balance between progress and
responsibility will likely define the next decade of technological development.
""".strip()

# ---------------------------------------------------------
# Step 4: Generate the summary
# ---------------------------------------------------------
result = summarizer(paragraph, max_length=80, min_length=25, do_sample=False)
summary = result[0]["summary_text"]

# ---------------------------------------------------------
# Step 5: Compare word counts
# ---------------------------------------------------------
original_word_count = len(paragraph.split())
summary_word_count = len(summary.split())
reduction_pct = 100 * (1 - summary_word_count / original_word_count)

print("\n" + "=" * 60)
print("ORIGINAL PARAGRAPH:")
print("=" * 60)
print(paragraph)

print("\n" + "=" * 60)
print("GENERATED SUMMARY:")
print("=" * 60)
print(summary)

print("\n" + "=" * 60)
print("WORD COUNT COMPARISON:")
print("=" * 60)
print(f"Original word count : {original_word_count}")
print(f"Summary word count  : {summary_word_count}")
print(f"Reduction           : {reduction_pct:.1f}%")
