import json
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(json_data, font_path, output_path, lang='zh', stopwords_path='stopwords.txt', max_words=100):
    """
    Generates and displays a word cloud based on the input JSON data.

    Parameters:
    - json_data: Path to the JSON file containing the data
    - font_path: Path to the font file for word cloud rendering (use appropriate font for different languages)
    - output_path: Path where the generated word cloud image will be saved
    - lang: Language of the text ('zh' for Chinese, 'en' for English, etc.)
    - stopwords_path: Path to the stopwords file
    - max_words: Maximum number of words to display in the word cloud
    """
    # Read the JSON data
    with open(json_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    captions = ""

    # Extract titles from JSON lines
    for line in lines:
        item = json.loads(line.strip())
        captions += item.get("title", "").lower()  # Safely handle missing keys

    # Perform text segmentation based on language
    if lang == 'zh':
        words = jieba.lcut(captions)  # Segment Chinese text
    else:
        words = captions.split()  # Simple space-based split for other languages

    # Load stopwords from file
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = set(line.strip() for line in f)

    # Count word frequencies, excluding stopwords
    word_freq = {}
    for word in words:
        word = word.strip()
        if word and word not in stopwords:  # Ensure word is non-empty and not a stopword
            word_freq[word] = word_freq.get(word, 0) + 1

    # Generate the word cloud from the word frequencies
    wordcloud = WordCloud(
        width=1000, height=1000, background_color='white', font_path=font_path,
        max_words=max_words
    ).generate_from_frequencies(word_freq)

    # Save the word cloud image
    wordcloud.to_file(output_path)

    # Display the word cloud
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    plt.show()

if __name__ == '__main__':
    
    # json_data = '../data/FakeSV/data.json'
    # font_path = 'C:/Windows/Fonts/simhei.ttf'  # Chinese font path
    # output_path = 'wordcloud_fakesv.png'

    # Optional configuration for English data
    json_data = '../data/FakeTT/data.json'
    font_path = 'C:/Windows/Fonts/Arial.TTF'  # English font path
    output_path = 'wordcloud_fakett.png'

    generate_wordcloud(json_data, font_path, output_path)
