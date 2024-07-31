from flask import Flask, request, jsonify
import pandas as pd
import textdistance
import re
from collections import Counter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


words = []

with open('book.txt', 'r', encoding='utf-8') as f:
    file_name_data = f.read()
    file_name_data = file_name_data.lower()
    words = re.findall(r'\w+', file_name_data)

V = set(words)

word_freq_dict = Counter(words)

probs = {}
Total = sum(word_freq_dict.values())
for k in word_freq_dict.keys():
    probs[k] = word_freq_dict[k] / Total

def my_autocorrect(input_word, V, word_freq_dict, probs, threshold=0.5, top_n=5):
    input_word = input_word.lower()
    if input_word in V:
        return ['Your word seems to be correct']
    else:
        similarities = [1 - textdistance.Jaccard(qval=2).distance(v, input_word) for v in word_freq_dict.keys()]
        df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
        df = df.rename(columns={'index': 'Word', 0: 'Prob'})
        df['Similarity'] = similarities

        df_filtered = df[df['Similarity'] >= threshold]

        output = df_filtered.sort_values(['Similarity', 'Prob'], ascending=False).head(top_n)

        return output['Word'].tolist()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_word = data.get('text', '')
    result = my_autocorrect(input_word, V, word_freq_dict, probs)
    return jsonify({'suggestions': result})

if __name__ == '__main__':
    app.run(debug=True)