import nltk
from flask import Flask, request, jsonify

# 下载必要的 nltk 数据
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)


@app.route('/pos-tag', methods=['POST'])
def pos_tagging():
    try:
        data = request.get_json(force=True)
        text = data.get('text')
        if text is None:
            return jsonify({"error": "Missing 'text' in request data"}), 400
        # 对输入文本进行分词
        tokens = nltk.word_tokenize(text)
        # 进行词性标注
        tagged_words = nltk.pos_tag(tokens)
        result = dict(tagged_words)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
    