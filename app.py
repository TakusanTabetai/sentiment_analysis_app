from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.optim import AdamW
import os
import json
from collections import Counter

app = Flask(__name__)

# モデルとトークナイザーのパス
model_path = 'models/sentiment_model'
tokenizer_path = 'models/tokenizer'

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルとトークナイザーの読み込み
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
else:
    # 事前学習済みモデルの読み込み
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

model.to(device)

# オプティマイザーの設定
optimizer = AdamW(model.parameters(), lr=2e-5)

@app.route('/')
def home():
    stats = get_feedback_stats()
    return render_template('home.html', stats=stats)

def get_feedback_stats():
    if os.path.exists('feedback.json'):
        with open('feedback.json', 'r') as f:
            all_feedbacks = json.load(f)
        
        total_feedbacks = len(all_feedbacks)
        
        # 正確度の計算には最新の100件を使用
        recent_feedbacks = all_feedbacks[-100:]
        correct_recent = sum(1 for feedback in recent_feedbacks if feedback['isCorrect'])
        accuracy = (correct_recent / len(recent_feedbacks)) * 100 if recent_feedbacks else 0
        
        # 正確な回答数と他の統計は全データを使用
        correct_feedbacks = sum(1 for feedback in all_feedbacks if feedback['isCorrect'])
        sentiment_counts = Counter(feedback['modelSentiment'] for feedback in all_feedbacks)
        
        return {
            'total_feedbacks': total_feedbacks,
            'correct_feedbacks': correct_feedbacks,
            'accuracy': accuracy,
            'angry_count': sentiment_counts['angry'],
            'not_angry_count': sentiment_counts['not_angry']
        }
    else:
        return {
            'total_feedbacks': 0,
            'correct_feedbacks': 0,
            'accuracy': 0,
            'angry_count': 0,
            'not_angry_count': 0
        }
    
@app.route('/batch_analysis')
def batch_analysis():
    return render_template('batch_analysis.html')

@app.route('/single_analysis')
def single_analysis():
    return render_template('single_analysis.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    text = request.json['text']
    
    # テキストの前処理とトークン化
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # 感情を予測
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        probability = probabilities[0].tolist()
    
    if prediction == 1:
        sentiment = 'angry'
        advice = "文章が怒りや不満を表現しているようです。より穏やかな表現を使うことを検討してみてはいかがでしょうか。"
    else:
        sentiment = 'not_angry'
        advice = "文章は穏やかに見えます。このスタイルを維持してください。"
    
    response = {
        'text': text,
        'sentiment': sentiment,
        'advice': advice,
        'probability': {
            'angry': probability[1],
            'not_angry': probability[0]
        }
    }
    
    # 分析結果を保存
    save_analysis_result(response)
    
    return jsonify(response)

def save_analysis_result(result):
    if not os.path.exists('analysis_results.json'):
        with open('analysis_results.json', 'w') as f:
            json.dump([], f)
    
    with open('analysis_results.json', 'r+') as f:
        results = json.load(f)
        results.append(result)
        f.seek(0)
        json.dump(results, f)

@app.route('/get_test_texts', methods=['GET'])
def get_test_texts():
    if os.path.exists('analysis_results.json'):
        with open('analysis_results.json', 'r') as f:
            all_results = json.load(f)
        return jsonify(all_results[-10:])  # 最新の10件を返す
    else:
        return jsonify([])

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.json
    
    # フィードバックを保存
    if not os.path.exists('feedback.json'):
        with open('feedback.json', 'w') as f:
            json.dump([], f)
    
    with open('feedback.json', 'r+') as f:
        feedbacks = json.load(f)
        feedbacks.append(feedback)
        f.seek(0)
        json.dump(feedbacks, f)
    
    # モデルを更新
    update_model(feedback)
    
    return jsonify({'status': 'success'})

def update_model(feedback):
    text = feedback['text']
    is_correct = feedback['isCorrect']
    model_sentiment = feedback['modelSentiment']
    
    # 正解ラベルを設定
    if is_correct:
        label = 1 if model_sentiment == 'angry' else 0
    else:
        label = 0 if model_sentiment == 'angry' else 1
    
    # テキストをトークン化
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    labels = torch.tensor([label]).to(device)
    
    # モデルを訓練モードに設定
    model.train()
    
    # 順伝播
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    
    # 逆伝播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # モデルを評価モードに戻す
    model.eval()
    
    # 更新されたモデルを保存
    model.save_pretrained(model_path)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)