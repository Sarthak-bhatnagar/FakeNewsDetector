from flask import Flask, request, render_template
import pickle
app = Flask(__name__,template_folder='template')
model = pickle.load(open('fakemodel.pickle', 'rb'))
vect = pickle.load(open('vectorizer1.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/random', methods=['GET'])
def random():
    data = pd.read_csv("./data/fake_or_real_news_testset.csv")
    index = randrange(0, len(data)-1, 1)
    response = jsonify({'title': data.loc[index].title, 'text': data.loc[index].text})
    return response	

@app.route('/predict',methods=['POST'])
def predict():
    text=request.form['textarea1']
    def find_label(text):
        vec_newtest=vect.transform([text])
        y=model.predict(vec_newtest)
        return y[0]
    val1=find_label(text)
    return render_template('index.html', prediction_text=val1)


if __name__ == "__main__":
    app.run(debug=True)