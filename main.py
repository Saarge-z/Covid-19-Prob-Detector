from flask import Flask, request, render_template
import pickle
# Create Flask app load app.config
app = Flask(__name__)

# open a file, where you data is stored and load it
file = open('model.pkl', 'rb')
model = pickle.load(file)
clf = model[0]
sc_X = model[1]
file.close()

@app.route('/', methods=['GET', 'POST'])
def screening():

    if request.method == 'POST':
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        pain = int(myDict['pain'])
        runnnyNose = int(myDict['runnyNose'])
        diffBreath = int(myDict['diffBreath'])


        # clf.predict([[100,1,21,1,1]]) # provides us with 0 or 1 (whether we have infec or not)
        inputFeatures = [fever, pain, age, runnnyNose, diffBreath]
        infProb = clf.predict_proba(sc_X.transform([inputFeatures]))[0][1] # predicts infection prob. along with no infection prob.
        return render_template('view.html', inf=round(infProb*100))
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)