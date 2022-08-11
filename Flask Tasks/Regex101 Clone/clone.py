# Step 1 - Importing important modules.
from ast import And
from flask import Flask, render_template, request
import re

# Step 2 - Instantiating or creating object.
app = Flask(__name__)

#########################################################################################
# Step 3 - Creating routes and binding with functionality.
@app.route('/', methods = ['GET', 'POST'])
def home_page():
    
    if request.method == 'POST':
        pat = request.form['in_re']
        str = request.form['in_str']

        pattern = re.compile(pat)
        n = str

        total = len(pattern.findall(n))
        matched = pattern.findall(n)

        return render_template("home.html", a = total, b = matched, c = pat, n = str)
    return render_template("home.html")

#########################################################################################
# Step 4 - Running the app.
if __name__ == '__main__':
    app.run(debug = True)