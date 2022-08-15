# Step 1 - Importing modules
import os
import pyshorteners
from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Step 2 - Instantiating/Creating object
app = Flask(__name__)

##################### SQL ALCHEMY CONFIGURATION #############################
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
Migrate(app, db)

#############################################################################
############################# CREATE A MODEL ################################
class hist_class(db.Model):
    __tablename__ = 'history_table'
    id = db.Column(db.Integer, primary_key = True)
    long = db.Column(db.Text)
    short = db.Column(db.Text)

    def __init__(self, long, short):
        self.long = long
        self.short = short

db.session.query(hist_class).delete()
db.session.commit()
#############################################################################
# Step 3 - Binding routes with functionalities

@app.route('/', methods = ['GET', 'POST'])
def home_func():
    if request.method == 'POST':
        long_url = request.form['in_long']
        shorten = pyshorteners.Shortener()
        short_url = shorten.tinyurl.short(long_url)

        new_link = hist_class(long_url, short_url)
        db.session.add(new_link)
        db.session.commit()

        return render_template('home.html', s = short_url)
    return render_template('home.html')


@app.route('/history')
def hist_func():
    all_link = hist_class.query.all()
    return render_template('history.html', all = all_link)


#############################################################################
# Step 4 - Running the app
if __name__ == '__main__':
    app.run(debug = True)