# export FLASK_APP=server.py
# export FLASK_DEBUG=1
# python -m flask run -p 3000 --host=0.0.0.0
import os
import datetime

from flask import Flask
from flask import render_template, send_from_directory
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
from tinydb import TinyDB, Query
import redis


app = Flask(__name__)
BASE_DIR = '/tmp/courses-server'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(BASE_DIR, exist_ok=True)
redis_db = redis.StrictRedis(
    host='localhost', port=6379, db=1,
    charset="utf-8", decode_responses=True)
results_db = redis.StrictRedis(
    host='localhost', port=6379, db=2,
    charset="utf-8", decode_responses=True)

ALLOWED_EXTENSIONS = set(['h5'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload-model', methods=['GET', 'POST'])
def upload_model():
    if request.method == 'GET':
        users = list(redis_db.hgetall('users_to_id').keys())
        return render_template('upload_model.html', users=users)
    if request.method == 'POST':
        # Handle usernames
        old_name = request.form.get('username_old')
        new_name = request.form.get('username_new')
        if not new_name and not old_name:
            error = "You should specify old or new name"
            return render_template('upload_model.html', error=error)
        if new_name and old_name:
            error = "You should add new name or choose old one"
            return render_template('upload_model.html', error=error)
        # try get existed name in any case
        if new_name:
            user_name = new_name
        elif old_name:
            user_name = old_name
        user_id = redis_db.hmget('users_to_id', user_name)[0]
        if not user_id:
            print("new name was used")
            user_id = str(len(redis_db.hgetall('users_to_id')) + 1)
            redis_db.hmset('users_to_id', {new_name: user_id})
            redis_db.hmset('ids_to_user', {user_id: new_name})

        # Handle files
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            # flash('No selected file')
            return redirect(request.url)

        if file and not allowed_file(file.filename):
            error = "Failed to upload file with such extensions"
            return render_template('upload_model.html', error=error)

        user_folder_dir = os.path.join(MODELS_DIR, str(user_id))
        os.makedirs(user_folder_dir, exist_ok=True)

        filename = secure_filename(file.filename)
        today = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S_')
        filename = today + filename
        file_path = os.path.join(user_folder_dir, filename)
        file.save(file_path)

        # schedule the task
        redis_db.hmset('task_to_user_id', {file_path: user_id})
        redis_db.lpush('scheduled_tasks', file_path)

        return render_template("upload_model_complete.html",
                               user_name=user_name)


@app.route('/show-leaderboard')
def show_leaderboard():
    all_keys = results_db.keys()
    results = []
    for key in all_keys:
        user_id = redis_db.hmget('task_to_user_id', key)[0]
        user_name = redis_db.hmget('ids_to_user', user_id)[0]
        operations, model_size, accuracy = results_db.hmget(
            key, ['operations', 'model_size', 'accuracy'])
        result = {
            'user_name': user_name,
            'user_id': user_id,
            # 'model': key.split('/')[-1],
            'model': key,
            'operations': int(operations),
            'model_size': int(model_size),
            'accuracy': int(accuracy)
        }
        results.append(result)
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return render_template("show_leaderboard.html", results=results)


@app.route('/show-user-submits/<int:user_id>')
def show_user_submits(user_id):
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0')
